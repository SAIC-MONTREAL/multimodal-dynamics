import os
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

from mmdyn.pytorch.utils.training import progress_bar, save_pkl
from mmdyn.pytorch.utils.datasets import dataset_setup
from mmdyn.pytorch.models.models import setup_model
from mmdyn.pytorch import config
from mmdyn.pytorch.utils.plots import plot_pose_tensorboard, plot_single_pose_tensorboard


class Problem:
    def __init__(self, problem_args, log_dir=None, load_dataset=None):
        self._model = None
        self._log_dir = None
        self._checkpoint_dir = None
        self._tensorboard_dir = None
        self._plot_dir = None
        self._condition_dim = None
        self._classes = None
        self._criterion = None
        self._optimizer = None
        self._writer = None
        self.train_dataset, self.test_dataset = None, None
        self.train_loader, self.test_loader = None, None

        self._best_acc = 0
        self._best_loss = np.inf
        self._load_dataset = load_dataset
        self._logger_dict = defaultdict(list)
        self._logger_histogram = defaultdict(list)
        self._img_logger_dict = defaultdict()
        self._fig_logger_dict = defaultdict()
        self.parameters = vars(problem_args)
        self._cross_modal = self.parameters['input_type'] == 'visuotactile'
        self._kl_weight = self.parameters['kl_weight']
        self._pose_multiplier = self.parameters['pose_multiplier']
        self._conditional = self.parameters['conditional']
        self._categorical_conditions = None
        self._seq_length = None

        self._device = torch.device('cuda' if torch.cuda.is_available() and not self.parameters['no_cuda'] else 'cpu')
        assert (self.parameters['input_type'] in config.INPUT_TYPES), "Input type is not implemented"

        if log_dir:
            self.load_dir(log_dir)
            self._load_problem()
        else:
            self.set_dir()
            self._set_problem()

    def _set_problem(self):
        self.set_dataset()
        self.set_model()
        self.set_criterion()
        self.set_optimizer()

    def _load_problem(self):
        if self._load_dataset:
            self.set_dataset()
            self.set_model()

    def set_model(self):
        raise NotImplementedError

    def _set_condition_dim(self):
        raise NotImplementedError

    def load_dir(self, log_dir):
        self._log_dir = log_dir
        self._checkpoint_dir = self._log_dir + '/checkpoint/'
        self._tensorboard_dir = self._log_dir + '/tensorboard/'
        self._plot_dir = self._log_dir + '/plot/'

    def set_dir(self):
        date = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        self._log_dir = './logs/' + self.parameters['save_name'] + date
        self._checkpoint_dir = self._log_dir + '/checkpoint/'
        self._tensorboard_dir = self._log_dir + '/tensorboard/'
        self._plot_dir = self._log_dir + '/plot/'
        Path(self._log_dir).mkdir(parents=True, exist_ok=True)
        Path(self._checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self._tensorboard_dir).mkdir(parents=True, exist_ok=True)
        Path(self._plot_dir).mkdir(parents=True, exist_ok=True)

    def parse_input(self, data, target):
        model_input = None
        if not isinstance(data, list):
            model_input = data.to(self._device)
        else:
            if self.parameters['input_type'] == 'visual':
                model_input = data[0].to(self._device)
            elif self.parameters['input_type'] == 'tactile':
                model_input = data[1].to(self._device)
            elif self.parameters['input_type'] == 'visuotactile':
                model_input = [data[0].to(self._device), data[1].to(self._device)]
        model_target = target.to(self._device)
        return model_input, model_target

    def set_dataset(self):
        self._input_size = (64, 64)
        self._n_channels = 3
        self.dataset_dict = dataset_setup(self.parameters['dataset_path'],
                                          self.parameters['problem_type'],
                                          input_size=self._input_size,
                                          batchsize=self.parameters['batchsize'],
                                          shuffle=True)
        self.train_dataset, self.test_dataset = self.dataset_dict['train_dataset'], self.dataset_dict['test_dataset']
        self.train_loader, self.test_loader = self.dataset_dict['train_loader'], self.dataset_dict['test_loader']
        self._seq_length = self.dataset_dict['seq_length']
        print(self._seq_length)
        if 'classes' in self.dataset_dict.keys():
            self._classes = self.dataset_dict['classes']

        print(len(self.train_dataset), len(self.test_dataset))

    def set_criterion(self):
        raise NotImplementedError

    def set_optimizer(self):
        assert (self.parameters['optimizer'] in config.OPTIMIZERS), "loss name not implemented in Problem"
        if self.parameters['optimizer'] == 'SGD':
            self._optimizer = optim.SGD(self._model.parameters(),
                                        lr=self.parameters['lr'],
                                        momentum=0.9,
                                        weight_decay=5e-4)
        elif self.parameters['optimizer'] == 'Adam':
            self._optimizer = optim.Adam(self._model.parameters(), lr=self.parameters['lr'])

    def _evaluate_model(self, inputs, targets, **kwargs):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        print('Epoch: %d' % epoch)
        self._model.train()
        train_loss, inputs, outputs, targets = 0, [], [], []
        perf_measure = {'visual': 0, 'tactile': 0, 'pose': 0}
        for batch_idx, (data_input, data_target) in enumerate(self.train_loader):
            inputs, targets = self.parse_input(data_input, data_target)
            self._optimizer.zero_grad()
            outputs, loss = self._evaluate_model(inputs, targets)

            loss.backward()

            self._optimizer.step()
            train_loss += loss.item()

            if 'perf_measure' in outputs:
                for k, v in outputs['perf_measure'].items():
                    perf_measure[k] += v

            update_steps = epoch * len(self.train_loader) + batch_idx

            self._writer.add_scalar('Loss/train_step', loss.item(), update_steps)

            progress_bar(batch_idx + 1, len(self.train_loader), 'Loss %.3f' % loss.item())

        self._log_train_info(inputs, outputs, targets, train_loss, epoch,
                             perf_measure=perf_measure)

        return perf_measure

    def _test_epoch(self, epoch):
        self._model.train()
        validation_loss = 0
        perf_measure = {'visual': 0, 'tactile': 0, 'pose': 0}
        with torch.no_grad():
            for batch_idx, (data_input, data_target) in enumerate(self.test_loader):
                inputs, targets = self.parse_input(data_input, data_target)
                outputs, loss = self._evaluate_model(inputs, targets)
                validation_loss += loss.item()

                if 'perf_measure' in outputs:
                    for k, v in outputs['perf_measure'].items():
                        perf_measure[k] += v

                progress_bar(batch_idx + 1, len(self.test_loader), 'Loss %.3f' % loss)

            self._log_test_info(inputs, outputs, targets, validation_loss, epoch, perf_measure=perf_measure)

        return perf_measure

    def train(self, save=True):
        perf_measure = 0
        self._writer = SummaryWriter(self._tensorboard_dir)
        for epoch in range(self.parameters['num_epochs']):
            self._anneal_KL(epoch)
            self._train_epoch(epoch)
            perf_measure = self._test_epoch(epoch)
            # sample from the latent space in the case of latent variable models
            self._sample(n=50)
            for key in self._logger_dict:
                self._writer.add_scalar(key, self._logger_dict[key][epoch], epoch)
            for key in self._logger_histogram:
                self._writer.add_histogram(key, self._logger_histogram[key], global_step=epoch)
            self._write_images(epoch)

        self._writer.add_hparams(self.parameters, perf_measure)
        if save:
            save_pkl(self._logger_dict, os.path.join(self._log_dir, 'results.pkl'))

    def _anneal_KL(self, epoch):
        if epoch < self.parameters['annealing_epochs']:
            self._kl_weight = (epoch + 1) / self.parameters['annealing_epochs']
        else:
            self._kl_weight = 1

    def _sample(self, n=50):
        raise NotImplementedError

    def _log_train_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None):
        raise NotImplementedError

    def _log_test_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None):
        raise NotImplementedError

    def _write_images(self, epoch, n_images=100):
        raise NotImplementedError

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def model(self):
        return self._model

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def plot_dir(self):
        return self._plot_dir

    @property
    def dataset(self, test=True):
        return self.test_dataset if test else self.train_dataset

    @property
    def num_epochs(self):
        return self.parameters['num_epochs']

    @property
    def input_type(self):
        return self.parameters['input_type']

    @property
    def condition_dim(self):
        return self._condition_dim


class Regression(Problem):
    """
    Baseline for regression the Pose from images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_model(self):
        self._set_condition_dim()

        model_kwargs = {}
        model_kwargs['condition_dim'] = self._condition_dim
        model_kwargs['out_dim'] = 7
        model_kwargs['conditional'] = self._conditional

        self._model = setup_model(self.parameters['model_name'], **model_kwargs)
        self._model.to(self._device)
        print(self._model)

    def _set_condition_dim(self):
        # condition_dim is really the shock force dimension
        self._categorical_conditions = False
        try:
            self._condition_dim = len(self.train_dataset.data[0][0][4])
        except:
            self._condition_dim = len(self.train_dataset.data[0][-1])

    def parse_input(self, data, target):
        """
        Pose is sent as the model input (conditions of the CVAE)
        """
        l = self._seq_length
        model_input, target_output = None, None
        if not isinstance(data, list):
            model_input = data.to(self._device)
            target_output = target.to(self._device)
        elif len(data) == 1:
            model_input = data[0].to(self._device)
            target_output = target[0].to(self._device)
        else:
            if self.parameters['input_type'] == 'visual':
                model_input = data[0][::l].to(self._device)
                target_output = target[2][::l].to(self._device)
            elif self.parameters['input_type'] == 'tactile':
                model_input = data[1][::l].to(self._device)
                target_output = target[2][::l].to(self._device)

        try:
            shock = data[4][::l].to(self._device)
        except:
            shock = None

        return {'model_input': model_input, 'shock': shock}, target_output

    def set_criterion(self):
        self._criterion = nn.MSELoss(reduction='sum')

    def _evaluate_model(self, inputs, targets, **kwargs):
        if self._conditional:
            out = self._model(inputs['model_input'], inputs['shock'])
        else:
            out = self._model(inputs['model_input'])
        loss = self._criterion(out.view(targets.size()), targets)

        with torch.no_grad():
            pose_measure = F.mse_loss(out.view(targets.size()), targets, reduction='mean')
        outputs = {'outputs': out, 'perf_measure': {'pose': pose_measure}}
        return outputs, loss

    def _sample(self, n=50):
        pass

    def _log_train_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None):
        self._logger_dict['Loss/train_epoch'].append(loss / len(self.train_loader))

        if perf_measure:
            for k, v in perf_measure.items():
                self._logger_dict['Perf_measure_train/' + k].append(v / len(self.train_loader))

    def _log_test_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None):
        self._logger_dict['Loss/validation_epoch'].append(loss / len(self.test_loader))

        if perf_measure:
            for k, v in perf_measure.items():
                self._logger_dict['Perf_measure_validation/' + k].append(v / len(self.test_loader))

        if loss < self._best_loss:
            state = {'model': self._model.state_dict(),
                     'loss': loss,
                     'epoch': epoch,
                     }
            torch.save(state, self._checkpoint_dir + '/epoch_' + str(epoch) + '.ckpt')
            self._best_loss = loss

    def _write_images(self, epoch, n_images=100):
        pass


class Reconstruction(Problem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_model(self):
        self._set_condition_dim()

        model_kwargs = {}
        model_kwargs['condition_dim'] = self._condition_dim
        model_kwargs['input_dim'] = np.prod(np.array(self._input_size))
        model_kwargs['architecture'] = self.parameters['model_name'].split('-')[0]
        model_kwargs['conditional'] = self._conditional
        model_kwargs['categorical_conditions'] = self._categorical_conditions
        try:
            model_kwargs['latent_size'] = self.parameters['latent_size']
        except:
            model_kwargs['latent_size'] = 256
        if 'mvae' in self.parameters['model_name']:
            model_kwargs['use_pose'] = self.parameters['use_pose']
        if 'transformer' in self.parameters['model_name']:
            model_kwargs['seq_length'] = self._seq_length

        self._model = setup_model(self.parameters['model_name'],
                                  cross_modal=self._cross_modal,
                                  **model_kwargs)
        self._model.to(self._device)
        print(self._model)

    def _set_condition_dim(self):
        self._categorical_conditions = True
        self._condition_dim = np.max(np.array(self.train_dataset.targets)).item() + 1

    def set_criterion(self):
        if 'mvae' in self.parameters['model_name']:
            self._criterion = self._mvae_elbo_loss
        else:
            self._criterion = self._elbo_loss

    def _elbo_loss(self, recon_x, x, means, log_var, loss_mask=None, reduce=None, reduction='sum'):
        """
        Use this with VAE and CVAE.
        """
        batch_size = x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())

        if loss_mask is not None:
            BCE = F.binary_cross_entropy_with_logits(torch.mul(recon_x.view(x.size()), loss_mask),
                                                     torch.mul(x, loss_mask),
                                                     reduce=reduce, reduction=reduction)
        else:
            BCE = F.binary_cross_entropy_with_logits(recon_x.view(x.size()), x, reduce=reduce, reduction=reduction)

        if reduce is not None:
            BCE = torch.sum(BCE, (1, 2, 3))
            return BCE + self._kl_weight * KLD

        return (BCE + self._kl_weight * KLD) / batch_size

    def _mvae_elbo_loss(self, recon_x, x, means, log_var, loss_mask=None, reduce=None, reduction='sum'):
        """
        Use this only with MVAE models.
        """
        assert len(recon_x) == len(x)

        batch_size = x[0].size(0)
        recon_error = 0
        kl_divergence = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())

        for i in range(len(recon_x)):
            # dimensionality > 1 implies an image
            if len(recon_x[i].size()) > 2:
                recon_i = recon_x[i].view(x[i].size())
                loss_fn = F.binary_cross_entropy_with_logits
                loss_multiplier = 1
                sum_dims = (1, 2, 3)
            else:
                # for non image reconstructions (e.g. pose) use MSE
                recon_i = recon_x[i]
                loss_fn = F.mse_loss
                loss_multiplier = self._pose_multiplier
                sum_dims = (1)

            if loss_mask is not None:
                e = loss_multiplier * loss_fn(torch.mul(recon_i, loss_mask), torch.mul(x[i], loss_mask),
                                              reduce=reduce, reduction=reduction)
            else:
                e = loss_multiplier * loss_fn(recon_i, x[i], reduce=reduce, reduction=reduction)

            if reduce is not None:
                e = torch.sum(e, sum_dims)
            recon_error += e

        if reduce is not None:
            return recon_error + self._kl_weight * kl_divergence

        return (recon_error + self._kl_weight * kl_divergence) / batch_size

    def _evaluate_model(self, x, targets, **kwargs):
        if 'mvae' in self.parameters['model_name']:
            return self._evaluate_mvae(x=x, targets=x)
        elif self._conditional:
            # targets here is really the conditions (class labels)
            recon_x, means, log_var = self._model(x, targets)
        else:
            recon_x, means, log_var = self._model(x)

        loss = self._criterion(recon_x, x, means, log_var)
        outputs = {'recon_x': recon_x, 'means': means, 'log_var': log_var}
        return outputs, loss

    def _evaluate_mvae(self, x, targets, loss_mask=None, reduce=None, reduction='sum', condition=None):
        assert isinstance(x, list) and isinstance(targets, list)
        # loss = 0

        # joint visual and tactile images
        visual_recon_joint, tactile_recon_joint, _, means, log_var = self._model([x[0], x[1]], condition=condition)
        loss = self._mvae_elbo_loss([visual_recon_joint, tactile_recon_joint], [targets[0], targets[1]],
                                    means, log_var, loss_mask=loss_mask, reduce=reduce, reduction=reduction)

        # visual image only
        visual_recon, _, _, means, log_var = self._model([x[0], None], condition=condition)

        visual_loss = self._mvae_elbo_loss([visual_recon], [targets[0]], means, log_var, loss_mask=loss_mask,
                                           reduce=reduce, reduction=reduction)
        loss += visual_loss

        # tactile image only
        _, tactile_recon, _, means, log_var = self._model([None, x[1]], condition=condition)
        tactile_loss = self._mvae_elbo_loss([tactile_recon], [targets[1]], means, log_var, loss_mask=loss_mask,
                                            reduce=reduce, reduction=reduction)
        loss += tactile_loss

        # log visual and tactile losses
        # self._logger_dict['Modality_loss/visual'].append(visual_loss.item())
        # self._logger_dict['Modality_loss/tactile'].append(tactile_loss.item())

        with torch.no_grad():
            visual_recon_measure = F.binary_cross_entropy_with_logits(visual_recon.view(targets[0].size()), targets[0],
                                                                      reduction='mean')
            tactile_recon_measure = F.binary_cross_entropy_with_logits(tactile_recon.view(targets[1].size()), targets[1],
                                                                       reduction='mean')

        if self.parameters['use_pose']:
            # visual, tactile, and pose
            visual_recon_joint, tactile_recon_joint, pose_recon_joint, means, log_var = self._model([x[0], x[1]],
                                                                                                    pose=x[2],
                                                                                                    condition=condition)
            loss += self._mvae_elbo_loss([visual_recon_joint, tactile_recon_joint, pose_recon_joint],
                                         [targets[0], targets[1], targets[2]], means, log_var, loss_mask=loss_mask,
                                         reduce=reduce, reduction=reduction)

            # pose and visual image
            visual_recon, _, pose_recon, means, log_var = self._model([x[0], None], pose=x[2], condition=condition)
            loss += self._mvae_elbo_loss([visual_recon, pose_recon], [targets[0], targets[2]],
                                         means, log_var, loss_mask=loss_mask, reduce=reduce, reduction=reduction)

            # pose and tactile image
            _, tactile_recon, pose_recon, means, log_var = self._model([None, x[1]], pose=x[2], condition=condition)
            loss += self._mvae_elbo_loss([tactile_recon, pose_recon], [targets[1], targets[2]],
                                         means, log_var, loss_mask=loss_mask, reduce=reduce, reduction=reduction)

            # pose only
            _, _, pose_recon, means, log_var = self._model([None, None], pose=x[2], condition=condition)

            pose_loss = self._mvae_elbo_loss([pose_recon], [targets[2]], means, log_var, loss_mask=loss_mask,
                                             reduce=reduce, reduction=reduction)
            loss += pose_loss

            # log pose loss
            # self._logger_dict['Modality_loss/pose'].append(pose_loss.item())

            with torch.no_grad():
                pose_recon_measure = F.mse_loss(pose_recon.view(targets[2].size()), targets[2], reduction='mean')

            outputs = {'recon_x': [visual_recon_joint, tactile_recon_joint, pose_recon_joint], 'means': means,
                       'log_var': log_var, 'perf_measure': {'visual': visual_recon_measure.item(),
                                                            'tactile': tactile_recon_measure.item(),
                                                            'pose': pose_recon_measure.item()}}

        else:
            outputs = {'recon_x': [visual_recon_joint, tactile_recon_joint], 'means': means, 'log_var': log_var,
                       'perf_measure': {'visual': visual_recon_measure.item(), 'tactile': tactile_recon_measure.item()}}

        return outputs, loss

    def _sample(self, n=50):
        with torch.no_grad():
            if self._conditional:
                if self._categorical_conditions:
                    y = torch.randint(0, self._condition_dim, (n,)).unsqueeze(1).to(self._device)
                else:
                    y = torch.rand((n, self._condition_dim)).to(self._device)
                x = self._model.inference(n=n, c=y)
            else:
                x = self._model.inference(n=n)

            self._img_logger_dict['Samples/latent_space'] = self.apply_sigmoid(x)

    def _log_train_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None):
        self._logger_dict['Loss/train_epoch'].append(loss / len(self.train_loader))
        self._logger_dict['KL_annealing/train_epoch'].append(self._kl_weight)
        self._img_logger_dict['Input_img/train'] = inputs
        self._img_logger_dict['Output_img/train'] = self.apply_sigmoid(outputs['recon_x'])

        if perf_measure:
            for k, v in perf_measure.items():
                self._logger_dict['Perf_measure_train/' + k].append(v / len(self.train_loader))

    def _log_test_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None):
        self._logger_dict['Loss/validation_epoch'].append(loss / len(self.test_loader))
        self._img_logger_dict['Input_img/validation'] = inputs
        self._img_logger_dict['Output_img/validation'] = self.apply_sigmoid(outputs['recon_x'])

        if perf_measure:
            for k, v in perf_measure.items():
                self._logger_dict['Perf_measure_validation/' + k].append(v / len(self.test_loader))

        if loss < self._best_loss:
            state = {'model': self._model.state_dict(),
                     'loss': loss,
                     'epoch': epoch,
                     }
            torch.save(state, self._checkpoint_dir + '/epoch_' + str(epoch) + '.ckpt')
            self._best_loss = loss

    def _write_images(self, epoch, n_images=120):
        nrow = self._seq_length if (self._seq_length > 1 and 'sv' not in self.parameters['dataset_path']) else int(math.sqrt(self.parameters['batchsize']))
        if 'modeling' in self.parameters['problem_type'] and 'sv' not in self.parameters['dataset_path']:
            n_images = min(self.parameters['batchsize'] * self._seq_length, n_images)
        else:
            n_images = min(self.parameters['batchsize'], n_images)
        for key, v in self._img_logger_dict.items():
            if self.parameters['model_name'] == 'mlp-vae':
                v = v.view(-1, self._n_channels, *self._input_size)
            if isinstance(v, list) or isinstance(v, tuple):
                # cross modal case
                img = torch.cat((v[0][:n_images, :, :, :], v[1][:n_images, :, :, :]), dim=0)
                img_grid = torchvision.utils.make_grid(img, nrow=nrow)
            else:
                img_grid = torchvision.utils.make_grid(v[:n_images, :, :, :], nrow=nrow)
            self._writer.add_image(key, img_grid, global_step=epoch)

        for key, v in self._fig_logger_dict.items():
            if isinstance(v, list):
                self._writer.add_figure(key, plot_pose_tensorboard(pose_1=v[0][:n_images, :].cpu().detach().numpy(),
                                                                   pose_2=v[1][:n_images, :].cpu().detach().numpy(),
                                                                   seq_length=self._seq_length),
                                        global_step=epoch)
            else:
                self._writer.add_figure(key, plot_single_pose_tensorboard(pose=v[:n_images, :].cpu().detach().numpy(),
                                                                          seq_length=self._seq_length),
                                        global_step=epoch)

    def apply_sigmoid(self, img):
        """
        Applies Sigmoid to the output of the network.
        Only use this for visualization of the images.
        """
        with torch.no_grad():
            if isinstance(img, list) or isinstance(img, tuple):
                img = [F.sigmoid(x) for x in img]
            else:
                img = F.sigmoid(img)
        return img


class SeqModeling(Reconstruction, Problem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_input(self, data, target):
        """
        Pose is sent as the model input (conditions of the CVAE)
        """
        l = self._seq_length
        model_input, target_output = None, None
        if not isinstance(data, list):
            model_input = data.to(self._device)
            target_output = target.to(self._device)
        elif len(data) == 1:
            model_input = data[0].to(self._device)
            target_output = target[0].to(self._device)
        else:
            if self.parameters['input_type'] == 'visual':
                model_input = data[0][::l].to(self._device)
                target_output = target[0][::l].to(self._device)
            elif self.parameters['input_type'] == 'tactile':
                model_input = data[1][::l].to(self._device)
                target_output = target[1][::l].to(self._device)
            elif self.parameters['input_type'] == 'visuotactile':
                model_input = [data[0][::l].to(self._device), data[1][::l].to(self._device)]
                target_output = [target[0][::l].to(self._device), target[1][::l].to(self._device)]

        if len(data) > 2:
            input_object_pose = [data[2][::l].to(self._device)]
            input_available_modals = data[3][::l].to(self._device)

            target_object_pose = [target[2][::l].to(self._device)]
            loss_mask = target[3][::l].to(self._device)

            try:
                shock = data[4][::l].to(self._device)
            except:
                shock = None
        else:
            input_object_pose, input_available_modals, target_object_pose, loss_mask, shock = None, None, None, None, None

        return {'model_input': model_input, 'input_object_pose': input_object_pose,
                'input_available_modals': input_available_modals, 'shock': shock}, \
               {'target_output': target_output, 'target_object_pose': target_object_pose, 'loss_mask': loss_mask}

    def _set_condition_dim(self):
        # condition_dim is really the shock force dimension
        self._categorical_conditions = False
        try:
            self._condition_dim = len(self.train_dataset.data[0][0][4])
        except:
            self._condition_dim = len(self.train_dataset.data[0][-1])

    def _evaluate_model(self, x, targets, reduction='sum', reduce=None, **kwargs):
        loss_mask = targets['loss_mask'] if self.parameters['mask_loss'] else None

        # quick hack to make the code backward compatible
        if 'shock' not in x.keys():
            x['shock'] = None

        if 'mvae' in self.parameters['model_name']:
            if self.parameters['use_pose']:
                return self._evaluate_mvae(x=x['model_input'] + x['input_object_pose'],
                                           targets=targets['target_output'] + targets['target_object_pose'],
                                           loss_mask=loss_mask, reduce=reduce, reduction=reduction,
                                           condition=x['shock'])
            else:
                return self._evaluate_mvae(x=x['model_input'],
                                           targets=targets['target_output'],
                                           loss_mask=loss_mask, reduce=reduce, reduction=reduction,
                                           condition=x['shock'])

        elif self._conditional:
            recon_x, means, log_var = self._model(x['model_input'], x['shock'])
        else:
            recon_x, means, log_var = self._model(x['model_input'])

        loss = self._elbo_loss(recon_x, targets['target_output'], means, log_var, loss_mask=loss_mask,
                               reduce=reduce, reduction=reduction)

        with torch.no_grad():
            recon_measure = F.binary_cross_entropy_with_logits(recon_x.view(targets['target_output'].size()),
                                                               targets['target_output'], reduction='mean')

        outputs = {'recon_x': recon_x, 'means': means, 'log_var': log_var,
                   'perf_measure': {self.parameters['input_type']: recon_measure.item()}}
        return outputs, loss

    def _log_train_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None, log_pose=False):
        self._logger_dict['Loss/train_epoch'].append(loss / len(self.train_loader))
        self._logger_dict['KL_annealing/train_epoch'].append(self._kl_weight)
        self._img_logger_dict['Input_img/train'] = inputs['model_input']
        self._img_logger_dict['Output_img/train'] = self.apply_sigmoid(outputs['recon_x'])
        self._img_logger_dict['Target_img/train'] = targets['target_output']
        # self._img_logger_dict['Loss_mask/train'] = targets['loss_mask']

        if perf_measure:
            for k, v in perf_measure.items():
                self._logger_dict['Perf_measure_train/' + k].append(v / len(self.train_loader))

        if self.parameters['use_pose'] and log_pose:
            self._fig_logger_dict['Pose_train/input'] = inputs['input_object_pose'][0]
            self._fig_logger_dict['Pose_train/output_vs_target'] = [outputs['recon_x'][2],
                                                                    targets['target_object_pose'][0]]

    def _log_test_info(self, inputs, outputs, targets, loss, epoch, perf_measure=None, log_pose=False):
        self._logger_dict['Loss/validation_epoch'].append(loss / len(self.test_loader))
        self._img_logger_dict['Input_img/validation'] = inputs['model_input']
        self._img_logger_dict['Output_img/validation'] = self.apply_sigmoid(outputs['recon_x'])
        self._img_logger_dict['Target_img/validation'] = targets['target_output']
        # self._img_logger_dict['Loss_mask/validation'] = targets['loss_mask']

        if perf_measure:
            for k, v in perf_measure.items():
                self._logger_dict['Perf_measure_validation/' + k].append(v / len(self.test_loader))

        if self.parameters['use_pose'] and log_pose:
            self._fig_logger_dict['Pose_validation/input'] = inputs['input_object_pose'][0]
            self._fig_logger_dict['Pose_validation/output_vs_target'] = [outputs['recon_x'][2],
                                                                         targets['target_object_pose'][0]]

        if loss < self._best_loss:
            state = {'model': self._model.state_dict(),
                     'loss': loss,
                     'epoch': epoch,
                     }
            torch.save(state, self._checkpoint_dir + '/epoch_' + str(epoch) + '.ckpt')
            self._best_loss = loss


class DynModeling(SeqModeling):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_input(self, data, target):
        """
        Shifts the inputs to act as a one-step dynamics model.
        """
        model_input, target_output = None, None
        if not isinstance(data, list):
            model_input = data.to(self._device)
        else:
            l = self._seq_length
            # set the last token in the sequence as the final target image
            if self.parameters['input_type'] == 'visual':
                model_input = data[0].to(self._device)
                target_output = torch.roll(data[0], -1, dims=0).to(self._device)
                target_output[l-1::l] = target[0][l-1::l].to(self._device)
            elif self.parameters['input_type'] == 'tactile':
                model_input = data[1].to(self._device)
                target_output = torch.roll(data[1], -1, dims=0).to(self._device)
                target_output[l-1::l] = target[1][l-1::l].to(self._device)
            elif self.parameters['input_type'] == 'visuotactile':
                model_input = [data[0].to(self._device), data[1].to(self._device)]
                target_output = [torch.roll(data[0], -1, dims=0).to(self._device),
                                 torch.roll(data[1], -1, dims=0).to(self._device)]
                target_output[0][l-1::l] = target[0][l-1::l].to(self._device)
                target_output[1][l-1::l] = target[1][l-1::l].to(self._device)

        input_object_pose = [data[2].to(self._device)]
        input_available_modals = data[3].to(self._device)

        try:
            shock = data[4].to(self._device)
        except:
            shock = None

        target_object_pose = [torch.roll(data[2], -1, dims=0).to(self._device)]
        loss_mask = target[3].to(self._device)

        return {'model_input': model_input, 'input_object_pose': input_object_pose,
                'input_available_modals': input_available_modals, 'shock': shock}, \
               {'target_output': target_output, 'target_object_pose': target_object_pose, 'loss_mask': loss_mask}
