"""
Variational Autoencoder (VAE) and Multimodal VAE (MVAE).
Code for MVAE from https://github.com/mhw32/multimodal-vae-public
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from mmdyn.pytorch import config


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Autoencoder(nn.Module):
    """
    Base class for Autoencoders.
    """

    def __init__(self, input_dim=784, encoder_hid=[256, 256], latent_size=8,
                 decoder_hid=[256, 256], condition_dim=None, architecture='mlp',
                 conditional=False, categorical_conditions=False):
        """
        inputs to 'mlp':           (n, input_dim)
        inputs to 'cnn':           (n, 3, 64, 64)
        """
        super().__init__()

        assert type(encoder_hid) == list
        assert type(latent_size) == int
        assert type(decoder_hid) == list
        assert architecture in config.ARCHITECTURES

        self.latent_size = latent_size
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.architecture = architecture
        self.conditional = conditional
        self.categorical_conditions = categorical_conditions

    def reparametrize(self, means, log_var):
        batch_size = means.size(0)
        is_cuda = next(self.parameters()).is_cuda
        device = torch.device('cuda') if is_cuda else torch.device('cpu')

        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        z = eps * std + means

        return z

    def forward(self, x):
        raise NotImplementedError

    def inference(self, n=1):
        raise NotImplementedError


class VAE(Autoencoder):
    """
    Vanilla VAE.
    """

    def __init__(self, use_pose=False, **kwargs):
        super().__init__(**kwargs)

        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, x, c=None):
        if x.dim() > 2 and self.architecture == 'mlp':
            x = x.view(-1, self.input_dim)
        means, log_var = self.encoder(x, c)
        z = self.reparametrize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var

    def inference(self, n=1, c=None):
        batch_size = n
        is_cuda = next(self.parameters()).is_cuda
        device = torch.device('cuda') if is_cuda else torch.device('cpu')

        z = torch.randn([batch_size, self.latent_size]).to(device)
        recon_x = self.decoder(z, c)

        return recon_x


class MVAE(Autoencoder):
    """
    Implementation of 'Multimodal Generative Models for Scalable Weakly-Supervised Learning'
    https://arxiv.org/abs/1802.05335
    """

    def __init__(self, use_pose=False, **kwargs):
        super().__init__(**kwargs)

        assert kwargs['architecture'] != 'mlp', "MVAE is not implemented with MLP"
        self._use_pose = use_pose

        self.visual_encoder = Encoder(**kwargs)
        self.visual_decoder = Decoder(**kwargs)
        self.tactile_encoder = Encoder(**kwargs)
        self.tactile_decoder = Decoder(**kwargs)
        if self._use_pose:
            self.pose_encoder = Encoder(input_dim=7, layer_sizes=[512, 512],
                                        latent_size=kwargs["latent_size"],
                                        condition_dim=0, architecture="mlp")
            self.pose_decoder = Decoder(output_dim=7, layer_sizes=[512, 512],
                                        latent_size=kwargs["latent_size"],
                                        condition_dim=0, architecture="mlp")
        self.experts = ProductOfExperts()

    def forward(self, x, pose=None, condition=None):
        assert isinstance(x, list) or isinstance(x, tuple)
        visual, tactile = x
        if visual is not None:
            batch_size = visual.size(0)
        elif tactile is not None:
            batch_size = tactile.size(0)
        else:
            batch_size = pose.size(0)
        is_cuda = next(self.parameters()).is_cuda
        device = torch.device('cuda') if is_cuda else torch.device('cpu')

        # initialize the universal prior expert
        means, logvar = prior_expert((1, batch_size, self.latent_size), device=device)

        if visual is not None:
            visual_means, visual_log_var = self.visual_encoder(visual, c=condition)
            means = torch.cat((means, visual_means.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, visual_log_var.unsqueeze(0)), dim=0)

        if tactile is not None:
            tactile_means, tactile_log_var = self.tactile_encoder(tactile, c=condition)
            means = torch.cat((means, tactile_means.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, tactile_log_var.unsqueeze(0)), dim=0)

        if pose is not None and self._use_pose:
            pose_means, pose_log_var = self.pose_encoder(pose, c=condition)
            means = torch.cat((means, pose_means.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, pose_log_var.unsqueeze(0)), dim=0)

        # product of experts to combine Gaussians
        means, log_var = self.experts(means, logvar)

        z = self.reparametrize(means, log_var)
        visual_recon = self.visual_decoder(z, c=condition)
        tactile_recon = self.tactile_decoder(z, c=condition)

        pose_recon = self.pose_decoder(z, c=condition) if self._use_pose else None

        return visual_recon, tactile_recon, pose_recon, means, log_var

    def inference(self, n=1, c=None):
        batch_size = n
        is_cuda = next(self.parameters()).is_cuda
        device = torch.device('cuda') if is_cuda else torch.device('cpu')

        z = torch.randn([batch_size, self.latent_size]).to(device)
        visual_recon = self.visual_decoder(z, c)
        tactile_recon = self.tactile_decoder(z, c)

        return visual_recon, tactile_recon


class Encoder(nn.Module):

    def __init__(self, input_dim=784, layer_sizes=[256, 256], latent_size=8,
                 architecture='mlp', conditional=False, categorical_conditions=False,
                 condition_dim=None, **kwargs):
        super().__init__()

        self.architecture = architecture
        self.conditional = conditional
        self.categorical_conditions = categorical_conditions
        self.condition_dim = condition_dim
        if categorical_conditions:
            assert condition_dim is not None, "Num conditions is not specified for categorical conditions."

        if architecture == 'cnn':
            # DCGAN style
            cnn_features_out = 256 * 5 * 5
            cnn_features_comp = 512 + self.conditional * self.condition_dim
            self.conv_net = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1, bias=False),
                Swish(),
                nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                Swish(),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                Swish(),
                nn.Conv2d(128, 256, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                Swish()
            )
            self.fc_net = nn.Sequential(
                nn.Linear(cnn_features_out, 512),
                Swish(),
                nn.Dropout(p=0.1),
            )
            self.linear_means = nn.Linear(cnn_features_comp, latent_size)
            self.linear_log_var = nn.Linear(cnn_features_comp, latent_size)

        else:
            layer_sizes = [input_dim] + layer_sizes
            self.fc_net = mlp(layer_sizes, nn.ReLU, nn.Identity)
            self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
            self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.architecture != 'mlp':
            x = self.conv_net(x)
            x = x.view(x.size(0), -1)

        x = self.fc_net(x)

        if self.conditional:
            if self.categorical_conditions:
                c = idx2onehot(c, n=self.condition_dim)
            elif c.dim() == 1:
                c = c.unsqueeze(1)

            x = torch.cat((x, c.float()), dim=-1)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, output_dim=784, layer_sizes=[256, 256], latent_size=2,
                 architecture='mlp', conditional=False, categorical_conditions=False,
                 condition_dim=None, **kwargs):

        super().__init__()

        self.architecture = architecture
        self.conditional = conditional
        self.categorical_conditions = categorical_conditions
        self.condition_dim = condition_dim
        latent_size = latent_size + self.conditional * self.condition_dim
        if categorical_conditions:
            assert condition_dim is not None, "Num conditions is not specified for categorical conditions."

        if architecture == 'cnn':
            # DCGAN style
            self.upsample = nn.Sequential(
                nn.Linear(latent_size, 256 * 5 * 5),
                Swish()
            )
            self.hallucinate = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
                nn.BatchNorm2d(128),
                Swish(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                Swish(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                Swish(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
                # nn.Sigmoid()
            )

        else:
            layer_sizes = [latent_size] + layer_sizes + [output_dim]
            self.deconv_net = mlp(layer_sizes, nn.ReLU, nn.Identity)

    def forward(self, z, c=None):
        if self.conditional:
            if self.categorical_conditions:
                c = idx2onehot(c, n=self.condition_dim)
            elif c.dim() == 1:
                c = c.unsqueeze(1)
            z = torch.cat((z, c.float()), dim=-1)

        if self.architecture == 'cnn':
            h1 = self.upsample(z)
            deconv_input = h1.view(-1, 256, 5, 5)
            x = self.hallucinate(deconv_input)

        else:
            x = self.deconv_net(z)

        return x


class ProductOfExperts(nn.Module):

    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    param mu                : M x D for M experts
    param logvar            : M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


def prior_expert(size, device=torch.device('cpu')):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    param size (int)        : dimensionality of Gaussian
    """
    mu = Variable(torch.zeros(size)).to(device)
    logvar = Variable(torch.zeros(size)).to(device)
    return mu, logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    return onehot
