import os
import random
import cv2
import copy
import numpy as np
import pickle
import json

from PIL import ImageChops, Image
from collections import defaultdict
from pathlib import Path

import torchvision
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def dataset_setup(dataset_path, problem_type, **kwargs):
    collate_fn = None
    print("Loading dataset from {}".format(dataset_path))
    transform_train = transforms.Compose([
        torchvision.transforms.Resize(kwargs['input_size']),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        torchvision.transforms.Resize(kwargs['input_size']),
        transforms.ToTensor(),
    ])

    # Load dataset
    train_dataset = VisuoTactileDataset(train=True,
                                        transform=transform_train,
                                        dataset_path=dataset_path,
                                        )
    test_dataset = VisuoTactileDataset(train=False,
                                       transform=transform_test,
                                       dataset_path=dataset_path,
                                       )
    if 'seq' in problem_type:
        collate_fn = seq_collate_fn

    # Define data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=kwargs['batchsize'],
                                               collate_fn=collate_fn,
                                               drop_last=True,
                                               shuffle=kwargs['shuffle'])

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=kwargs['batchsize'],
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              shuffle=False)
    out_dict = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'seq_length': train_dataset.seq_length
    }
    if hasattr(train_dataset, 'classes'):
        out_dict['classes'] = train_dataset.classes
    return out_dict


class VisuoTactileDataset(VisionDataset):
    """Dataset manager for visuo-tactile datasets"""

    def __init__(self, train=True, transform=None, dataset_path=None,
                 real_dataset=False, train_frac=0.8,
                 compiled_name='compiled_dataset_array',
                 background_subtraction=False):
        self._train_frac = train_frac
        self.transform = transform
        self.train = train
        self.targets = None
        self.seq_length = None
        self._compiled_name = compiled_name
        self._background_subtraction = background_subtraction
        self.__get_dataset__(dataset_path=dataset_path, real_dataset=real_dataset)

    def __get_dataset_dir__(self, dataset_path):
        self.root = os.path.expanduser(dataset_path)

    def __get_dataset__(self, dataset_path, real_dataset=False):
        """
        Define dataset inputs and outputs
        """
        self.__get_dataset_dir__(dataset_path)
        self.dataset_path = os.path.join(self.root, self._compiled_name + ".pickle")

        if not os.path.exists(self.dataset_path):
            self._generate_object_seq(real_dataset, sv='sv' in dataset_path)
        datapoint_dict = self._load_dataset()

        len_dataset = len(datapoint_dict['targets'])
        frac_index = int(self._train_frac * len_dataset)
        if 'classes' in datapoint_dict.keys():
            self.classes = datapoint_dict['classes']
        if self.train:
            self.data = datapoint_dict['data'][0:frac_index]
            self.targets = datapoint_dict['targets'][0:frac_index]
        else:
            self.data = datapoint_dict['data'][frac_index:-1]
            self.targets = datapoint_dict['targets'][frac_index:-1]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.targets)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Sahand's hack to accommodate all needs across reconstruction, classification and dynamics models
        data = self.data[index]
        target = self.targets[index]

        if self.transform is None:
            return data, target

        else:
            # data: anything that goes into the model as an input
            if isinstance(data, list):
                # nested list infers sequential data
                if any(isinstance(i, list) for i in data):
                    data_tmp = []
                    for x in data:
                        data_tmp.append(self._parse_list_data(x))
                    data_tmp = list(map(list, zip(*data_tmp)))
                    data_tmp = [torch.stack(data_tmp[i]) for i in range(len(data_tmp))]

                else:
                    data_tmp = self._parse_list_data(data)
            else:
                if isinstance(data, np.ndarray):
                    data = Image.fromarray(data)
                data_tmp = self.transform(data)

            # target: anything that is used to train the model as its targets
            if isinstance(target, list):
                # nested list infers sequential data
                if any(isinstance(i, list) for i in target):
                    target_tmp = []
                    for x in target:
                        target_tmp.append(self._parse_list_data(x))
                    target_tmp = list(map(list, zip(*target_tmp)))
                    target_tmp = [torch.stack(target_tmp[i]) for i in range(len(target_tmp))]
                else:
                    target_tmp = self._parse_list_data(target)
            else:
                if isinstance(target, np.ndarray):
                    target = Image.fromarray(target)
                target_tmp = self.transform(target)

        return data_tmp, target_tmp

    def _generate_object_seq(self, real_dataset=False, sv=False):
        """
        Generate dataset for a sequence of images. The each data entry contains:
            data: [visual_img, tactile_img] of time step k in sequence i
            data: [visual_img, tactile_img] of time step k in sequence i
            targets: [visual_img, tactile_img] of the last time step in the sequence i
            pose: [position, orientation] of the object in time step k in sequence i
        """
        datapoint_dict = defaultdict(list)
        seq_data, seq_targets = [], []

        if not real_dataset:
            root = Path(self.root).joinpath("dataset")
            tactile_images = sorted(root.glob('**/tactile_*.png'))
            visual_images = sorted(root.glob('**/visual_*.png'))
            seg_images = sorted(root.glob('**/seg_*.png'))
            data = sorted(root.glob('**/data.json'))
            seq_length = int(len(visual_images) / len(data))
            final_visual_images = sorted(root.glob('**/visual_' + str(seq_length - 1).zfill(4) + '.png'))
            final_tactile_images = sorted(root.glob('**/tactile_' + str(seq_length - 1).zfill(4) + '.png'))
            final_seg_images = sorted(root.glob('**/seg_' + str(seq_length - 1).zfill(4) + '.png'))
            self.seq_length = seq_length

            print("Visual images: {}, Tactile images: {}, Sequences: {}, Sequence length: {}".format(
                len(visual_images), len(tactile_images), len(data), seq_length
            ))

            # compute pose min and max, shock min and max
            pose_list, shock_list = [], []
            for d in data:
                with open(str(d)) as f:
                    info = json.load(f)
                pose_list.append(np.concatenate((info['position'], info['orientation']), axis=1))
                if 'shock' in info.keys():
                    shock_list.append(np.array(info['shock']))
                else:
                    shock_list.append(np.zeros(1))

            pose_list, shock_list = np.concatenate(pose_list, axis=0), np.concatenate(shock_list, axis=0)

            pose_min, pose_max = np.min(pose_list, axis=0), np.max(pose_list, axis=0)
            shock_min, shock_max = np.min(shock_list, axis=0), np.max(shock_list, axis=0)

            # override for quaternions
            pose_min[3:] = np.array([-1, -1, -1, -1])
            pose_max[3:] = np.array([1, 1, 1, 1])

            for i, (visual_img, tactile_img, seg_img) in enumerate(zip(visual_images, tactile_images, seg_images)):
                seq_counter = i // seq_length
                time_step_in_seq = i % seq_length

                if time_step_in_seq == 0:
                    # save the previous sequence
                    if seq_counter != 0:
                        if sv:
                            for i in range(seq_length // 5):
                                tmp_data = copy.copy(seq_data)
                                tmp_targets = copy.copy(seq_targets)
                                tmp_data[i] = seq_data[i]
                                tmp_targets[i] = seq_targets[i]
                                datapoint_dict['data'].append(tmp_data)
                                datapoint_dict['targets'].append(tmp_targets)
                        else:
                            datapoint_dict['data'].append(seq_data)
                            datapoint_dict['targets'].append(seq_targets)
                        seq_data, seq_targets = [], []

                    print("Sequence #{}".format(seq_counter))
                    with open(str(data[seq_counter])) as f:
                        info = json.load(f)
                    final_seg_img_np = self._load_image(final_seg_images[seq_counter], resize=False)
                    bbox = self._bounding_box(final_seg_img_np)
                    final_visual_img_np = self._load_image(final_visual_images[seq_counter], bounding_box=bbox)
                    final_tactile_img_np = self._load_image(final_tactile_images[seq_counter], bounding_box=bbox)
                    final_pose = np.concatenate((info['position'][-1], info['orientation'][-1]))
                    final_pose = normalize(final_pose, pose_min, pose_max)

                seg_img_np_original = self._load_image(seg_img, resize=False)
                bbox = self._bounding_box(seg_img_np_original)

                seg_img_np = self._load_image(seg_img, bounding_box=bbox)
                seg_img_np = np.where(seg_img_np == 1, 0, seg_img_np)

                visual_img_np = self._load_image(visual_img, bounding_box=bbox)
                tactile_img_np = self._load_image(tactile_img, bounding_box=bbox)
                pose = np.concatenate((info['position'][time_step_in_seq], info['orientation'][time_step_in_seq]))
                pose = normalize(pose, pose_min, pose_max)

                visual_std = np.std(visual_img_np, axis=(0, 1))
                tactile_std = np.std(tactile_img_np, axis=(0, 1))
                available_modals = np.array([float(visual_std.any()), float(tactile_std.any())])

                if 'shock' in info.keys():
                    shock = np.array(info['shock'][time_step_in_seq])
                    shock = normalize(shock, shock_min, shock_max)
                    seq_data.append([visual_img_np, tactile_img_np, pose, available_modals, shock])
                else:
                    seq_data.append([visual_img_np, tactile_img_np, pose, available_modals])
                seq_targets.append([final_visual_img_np, final_tactile_img_np, final_pose, seg_img_np])

            # shuffle the dataset
            combined = list(zip(datapoint_dict['data'], datapoint_dict['targets']))
            random.shuffle(combined)
            datapoint_dict['data'], datapoint_dict['targets'] = zip(*combined)

            save_dir = os.path.join(self.root, self._compiled_name + ".pickle")
            with open(save_dir, 'wb') as f:
                pickle.dump(datapoint_dict, f)
                return datapoint_dict

        else:

            root = Path(self.root).joinpath("dataset")
            initial_visual_images = sorted(root.glob('**/visual/initial.png'))
            initial_tactile_images = sorted(root.glob('**/tactile/initial.png'))
            final_visual_images = sorted(root.glob('**/visual/final.png'))
            final_tactile_images = sorted(root.glob('**/tactile/final.png'))
            # visual_bg = Image.open(sorted(root.glob('**/visual_reference.png'))[0])
            # tactile_bg = Image.open(sorted(root.glob('**/tactile_reference.png'))[0])

            seq_length = 2
            crop_size = 40, 10, 330, 290

            print("Visual images: {}, Tactile images: {}, Sequences: {}, Sequence length: {}".format(
                len(initial_visual_images) * seq_length,
                len(initial_tactile_images) * seq_length,
                len(initial_visual_images), seq_length
            ))
            # bbox = self._middle_bounding_box(410, 308)
            for i in range(len(initial_visual_images)):
                print("Sequence #{}".format(i))

                mask = self._color_mask(final_visual_images[i], crop_size)

                visual_img_np = self._load_image(initial_visual_images[i], bounding_box=None, background=None)
                tactile_img_np = self._load_image(initial_tactile_images[i], bounding_box=None, background=None)

                final_visual_img_np = self._load_image(final_visual_images[i], bounding_box=None,
                                                       background=None, mask=mask, crop_size=crop_size)
                final_tactile_img_np = self._load_image(final_tactile_images[i], bounding_box=None,
                                                        background=None, mask=mask, crop_size=crop_size)

                datapoint_dict['data'].append([[visual_img_np, tactile_img_np]])
                datapoint_dict['targets'].append([[final_visual_img_np, final_tactile_img_np]])

            # shuffle the dataset
            combined = list(zip(datapoint_dict['data'], datapoint_dict['targets']))
            random.shuffle(combined)
            datapoint_dict['data'], datapoint_dict['targets'] = zip(*combined)

            save_dir = os.path.join(self.root, self._compiled_name + ".pickle")
            with open(save_dir, 'wb') as f:
                pickle.dump(datapoint_dict, f)
                return datapoint_dict

    def _load_dataset(self):
        with open(self.dataset_path, 'rb') as f:
            return pickle.load(f)

    def _load_image(self, img_path, bounding_box=None, resize=True, background=None, mask=None, crop_size=None):
        """
        Helper function to load and convert the image.
        """
        if mask is None:
            img = Image.open(img_path)
        else:
            assert crop_size is not None
            x, y, w, h = crop_size
            img = cv2.imread(str(img_path))[y:y+h, x:x+w]
            img = cv2.bitwise_and(img, img, mask=mask)
            img[mask == 0] = [210] * 3
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if background is not None:
            img = ImageChops.subtract(img, background)

        if bounding_box is not None:
            img = img.crop(bounding_box)
        if resize:
            img = img.resize((256, 256))
        np_img = np.array(img).copy()

        if np_img.ndim == 2:
            np_img = np.repeat(np_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

        img.close()
        return np_img

    def _bounding_box(self, img):
        """
        Find the bounding box based on the segmentation image.
        """
        mask = np.where(img == np.max(img))
        ymin, ymax = np.min(mask[0]), np.max(mask[0])
        xmin, xmax = np.min(mask[1]), np.max(mask[1])

        height = ymax - ymin
        width = xmax - xmin
        diff = height - width

        if diff > 0:
            xmin = max(0, xmin - diff / 2)
            xmax = min(img.shape[1], xmax + diff / 2)
        elif diff < 0:
            ymin = max(0, ymin - abs(diff) / 2)
            ymax = min(img.shape[0], ymax + abs(diff) / 2)

        return xmin, ymin, xmax, ymax

    def _color_mask(self, img, crop_size):
        x, y, w, h = crop_size
        img = cv2.imread(str(img))[y:y+h, x:x+w]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 50, 50])
        upper = np.array([150, 255, 255])

        mask = cv2.bitwise_not(cv2.inRange(hsv, lower, upper))

        return mask

    def _middle_bounding_box(self, width, height):
        return 1/4*width, 1/4*height, 3/4*width, 3/4*height

    def _parse_list_data(self, data):
        data_tmp = data.copy()
        for i, d in enumerate(data):
            if d.ndim > 1:
                # image
                data_tmp[i] = self.transform(Image.fromarray(d))
            else:
                # non-image
                data_tmp[i] = torch.from_numpy(d).float()

        return data_tmp


def seq_collate_fn(batch):
    data_input, data_target = zip(*batch)

    data_input = list(map(list, zip(*data_input)))
    data_target = list(map(list, zip(*data_target)))

    data_input = [torch.cat(data_input[i], dim=0) for i in range(len(data_input))]
    data_target = [torch.cat(data_target[i], dim=0) for i in range(len(data_target))]

    return data_input, data_target


def normalize(x, min, max):
    return np.nan_to_num((x - min) / (max - min), nan=0.)
