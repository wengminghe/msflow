import os
import random

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np

CLASS_NAMES = ['lace', 'fiber']


class TextileDataset(Dataset):
    def __init__(self, data_dir, class_name, is_train, input_size=512, random_mask=False, **kwargs):
        assert class_name in CLASS_NAMES, ('class_name: {}, should be in {}'.format(class_name, CLASS_NAMES))
        self.data_dir = data_dir
        self.class_name = class_name
        self.is_train = is_train
        self.input_size = input_size
        self.random_mask = random_mask

        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = T.Compose([
            # T.Resize(input_size, T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.transform_mask = T.Compose([
            # T.Resize(input_size, T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = cv.imread(x)

        if self.random_mask:
            if y == 0:
                if self.is_train:
                    if random.random() < 0.5:
                        t = random.choices([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0]
                        t = int(x.shape[1] * t)
                        if random.random() < 0.5:
                            x[:, t:, :] = 0
                        else:
                            x[:, :t, :] = 0
                else:
                    if idx % 2 == 0:
                        t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][(idx // 2) % 10]
                        t = int(x.shape[1] * t)
                        if (idx // 10) % 2 == 0:
                            x[:, t:, :] = 0
                        else:
                            x[:, :t, :] = 0

        input_size = (self.input_size * x.shape[0] // 1000, self.input_size * x.shape[1] // 1000)
        x = cv.resize(x, (input_size[1], input_size[0]), interpolation=cv.INTER_LANCZOS4)
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.input_size, self.input_size])
        else:
            mask = cv.imread(mask)
            mask = cv.resize(mask, (input_size[1], input_size[0]), interpolation=cv.INTER_NEAREST)
            mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
            mask = self.transform_mask(mask)

        if input_size[0] != input_size[1]:
            x = torch.chunk(x, chunks=4, dim=-1)
            x = torch.stack(x)
            mask = torch.chunk(mask, chunks=4, dim=-1)
            mask = torch.stack(mask)
        else:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
        mask = mask[:, 0:1, :, :]

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []
        img_dir = os.path.join(self.data_dir, self.class_name, 'train' if self.is_train else 'test')
        gt_dir = os.path.join(self.data_dir, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
