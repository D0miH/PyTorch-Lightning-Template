from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Any, List

from datasets.dataset_interface import DatasetInterface


class FaceScrub(Dataset, DatasetInterface):
    targets: List[Any]

    def __init__(
        self,
        group='all',
        train=True,
        transform=None,
        cropped=True,
        root='data/facescrub',
        test_set_split_ratio=0.1,
        download=None
    ):
        """
        group: 'actors', 'actresses', 'all
        train: True, False
        """

        if group == 'actors':
            if cropped:
                root = os.path.join(root, 'actors/faces')
            else:
                root = os.path.join(root, 'actors/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.classes = self.dataset.classes
            self.class_to_idx = self.dataset.class_to_idx

            name_fix_dict = {
                'Freddy_Prinze_Jr.': 'Freddie_Prinze_Jr',
                'Leslie_Neilsen': 'Leslie_Nielsen',
                'Robert_Di_Niro': 'Robert_De_Niro'
            }

            for wrong_name, correct_name in name_fix_dict.items():
                wrong_name_index = self.classes.index(wrong_name)
                self.classes[wrong_name_index] = correct_name

            for class_name, idx in self.class_to_idx.copy().items():
                if class_name in name_fix_dict.keys():
                    del self.class_to_idx[class_name]
                    self.class_to_idx[name_fix_dict[class_name]] = idx

            self.targets = self.dataset.targets
        elif group == 'actresses':
            if cropped:
                root = os.path.join(root, 'actresses/faces')
            else:
                root = os.path.join(root, 'actresses/images')
            self.dataset = ImageFolder(root=root, transform=None)
            self.classes = self.dataset.classes
            self.class_to_idx = self.dataset.class_to_idx

            name_fix_dict = {'Tatyana_M._Ali': 'Tatyana_Ali'}

            for wrong_name, correct_name in name_fix_dict.items():
                wrong_name_index = self.classes.index(wrong_name)
                self.classes[wrong_name_index] = correct_name

            for class_name, idx in self.class_to_idx.copy().items():
                if class_name in name_fix_dict.keys():
                    del self.class_to_idx[class_name]
                    self.class_to_idx[name_fix_dict[class_name]] = idx

            self.targets = self.dataset.targets
        elif group == 'all':
            if cropped:
                root_actors = os.path.join(root, 'actors/faces')
                root_actresses = os.path.join(root, 'actresses/faces')
            else:
                root_actors = os.path.join(root, 'actors/images')
                root_actresses = os.path.join(root, 'actresses/images')
            dataset_actors = ImageFolder(root=root_actors, transform=None)
            target_transform_actresses = lambda x: x + len(dataset_actors.classes)
            dataset_actresses = ImageFolder(
                root=root_actresses, transform=None, target_transform=target_transform_actresses
            )
            dataset_actresses.class_to_idx = {
                key: value + len(dataset_actors.classes)
                for key, value in dataset_actresses.class_to_idx.items()
            }
            self.dataset = ConcatDataset([dataset_actors, dataset_actresses])
            self.classes = dataset_actors.classes + dataset_actresses.classes

            name_fix_dict = {
                'Freddy_Prinze_Jr.': 'Freddie_Prinze_Jr',
                'Leslie_Neilsen': 'Leslie_Nielsen',
                'Robert_Di_Niro': 'Robert_De_Niro',
                'Tatyana_M._Ali': 'Tatyana_Ali'
            }

            for wrong_name, correct_name in name_fix_dict.items():
                wrong_name_index = self.classes.index(wrong_name)
                self.classes[wrong_name_index] = correct_name

            self.class_to_idx = {**dataset_actors.class_to_idx, **dataset_actresses.class_to_idx}

            for class_name, idx in self.class_to_idx.copy().items():
                if class_name in name_fix_dict.keys():
                    del self.class_to_idx[class_name]
                    self.class_to_idx[name_fix_dict[class_name]] = idx

            self.targets = dataset_actors.targets + [t + len(dataset_actors.classes) for t in dataset_actresses.targets]
        else:
            raise ValueError(f'Dataset group {group} not found. Valid arguments are \'actors\' and \'actresses\'.')

        self.transform = transform

        # split the dataset into a train and a test set using the indices
        if test_set_split_ratio > 0:
            indices = list(range(len(self.dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=test_set_split_ratio, random_state=42, stratify=self.targets)

            if train:
                self.dataset = Subset(self.dataset, train_idx)
                self.targets = np.array(self.targets)[train_idx].tolist()
            else:
                self.dataset = Subset(self.dataset, test_idx)
                self.targets = np.array(self.targets)[test_idx].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]

        if self.transform is not None:
            im = self.transform(im)

        return im, self.targets[idx]


class FaceScrubUncropped(FaceScrub):
    def __init__(
        self, group='all', train=True, transform=None, root='data/facescrub', test_set_split_ratio=0, download=None
    ):
        super().__init__(
            group,
            train,
            transform=transform,
            cropped=False,
            root=root.replace('uncropped', ''),
            test_set_split_ratio=test_set_split_ratio
        )


class FaceScrubCropped(FaceScrub):
    def __init__(
        self, group='all', train=True, transform=None, root='data/facescrub', test_set_split_ratio=0, download=None
    ):
        super().__init__(
            group,
            train,
            transform=transform,
            cropped=True,
            root=root.replace('cropped', ''),
            test_set_split_ratio=test_set_split_ratio
        )
