import os
from collections import ChainMap
from typing import Optional, Type, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, DataLoader

from datasets.dataset_interface import DatasetInterface


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        batch_size: int,
        dataset_args: Optional[Dict[str, Any]] = None,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        dataloader_args: Optional[Dict[str, Any]] = None,
        train_dataset_cls: Type[Dataset] = None,
        train_dataset_seed: int = None,
        val_dataset_cls: Type[Dataset] = None,
        val_dataset_seed: int = None,
        test_dataset_cls: Type[Dataset] = None,
        test_dataset_seed: int = None,
    ):
        super().__init__()

        self.dataset_cls = dataset_cls
        self.dataset_args = dataset_args or {}
        self.batch_size = batch_size
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._test_transforms = test_transforms
        self.dataloader_args = dataloader_args or {}
        self.train_dataset_cls = train_dataset_cls
        self.val_dataset_cls = val_dataset_cls
        self.test_dataset_cls = test_dataset_cls
        self.train_dataset_seed = train_dataset_seed
        self.val_dataset_seed = val_dataset_seed
        self.test_dataset_seed = test_dataset_seed

        # those attributes will be set during setup
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        train_set_cls = self.train_dataset_cls or self.dataset_cls
        train_set_cls(**ChainMap({'train': True}, self.dataset_args))

        test_set_cls = self.test_dataset_cls or self.dataset_cls
        test_set_cls(**ChainMap({'train': False}, self.dataset_args))

    def setup(self, stage: Optional[str] = None, validation_split_ratio: float = 0.1):
        # even though these datasets are identical, we have to create it once for the training data and once for
        # the validation data because of the different transformations
        train_set_cls = self.train_dataset_cls or self.dataset_cls
        val_set_cls = self.val_dataset_cls or self.dataset_cls

        train_data: DatasetInterface = train_set_cls(
            transform=self._train_transforms, **ChainMap({'train': True}, self.dataset_args)
        )
        train_indices = [i for i in range(len(train_data))]
        # if a training and validation dataset are the same class we have to split the dataset
        if self.train_dataset_cls == self.val_dataset_cls:
            train_indices, _ = train_test_split(
                list(range(len(train_data))),
                test_size=validation_split_ratio,
                random_state=self.train_dataset_seed or int(os.environ['PL_GLOBAL_SEED']),
                shuffle=True,
                stratify=train_data.targets
            )
        self.train_data = Subset(train_data, train_indices)

        val_data: DatasetInterface = val_set_cls(
            transform=self._val_transforms, **ChainMap({'train': True}, self.dataset_args)
        )

        _, val_indices = train_test_split(
            list(range(len(val_data))),
            test_size=validation_split_ratio,
            random_state=self.val_dataset_seed or int(os.environ['PL_GLOBAL_SEED']),
            shuffle=True,
            stratify=val_data.targets
        )
        self.val_data = Subset(val_data, val_indices)

        test_set_cls = self.test_dataset_cls or self.dataset_cls
        self.test_data = test_set_cls(transform=self._test_transforms, **ChainMap({'train': False}, self.dataset_args))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, batch_size=self.batch_size, **self.dataloader_args)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data, batch_size=self.batch_size, **ChainMap({'shuffle': False}, self.dataloader_args)
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, **ChainMap({'shuffle': False}, self.dataloader_args)
        )

    def num_classes(self) -> int:
        dataset_cls = self.train_dataset_cls or self.dataset_cls
        return len(dataset_cls(**ChainMap({'train': False}, self.dataset_args)).classes)
