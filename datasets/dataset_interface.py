import inspect
from abc import ABC
from typing import Any, List

import numpy.typing
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torchvision.datasets import ImageFolder


class CustomImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        valid_kwargs = inspect.signature(ImageFolder.__init__).parameters
        cls_kwargs = {name: kwargs[name] for name in valid_kwargs if name in kwargs}
        super().__init__(**cls_kwargs)
        self.data = [s[0] for s in self.imgs]


class DatasetInterface(ABC):
    """
    Wrapper for dataset classes that need to implement certain attributes.
    """

    data: numpy.typing.NDArray[Any]
    targets: List[Any]

    def __len__(self) -> int:
        return len(self.data)


class PartitionMixin(DatasetInterface):
    def __init__(self, partitions: int = 1, used_partition: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partitions = partitions
        self.used_partition = used_partition

        if self.partitions == 1 or not kwargs['train']:
            return

        indices = np.arange(len(self))
        partition_indices = np.array_split(indices, self.partitions)[self.used_partition]

        if self.partitions == 2:
            splits = train_test_split(
                indices,
                train_size=1 / self.partitions,
                random_state=int(os.environ['PL_GLOBAL_SEED']),
                stratify=self.targets
            )
        else:
            raise RuntimeError('At the moment only 1 or two partitions are supported.')

        self.data = np.take(self.data, splits[self.used_partition], axis=0)
        self.targets = np.take(self.targets, splits[self.used_partition]).tolist()
