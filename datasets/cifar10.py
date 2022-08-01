from typing import Any

from torchvision.datasets import CIFAR10 as TorchCifar10

from datasets.dataset_interface import DatasetInterface


class CIFAR10(TorchCifar10, DatasetInterface):
    """
    Wrapper for the `standard` CIFAR10 dataset implementation of TorchVision to comply with the standard of having
    train=True/False.
    """
    def __init__(self, root: str, train: bool = True, download: bool = True, **kwargs: Any) -> None:
        super().__init__(root, train=train, download=download, **kwargs)

        # this is nonsensical. However, the targets of `DatasetInterface` has to be set.
        self.targets = self.targets
