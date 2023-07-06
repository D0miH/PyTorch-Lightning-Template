from typing import Type, Optional, Dict, Any, Literal, List
import inspect

import torch
import torchattacks
import torchattacks.attack

from pl_modules import Classifier
from models.cifar10_resnet import ResNet as PyTorchResNet
from models.cifar10_resnet import BasicBlock, Bottleneck


class CifarResNet(Classifier):
    def __init__(
        self,
        block: Literal['BasicBlock', 'Bottleneck'],
        num_blocks: List[int],
        **kwargs
    ):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(**classifier_kwargs)

        self.save_hyperparameters()

        blocks = {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}

        self.model = PyTorchResNet(block=blocks[block], num_blocks=num_blocks, num_classes=classifier_kwargs['num_classes'])


class CifarResNet18(CifarResNet):
    def __init__(self, **kwargs):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(
            block='BasicBlock',
            num_blocks=[2, 2, 2, 2],
            **classifier_kwargs
        )
