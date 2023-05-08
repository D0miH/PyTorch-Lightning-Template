from typing import Type, Optional, Dict, Any, Literal, List

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
        lr: float = 1e-4,
        num_classes=10,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        lr_scheduler_cls: Optional[Any] = None,
        lr_scheduler_args: Optional[Dict[str, Any]] = None,
        adv_attack_cls: Optional[Type[torchattacks.attack.Attack]] = None,
        adv_attack_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            lr=lr,
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_args=lr_scheduler_args,
            adv_attack_cls=adv_attack_cls,
            adv_attack_args=adv_attack_args
        )
        self.save_hyperparameters()

        blocks = {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}

        self.model = PyTorchResNet(block=blocks[block], num_blocks=num_blocks, num_classes=num_classes)


class CifarResNet18(CifarResNet):
    def __init__(
        self,
        lr: float = 1e-4,
        num_classes=10,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        lr_scheduler_cls: Optional[Any] = None,
        lr_scheduler_args: Optional[Dict[str, Any]] = None,
        adv_attack_cls: Optional[Type[torchattacks.attack.Attack]] = None,
        adv_attack_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            block='BasicBlock',
            num_blocks=[2, 2, 2, 2],
            lr=lr,
            num_classes=num_classes,
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_args=lr_scheduler_args,
            adv_attack_cls=adv_attack_cls,
            adv_attack_args=adv_attack_args
        )
