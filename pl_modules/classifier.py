import functools
import inspect
from argparse import Namespace, ArgumentParser
from typing import Any, Dict, Optional, Type, Union, Callable, Iterator

import torchattacks
import torchattacks.attack
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import LRScheduler

import pytorch_lightning as pl
import torchmetrics


class Classifier(pl.LightningModule):
    model: nn.Module
    adv_attack: torchattacks.attack.Attack = None

    def __init__(
        self,
        lr: float = 1e-4,
        num_classes=10,
        partial_optimizer: Callable[[Iterator[Parameter]], torch.optim.Optimizer] = functools.partial(torch.optim.Adam),
        partial_lr_scheduler: Optional[Callable[[torch.optim.Optimizer], LRScheduler]] = None,
        partial_adv_attack: Optional[Callable[[nn.Module], torchattacks.attack.Attack]] = None,
        checkpoint_path: Optional[Any] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self._num_classes = num_classes
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.partial_optimizer = partial_optimizer
        self.partial_lr_scheduler = partial_lr_scheduler
        self.partial_adv_attack = partial_adv_attack
        self.checkpoint_path = checkpoint_path

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # if there are labels or other things given just ignore them
        if len(batch) >= 2:
            return self(batch[0])

        return self(batch)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.adv_attack is not None:
            self.adv_attack.device = self.device
            self.eval()
            adv_examples = self.adv_attack(x, y)
            self.train()
            x = adv_examples

        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_acc", self.accuracy(output.softmax(1), y).item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.log("val_acc", self.accuracy(output.softmax(1), y).item(), prog_bar=True)
        self.log("val_loss", loss.item(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.log("test_acc", self.accuracy(output.softmax(1), y).item())
        self.log("test_loss", loss.item())

    def configure_optimizers(self):
        optim = self.partial_optimizer(self.model.parameters())

        if self.partial_lr_scheduler is None:
            return optim

        scheduler = self.partial_lr_scheduler(optim)
        lr_scheduler_config = {'scheduler': scheduler, 'interval': 'step', 'monitor': 'val_loss', 'name': 'LR'}

        return [optim], [lr_scheduler_config]

    def configure_adv_attack(self, normalization_mean=None, normalization_std=None):
        self.adv_attack = self.partial_adv_attack(self)
        if normalization_mean is not None and normalization_std is not None:
            self.adv_attack.set_normalization_used(mean=normalization_mean, std=normalization_std)

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, num_classes: int):
        raise NotImplementedError

    def get_architecture_name(self) -> str:
        return type(self).__name__

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser

    @classmethod
    def from_argparse_args(cls: Type["Classifier"], args: Namespace) -> "Classifier":
        params = vars(args)

        valid_kwargs = inspect.signature(cls.__init__).parameters
        cls_kwargs = {name: params[name] for name in valid_kwargs if name in params}

        return cls(**cls_kwargs)
