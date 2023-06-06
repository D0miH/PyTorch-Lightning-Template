import inspect
from argparse import Namespace, ArgumentParser
from typing import Any, Dict, Optional, Type, Union

import torchattacks
import torchattacks.attack
from omegaconf import DictConfig
from torch import nn
import torch.nn.functional as F
import torch

import pytorch_lightning as pl
import torchmetrics

from utils import get_class_from_module


class Classifier(pl.LightningModule):

    model: nn.Module
    adv_attack: torchattacks.attack.Attack = None

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
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self._num_classes = num_classes
        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_args = lr_scheduler_args
        self.adv_attack_cls = adv_attack_cls
        self.adv_attack_args = adv_attack_args

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
        self.log("train_loss", round(loss.item(), 3), prog_bar=True)
        self.log("train_acc", round(self.accuracy(output.softmax(1), y).item(), 3), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.log("val_acc", round(self.accuracy(output.softmax(1), y).item(), 3), prog_bar=True)
        self.log("val_loss", round(loss.item(), 3), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        self.log("test_acc", round(self.accuracy(output.softmax(1), y).item(), 3))
        self.log("test_loss", round(loss.item(), 3))

    def configure_optimizers(self):
        optim = self.optimizer_cls(self.model.parameters(), lr=self.lr, **(self.optimizer_args or {}))

        if self.lr_scheduler_cls is None:
            return optim

        scheduler = self.lr_scheduler_cls(optim, **(self.lr_scheduler_args or {}))
        lr_scheduler_config = {'scheduler': scheduler, 'interval': 'step', 'monitor': 'val_loss', 'name': 'LR'}

        return [optim], [lr_scheduler_config]

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

    @classmethod
    def from_cfg(cls: Type["Classifier"], cfg: DictConfig) -> "Classifier":

        adv_attack_cls = None
        adv_attack_args = None
        if cfg.training.adv_training.use_adv_training:
            adv_attack_cls = get_class_from_module(torchattacks, cfg.training.adv_training.adv_attack.class_name)
            adv_attack_args = cfg.training.adv_training.adv_attack.args

        model_args = {
            'lr': cfg.optimizer.lr,
            'num_classes': cfg.dataset.num_classes,
            'optimizer_args': cfg.optimizer.optimizer_args,
            'adv_attack_cls': adv_attack_cls,
            'adv_attack_args': adv_attack_args
        }

        if cfg.model.model_args is not None:
            model_args.update(cfg.model.model_args)

        return cls(**model_args)