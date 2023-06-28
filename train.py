import argparse
from typing import Optional

from pytorch_lightning import seed_everything
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import MultiStepLR

import hydra
from omegaconf import DictConfig, OmegaConf

import pl_models
import datasets

from pl_modules import Classifier
from pl_modules.datamodule import DataModule
from utils import get_class_from_module, LightningRtpt


@hydra.main(version_base=None, config_path='configs', config_name='defaults')
def train_model(cfg: DictConfig):
    # resolve the config using omegaconf
    OmegaConf.resolve(cfg)

    # get the model
    model_cls: Classifier = get_class_from_module(pl_models, cfg.model.arch)

    seed_everything(cfg.seed, workers=True)

    # define the training and eval transforms
    eval_transforms_list = []
    for augm in cfg.test_augmentations:
        augm_name, augm_args = list(augm.items())[0]
        augm_cls = get_class_from_module(T, augm_name)
        eval_transforms_list.append(augm_cls(**augm_args['args']))
    eval_transforms_list.extend([
        T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    eval_transforms = T.Compose(eval_transforms_list)

    train_transforms_list = []
    if cfg.training.train_augmentations is None:
        cfg.training.train_augmentations = cfg.test_augmentations

    for augm in cfg.training.train_augmentations:
        augm_name, augm_args = list(augm.items())[0]
        augm_cls = get_class_from_module(T, augm_name)
        train_transforms_list.append(augm_cls(**augm_args['args']))
    train_transforms_list.extend([
        T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_transforms = T.Compose(train_transforms_list)

    # get the datamodule
    dataset_cls = get_class_from_module(datasets, cfg.dataset.dataset)
    datamodule = DataModule(
        dataset_cls,
        dataset_args={
            'root': f'./data/{cfg.dataset.dataset.lower()}', 'download': True
        },
        dataloader_args={
            'num_workers': 8
        },
        batch_size=cfg.training.batch_size,
        train_transforms=train_transforms,
        val_transforms=eval_transforms,
        test_transforms=eval_transforms
    )
    if cfg.dataset.num_classes is None:
        cfg.dataset.num_classes = datamodule.num_classes()

    model = model_cls.from_cfg(cfg)
    if model.adv_attack_cls is not None:
        model.adv_attack = model.adv_attack_cls(model, **model.adv_attack_args)
        model.adv_attack.set_normalization_used(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # model.adv_attack = model.adv_attack.to(model.device)


    model_checkpoint = ModelCheckpoint(save_last=True, verbose=True)
    rtpt = LightningRtpt(name_initials="DH", experiment_name=model.get_architecture_name(), max_iterations=cfg.training.epochs)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [rtpt, model_checkpoint, lr_monitor]

    # if not disabled use early stopping
    if cfg.training.early_stopping.use_early_stopping:
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.early_stopping_patience,
            min_delta=cfg.training.early_stopping_min_delta,
            verbose=True
        )
        callbacks.append(early_stopper)

    wandb_logger = True
    if cfg.wandb.use_wandb:
        arch_name = model.get_architecture_name()
        wandb_logger = WandbLogger(
            name=f'{arch_name} {cfg.dataset.dataset} {cfg.wandb.experiment_name}', project=cfg.wandb.project, entity=cfg.wandb.entity, log_model=not cfg.training.do_not_save_model
        )
        wandb_logger.watch(model)
        wandb_logger.log_hyperparams(cfg)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        max_epochs=cfg.training.epochs,
        deterministic=True,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    train_model()
