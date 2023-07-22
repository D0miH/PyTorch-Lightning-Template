import sys

from pytorch_lightning import seed_everything
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import MultiStepLR
import wandb
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import pl_models
import datasets

from pl_modules import Classifier
from pl_modules.datamodule import DataModule
from utils import get_class_from_module, LightningRtpt
from custom_resolvers import wandb_artifact

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("wandb_artifact", wandb_artifact)


@hydra.main(version_base=None, config_path='configs', config_name='defaults.yaml')
def train_model(cfg: DictConfig):
    # seed everything for reproducibility
    seed_everything(cfg.seed, workers=True)

    # define the training and eval transforms
    eval_transforms_list = []
    for aug_key, aug_dict in cfg.eval_transforms.items():
        eval_transforms_list.append(hydra.utils.instantiate(aug_dict))
    eval_transforms_list.extend([T.ToTensor(), T.Normalize(mean=cfg.normalization.mean, std=cfg.normalization.std)])
    eval_transforms = T.Compose(eval_transforms_list)

    train_transforms_list = []
    for aug_key, aug_dict in cfg.train_transforms.items():
        train_transforms_list.append(hydra.utils.instantiate(aug_dict))
    train_transforms_list.extend([T.ToTensor(), T.Normalize(mean=cfg.normalization.mean, std=cfg.normalization.std)])
    train_transforms = T.Compose(train_transforms_list)

    # get the datamodule
    dataset_cls = hydra.utils.instantiate(cfg.dataset.class_instance, _partial_=True)
    datamodule_cls = hydra.utils.instantiate(cfg.datamodule, _partial_=True)

    datamodule = datamodule_cls(
        dataset_cls=dataset_cls,
        train_transforms=train_transforms,
        val_transforms=eval_transforms,
        test_transforms=eval_transforms
    )

    # check if a lr scheduler or an adv. attack is given
    for key in ('lr_scheduler', 'adv_training'):
        # if the value is an empty dict assign None.
        # This is a dirty workaround since we cannot assign None values to default list items in hydra
        if cfg[key] == {}:
            cfg[key] = None

    model: Classifier = hydra.utils.instantiate(cfg.model.class_instance)
    if model.partial_adv_attack is not None:
        model.configure_adv_attack(normalization_mean=cfg.normalization.mean, normalization_std=cfg.normalization.std)

    model_checkpoint = ModelCheckpoint(save_last=True, verbose=True)
    rtpt = LightningRtpt(
        name_initials=cfg.rtpt.initials,
        experiment_name=model.get_architecture_name(),
        max_iterations=cfg.training.epochs
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [rtpt, model_checkpoint, lr_monitor]

    # if not disabled use early stopping
    if hasattr(cfg.training, 'early_stopping'):
        early_stopper = EarlyStopping(
            monitor='val_loss',
            patience=cfg.training.early_stopping.early_stopping_patience,
            min_delta=cfg.training.early_stopping.early_stopping_min_delta,
            verbose=True
        )
        callbacks.append(early_stopper)

    wandb_logger = WandbLogger(
        name=cfg.wandb.run_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        log_model=(cfg.training.save_model and not cfg.wandb.offline),
        offline=cfg.wandb.offline
    )
    wandb_logger.watch(model)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

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

    # save the hydra logs for debugging and reproducibility
    # save the hydra logs for debugging and reproducibility
    hydra_artifact = wandb.Artifact(f'hydra_config-{wandb.run.id}', type='hydra_config')
    hydra_artifact.add_dir('./' + hydra.core.hydra_config.HydraConfig.get().run.dir + '/.hydra/')
    wandb.run.log_artifact(hydra_artifact)


if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%Y-%m-%d_%H-%M-%S}')
    train_model()
