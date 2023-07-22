from typing import Type, Optional, Dict, Any, Literal, List
import inspect

import torch
import torchattacks
import torchattacks.attack

from pl_modules import Classifier
from models import CifarResNet as PyTorchResNet
from models import BasicBlock


class CifarResNet(Classifier):
    def __init__(self, num_blocks: List[int], block: Literal['BasicBlock'] = BasicBlock, **kwargs):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(**classifier_kwargs)

        self.save_hyperparameters()

        self.model = PyTorchResNet(block=block, num_blocks=num_blocks, num_classes=classifier_kwargs['num_classes'])

        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path)
            del checkpoint['state_dict']['model.linear.weight']
            del checkpoint['state_dict']['model.linear.bias']
            # load the model from the checkpoint. This will only load the weights.
            # The rest (epoch, state of optimizers, etc.) will not be loaded
            self.model.load_state_dict(
                {key.replace('model.', ''): value
                 for key, value in checkpoint['state_dict'].items()}, strict=False
            )
