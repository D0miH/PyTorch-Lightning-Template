import inspect

import torch.nn as nn
from torchvision.models import resnet50, resnet18
from pl_modules import Classifier


class ResNet18(Classifier):
    def __init__(self, **kwargs):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(**classifier_kwargs)

        self.save_hyperparameters()

        model_valid_kwargs = inspect.signature(resnet18).parameters
        model_kwargs = {name: kwargs[name] for name in model_valid_kwargs if name in kwargs}

        self.model = resnet18(model_kwargs)

        self.model.fc = nn.Linear(self.model.fc.in_features, kwargs['num_classes'])


class ResNet50(Classifier):
    def __init__(self, **kwargs):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(**classifier_kwargs)

        self.save_hyperparameters()

        model_valid_kwargs = inspect.signature(resnet50).parameters
        model_kwargs = {name: kwargs[name] for name in model_valid_kwargs if name in kwargs}

        self.model = resnet50(model_kwargs)

        self.model.fc = nn.Linear(self.model.fc.in_features, kwargs['num_classes'])
