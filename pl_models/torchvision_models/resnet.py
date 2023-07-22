import inspect

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from pl_modules import Classifier

cls_dict = {
    'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152
}


class ResNet(Classifier):
    def __init__(self, **kwargs):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(**classifier_kwargs)

        self.save_hyperparameters()

        cls = cls_dict[self._cls_name]

        model_valid_kwargs = inspect.signature(cls).parameters
        model_kwargs = {name: kwargs[name] for name in model_valid_kwargs if name in kwargs}

        self.model = cls(**model_kwargs)

        self.model.fc = nn.Linear(self.model.fc.in_features, kwargs['num_classes'])


class ResNet18(ResNet):
    _cls_name = 'resnet18'


class ResNet34(ResNet):
    _cls_name = 'resnet34'


class ResNet50(ResNet):
    _cls_name = 'resnet50'


class ResNet101(ResNet):
    _cls_name = 'resnet101'


class ResNet152(ResNet):
    _cls_name = 'resnet152'
