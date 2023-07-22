import inspect

import torch.nn as nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from pl_modules import Classifier

cls_dict = {
    'densenet121': densenet121, 'densenet161': densenet161, 'densenet169': densenet169, 'densenet201': densenet201
}


class DenseNet(Classifier):
    def __init__(self, **kwargs):
        classifier_valid_kwargs = inspect.signature(Classifier.__init__).parameters
        classifier_kwargs = {name: kwargs[name] for name in classifier_valid_kwargs if name in kwargs}
        super().__init__(**classifier_kwargs)

        self.save_hyperparameters()

        cls = cls_dict[self._cls_name]

        model_valid_kwargs = inspect.signature(cls).parameters
        model_kwargs = {name: kwargs[name] for name in model_valid_kwargs if name in kwargs}

        self.model = cls(**model_kwargs)

        self.model.classifier = nn.Linear(self.model.classifier.in_features, kwargs['num_classes'])


class DenseNet121(DenseNet):
    _cls_name = 'densenet121'


class DenseNet161(DenseNet):
    _cls_name = 'densenet161'


class DenseNet169(DenseNet):
    _cls_name = 'densenet169'


class DenseNet201(DenseNet):
    _cls_name = 'densenet201'
