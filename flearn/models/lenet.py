import torch
import torch.nn as nn
import numpy as np
import random

import torchstat

from flearn.models.base_model import BaseModel
from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear
from flearn.utils.model_utils import is_fc, is_conv
import torch.nn.functional as F


class LENET(BaseModel):
    def __init__(self, in_channel=3, dict_module: dict = None, use_mask=True, config=None, use_batchnorm=False, num_classes: int = 10):
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()

            features = nn.Sequential(
                DenseConv2d(in_channels=in_channel, out_channels=6, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                DenseConv2d(in_channels=6, out_channels=16, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            classifier = nn.Sequential(
                DenseLinear(16 * 5 * 5, 120, use_mask=True),
                nn.ReLU(inplace=True),
                DenseLinear(120, 84, use_mask=True),
                nn.ReLU(inplace=True),
                DenseLinear(84, num_classes, use_mask=True),
                nn.ReLU(inplace=True)
            )

            dict_module["features"] = features
            dict_module["classifier"] = classifier
        super(LENET, self).__init__(nn.CrossEntropyLoss(), dict_module)

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        self.prunable_layers = [layer for layer in self.param_layers if is_conv(layer) or is_fc(layer)]
        self.prunable_layer_prefixes = [pfx for ly, pfx in zip(self.param_layers, self.param_layer_prefixes) if
                                        is_conv(ly) or is_fc(ly)]
        self.relu_layers = [m for (k, m) in self.named_modules() if isinstance(m, nn.ReLU)]
        self.relu_layers_prefixes = [k for (k, m) in self.named_modules() if isinstance(m, nn.ReLU)]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LENETOrigin(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, config=None, linear_config =None):
        super(LENETOrigin, self).__init__()
        self.in_channel = in_channel
        if config is None:
            self.config = [6, "M", 16, "M"]
        else:
            self.config = config
        self.features = self._make_feature_layers()
        if linear_config is None:
            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.config[2] * 5 * 5, linear_config[0]),
                nn.ReLU(inplace=True),
                nn.Linear(linear_config[0], linear_config[1]),
                nn.ReLU(inplace=True),
                nn.Linear(linear_config[1], num_classes)
            )

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=5),
                               nn.ReLU(inplace=True)])
                in_channels = param
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LENET_header(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, config=None, linear_config =None):
        super(LENET_header, self).__init__()
        self.in_channel = in_channel
        if config is None:
            self.config = [6, "M", 16, "M"]
        else:
            self.config = config
        self.features = self._make_feature_layers()
        if linear_config is None:
            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(inplace=True),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.config[2] * 5 * 5, linear_config[0]),
                nn.ReLU(inplace=True),
                nn.Linear(linear_config[0], linear_config[1]),
                nn.ReLU(inplace=True),
                nn.Linear(linear_config[1], 84),
                nn.Linear(84, 84),
                nn.Linear(84, 256),
                nn.Linear(256, num_classes),
            )

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=5),
                               nn.ReLU(inplace=True)])
                in_channels = param
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # 设置随机种子
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model = LENET(in_channel=3, num_classes=10)
    torchstat.stat(model, (3, 32, 32))
    # inputs = torch.ones((1, 3, 32, 32))
    # outputs = model(inputs)
    # print(outputs.shape)
    #
    # rank = [0, torch.linspace(1, 16, steps=16)]
    # _, ind = model.unstructured_by_rank(rank, 0.6, 1, "cpu", baselinename=0)
    #
    # outputs = model(inputs)
    # channels = model.get_channels()
    # print(channels)
    # print("#################")