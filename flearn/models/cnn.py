import torch
from torch import nn
from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear
from flearn.models.base_model import BaseModel
import numpy as np
import torchvision
import random
from flearn.utils.model_utils import is_conv
import torchstat
import math


class CNN(BaseModel):
    def __init__(self, in_channel=3, num_classes=10, dict_module: dict = None, use_mask=True, config=None, use_batchnorm=False):
        self.in_channel = in_channel
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()
            self.batch_norm = use_batchnorm

            if config is None:
                self.config = [32, 'M', 64, 'M', 64]
            else:
                self.config = config

            features = self._make_feature_layers()
            classifier = nn.Sequential(
                DenseLinear(4 * 4 * self.config[-1], 64, use_mask=True),
                nn.ReLU(inplace=True),
                DenseLinear(64, num_classes, use_mask=True)
            )

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(CNN, self).__init__(nn.CrossEntropyLoss(), dict_module)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def _apply_mask(self, mask):
        for name, param in self.named_parameters():
            param.data *= mask[name]

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_nums = [ly_id for ly_id, ly in enumerate(self.param_layers)]
        self.prunable_layers = [self.param_layers[ly_id] for ly_id in prunable_nums]
        self.prunable_layer_prefixes = [self.param_layer_prefixes[ly_id] for ly_id in prunable_nums]
        self.relu_layers = [m for (k, m) in self.named_modules() if isinstance(m, nn.ReLU)]
        self.relu_layers_prefixes = [k for (k, m) in self.named_modules() if isinstance(m, nn.ReLU)]

        # print(self.relu_layers, self.relu_layers_prefixes)

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([DenseConv2d(in_channels, param, kernel_size=3, padding=0, use_mask=self.use_mask),
                               nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        # if mask is not None:
        #     self._apply_mask(mask)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNNOrigin(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, conv_config=None, linear_config = None, use_batchnorm=False):
        super(CNNOrigin, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if conv_config is None:
            self.config = [32, 'M', 64, 'M', 64]
        else:
            self.config = conv_config

        self.features = self._make_feature_layers()
        if linear_config is None:
            self.classifier = nn.Sequential(
                nn.Linear(4 * 4 * self.config[-1], 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(4 * 4 * self.config[-1], linear_config[0]),
                nn.ReLU(inplace=True),
                nn.Linear(linear_config[0], num_classes)
            )


    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=0),
                               nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_header(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, conv_config=None, linear_config = None, use_batchnorm=False):
        super(CNN_header, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if conv_config is None:
            self.config = [32, 'M', 64, 'M', 64]
        else:
            self.config = conv_config

        self.features = self._make_feature_layers()
        if linear_config is None:
            self.classifier = nn.Sequential(
                nn.Linear(4 * 4 * self.config[-1], 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(4 * 4 * self.config[-1], linear_config[0]),
                nn.ReLU(inplace=True),
                nn.Linear(linear_config[0], 84),
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
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=0),
                               nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__=="__main__":
    # 设置随机种子
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    np.random.seed(777)
    random.seed(777)
    torch.backends.cudnn.deterministic = True
