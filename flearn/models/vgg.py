import torch
from torch import nn
from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear
from flearn.models.base_model import BaseModel
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import random
from flearn.utils.model_utils import is_conv
import torchstat
from thop import profile
# from data.cifar10.cifar10_data import get_dataset
from flearn.optim.prox import Prox
from flearn.utils.model_utils import test_inference
import math

class VGG11(BaseModel):
    def __init__(self, in_channel=3, num_classes=10, dict_module: dict = None, use_mask=True, config=None, use_batchnorm=False):
        self.in_channel = in_channel
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()
            self.batch_norm = use_batchnorm

            if config is None:
                self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            else:
                self.config = config

            features = self._make_feature_layers()
            classifier = nn.Sequential(
                DenseLinear(self.config[-2], 512, use_mask=use_mask),
                nn.ReLU(inplace=True),
                DenseLinear(512, 512, use_mask=use_mask),
                nn.ReLU(inplace=True),
                DenseLinear(512, num_classes, use_mask=use_mask)
            )

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(VGG11, self).__init__(nn.CrossEntropyLoss(), dict_module)

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
                layers.extend([DenseConv2d(in_channels, param, kernel_size=3, padding=1, use_mask=self.use_mask),
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

class VGGOrigin(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
        super(VGGOrigin, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if config is None:
            self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        else:
            self.config = config

        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential(
            nn.Linear(self.config[-2], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_head(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
        super(VGG_head, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if config is None:
            self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        else:
            self.config = config

        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential(
            nn.Linear(self.config[-2], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=1),
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
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model = VGG11()
    # inputs = torch.ones((64, 3, 32, 32))
    # outputs = model(inputs)
    # print(outputs.shape)
    # input_image = torch.randn(1, 3, 32, 32)
    # flops, params = profile(model, inputs=(input_image,))
    # print(flops)
    # print(params)
    # rank = [torch.linspace(1, 64, steps=64)]
    # _, ind = model.unstructured_by_rank(rank, 0.6, 0, "cpu")
    # rank = [0, torch.linspace(1, 128, steps=128)]
    # _, ind = model.unstructured_by_rank(rank, 0.6, 1, "cpu")
    # outputs = model(inputs)
    # channels = model.get_channels()


    train_dataset, test_dataset, user_groups = get_dataset(num_data=40000, num_users=100, iid=False, num_share=4000, l=2,
                                                          unequal=True)
    train_loader = DataLoader(train_dataset, 100, shuffle=True, num_workers=0, drop_last=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    learning_rate = 1e-1
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = Prox(model.parameters(), lr=learning_rate, mu=0)

    # 设置网络的训练参数
    total_train_step = 0
    total_test_step = 0
    epoch = 1000

    model.train()
    # 开始训练
    for i in range(epoch):
        print(f"=== epoch {i} start ===")

        avg_loss = 0
        for data in train_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 10 == 0:
                model.recovery_model()
                print(loss)

        test_acc, test_loss = test_inference(model, test_dataset, device)
        print(test_acc)
