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
from data.cifar10.cifar10_data import get_dataset
from flearn.optim.prox import Prox
from flearn.utils.model_utils import test_inference
import math

class Alexnet(BaseModel):
    def __init__(self, in_channel=3, num_classes=10, dict_module: dict = None, use_mask=True, config=None, use_batchnorm=False):
        self.in_channel = in_channel
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()
            features = nn.Sequential(
                DenseConv2d(in_channel, 96, 3, 1, 1),  # in_channels, out_channels, kernel_size, stride, padding
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # kernel_size, stride
                # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
                DenseConv2d(96, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
                # 前两个卷积层后不使用池化层来减小输入的高和宽
                DenseConv2d(256, 384, 3, 1, 1),
                nn.ReLU(),
                DenseConv2d(384, 384, 3, 1, 1),
                nn.ReLU(),
                DenseConv2d(384, 256, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )

            # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
            classifier = nn.Sequential(
                DenseLinear(256 * 4 * 4, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                DenseLinear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
                DenseLinear(4096, num_classes),
            )
            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(Alexnet, self).__init__(nn.CrossEntropyLoss(), dict_module)


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

    def forward(self, img):
        feature = self.features(img)
        output = self.classifier(feature.view(img.shape[0], -1))
        return output

# class VGGOrigin(nn.Module):
#     def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
#         super(VGGOrigin, self).__init__()
#         self.in_channel = in_channel
#         self.batch_norm = use_batchnorm
#
#         if config is None:
#             self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
#         else:
#             self.config = config
#
#         self.features = self._make_feature_layers()
#         self.classifier = nn.Sequential(
#             nn.Linear(self.config[-2], 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_classes)
#         )
#
#     def _make_feature_layers(self):
#         layers = []
#         in_channels = self.in_channel
#         for param in self.config:
#             if param == 'M':
#                 layers.append(nn.MaxPool2d(kernel_size=2))
#             else:
#                 layers.extend([nn.Conv2d(in_channels, param, kernel_size=3, padding=1),
#                                nn.ReLU(inplace=True)])
#                 in_channels = param
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


if __name__=="__main__":
    # 设置随机种子
    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model = Alexnet()
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


    train_dataset, test_dataset, user_groups = get_dataset(num_data=50000, num_users=100, l=2,
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
