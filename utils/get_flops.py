import torch
from thop import profile

from flearn.models import cnn, vgg, resnet, lenet

cnn_config = [32, 'M', 64, 'M', 64]
vgg_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
resnet_config = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
lenet_config = [6, 16]


def get_flops(model="cnn", config=None, linear_config=None, dataset="cifar10"):
    """
    给定模型和通道数量，返回计算量和参数量
    :param model:
    :param config:
    :return:
    """
    # print(len(config))
    if dataset == "cifar10" or dataset == "svhn":
        in_channel = 3
        num_classes = 10
    elif dataset == "mnist" or dataset == "fashionmnist":
        in_channel = 1
        num_classes = 10
    elif dataset == "cifar100":
        in_channel = 3
        num_classes = 100
    elif dataset == "tinyimagenet":
        in_channel = 3
        num_classes = 200
    else:
        exit('Error: unrecognized dataset')
    if config is None:
        print(f"using default {model} config")
    if model == "vgg" and (config is None or len(config) == len(vgg_config)):
        model = vgg.VGGOrigin(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "resnet" and (config is None or len(config) == len(resnet_config)):
        model = resnet.ResNetOrigin(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "lenet":
        model = lenet.LENETOrigin(in_channel=in_channel, num_classes=num_classes, config=config, linear_config=linear_config)
    elif model == "cnn" and (config is None or len(config) == len(cnn_config)):
        model = cnn.CNNOrigin(in_channel=in_channel, num_classes=num_classes, conv_config=config, linear_config=linear_config)
    else:
        print("unknown model")
        return 0, 0

    input_image = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input_image,))
    return flops, params

def dropout_get_flops(model="vgg", config=None, dataset="cifar10", input_num = 0, layer_cls = 'conv'):
    """
    给定模型和通道数量，返回计算量和参数量
    :param model:
    :param config:
    :return:
    """
    if dataset == "cifar10" or dataset == "svhn":
        in_channel = 3
        num_classes = 10
    elif dataset == "mnist" or dataset == "fashionmnist":
        in_channel = 1
        num_classes = 10
    elif dataset == "cifar100":
        in_channel = 3
        num_classes = 100
    elif dataset == "tinyimagenet":
        in_channel = 3
        num_classes = 200
    else:
        exit('Error: unrecognized dataset')
    if config is None:
        print(f"using default {model} config")
    if model == "vgg" and (config is None or len(config) == len(vgg_config)):
        linear_input = 512
        model = vgg.VGGOrigin(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "resnet" and (config is None or len(config) == len(resnet_config)):
        linear_input = 512
        model = resnet.ResNetOrigin(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "lenet" and (config is None or len(config) == len(lenet_config)):
        linear_input = 400
        model = lenet.LENETOrigin(in_channel=in_channel, num_classes=num_classes)
    elif model == "cnn" and (config is None or len(config) == len(cnn_config)):
        linear_input = 1024
        model = cnn.CNNOrigin(in_channel=in_channel, num_classes=num_classes)
    else:
        print("unknown model")
        return 0, 0
    if layer_cls == 'conv':
        input_image = torch.randn(input_num, in_channel, 32, 32)
        flops, params = profile(model.features, inputs=(input_image,))
        print(flops)
    elif layer_cls == 'fc':
        input_image = torch.randn(input_num, linear_input)
        flops, params = profile(model.classifier, inputs=(input_image,))
        print(flops)
    return flops, params

def MOON_get_flops(model="cnn", config=None, linear_config=None, dataset="cifar10"):
    """
    给定模型和通道数量，返回计算量和参数量
    :param model:
    :param config:
    :return:
    """
    if dataset == "cifar10" or dataset == "svhn":
        in_channel = 3
        num_classes = 10
    elif dataset == "mnist" or dataset == "fashionmnist":
        in_channel = 1
        num_classes = 10
    elif dataset == "cifar100":
        in_channel = 3
        num_classes = 100
    elif dataset == "tinyimagenet":
        in_channel = 3
        num_classes = 200
    else:
        exit('Error: unrecognized dataset')
    if config is None:
        print(f"using default {model} config")
    if model == "vgg" and (config is None or len(config) == len(vgg_config)):
        model = vgg.VGG_head(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "resnet" and (config is None or len(config) == len(resnet_config)):
        model = resnet.ResNet_head(in_channel=in_channel, num_classes=num_classes, config=config)
    elif model == "lenet":
        model = lenet.LENET_header(in_channel=in_channel, num_classes=num_classes, config=config, linear_config=linear_config)
    elif model == "cnn" and (config is None or len(config) == len(cnn_config)):
        model = cnn.CNN_header(in_channel=in_channel, num_classes=num_classes, conv_config=config, linear_config=linear_config)
    else:
        print("unknown model")
        return 0, 0

    input_image = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input_image,))
    return flops, params
if __name__=="__main__":
    cnn_linear_test = [64, 10]
    lenet_linear_test = [120, 84, 10]
    # lenet_flops, lenet_size = get_flops("lenet", linear_config=lenet_linear_test, dataset="cifar100")
    # cnn_flops, cnn_size = get_flops("cnn", linear_config=cnn_linear_test, dataset="cifar100")
    # print("lenet_test", lenet_flops / 10 ** 6)
    # print("cnn_test", cnn_flops / 10 ** 6)
    #
    # lenet_flops, lenet_size = MOON_get_flops("lenet", linear_config=lenet_linear_test, dataset="cifar100")
    # cnn_flops, cnn_size = MOON_get_flops("cnn", linear_config=cnn_linear_test, dataset="cifar100")
    # print("lenet_header_test", lenet_flops/10**6)
    # print("cnn_header_test", cnn_flops/10**6)
    # resnet_flops, resnet_size = get_flops("resnet", dataset="tinyimagenet")
    # print("resnet_test", resnet_flops/10**6)
    # resnet_flops, resnet_size = MOON_get_flops("resnet", dataset="tinyimagenet")
    # print("resnet_header_test", resnet_flops/10**6)
    vgg_flops, vgg_size = get_flops("vgg", dataset="svhn")
    print("vgg_test", vgg_flops/10**6)
    vgg_flops, vgg_size = MOON_get_flops("vgg", dataset="svhn")
    print("vgg_header_test", vgg_flops/10**6)

    # print(get_flops("resnet", config=[58, 58, 58, 58, 58, 128, 123, 123, 123, 250, 250, 250, 250, 505, 505, 505, 505]))
    # print(get_flops("resnet"))