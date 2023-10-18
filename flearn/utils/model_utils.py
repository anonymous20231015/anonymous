import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear

def gradient_norm(model, w, steps):
    """
    Normalize the local gradient
    :param w:
    :param steps:
    :return:
    """
    w_global = model.state_dict()
    w_bef = copy.deepcopy(w_global)
    for i in range(len(w)):
        for key in w_bef.keys():
            delta = w[i][1][key] - w_bef[key]
            torch.div(delta, steps[i])
            w[i][1][key] = w_bef[key] + delta

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = 0
    for i in range(0, len(w)):
        total += w[i][0]
    for key in w_avg.keys():
        w_avg[key] *= w[0][0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key] * w[i][0]
        w_avg[key] = torch.div(w_avg[key], total)
    return w_avg


def ratio_combine(w1, w2, ratio=0):
    """
    将两个权重进行加权平均，ratio 表示 w2 的占比
    :param w1:
    :param w2:
    :param ratio:
    :return:
    """
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w


def ratio_minus(w1, P, ratio=0):
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] = w1[key] - P[key] * ratio
    return w


def test_inference(model, test_dataset, device):
    """ Returns the val accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return round(accuracy, 4), round(loss / (len(testloader)), 4)


def is_fc(layer):
    return isinstance(layer, DenseLinear)


def is_conv(layer):
    return isinstance(layer, DenseConv2d)


def is_bn(layer):
    return isinstance(layer, nn.BatchNorm2d)


def is_norm_conv(layer):
    return isinstance(layer, nn.Conv2d)


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._modules.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")


def v_adjust(w1, gama):
    w = copy.deepcopy(w1)
    for key in w1.keys():
        w[key] = w1[key] * gama
    return w
