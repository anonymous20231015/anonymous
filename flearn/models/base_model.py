from abc import ABC, abstractmethod
from typing import Union, Sized

import torch
from torch import nn as nn

from flearn.utils.model_utils import traverse_module, is_conv, is_fc, is_bn, is_norm_conv


class BaseModel(nn.Module, ABC):
    """
    基础模型，为模型增加剪枝功能
    """
    def __init__(self, loss_func, dict_module: dict):
        super(BaseModel, self).__init__()

        for module_name, module in dict_module.items():
            self.add_module(module_name, module)

        self.loss_func = loss_func
        self.param_layers: list = []
        self.param_layer_prefixes: list = []
        self.prunable_layers: list = []
        self.prunable_layer_prefixes: list = []
        self.relu_layers: list = []
        self.relu_layers_prefixes: list = []

        self.pre_model = None

        self.collect_layers()

    @torch.no_grad()
    def set_ind(self, ind=None, idx=None):
        """
        根据 通道序列号 和 层序列号进行裁剪
        :param ind: 要保留的通道序列号
        :param idx: 要裁剪的层的序列
        :return:
        """
        if ind is not None and idx is not None:
            layer = self.prunable_layers[idx]
            layer.prune_out_channels(ind)

            if idx < len(self.prunable_layers) - 1:
                prunable_next_layer = self.prunable_layers[idx + 1]
                prunable_next_layer.prune_in_channels(ind)

            key = self.prunable_layer_prefixes[idx]
            layer_prefix_idx = self.param_layer_prefixes.index(key)

            if layer_prefix_idx < len(self.param_layers) - 1:
                next_layer = self.param_layers[layer_prefix_idx + 1]

    # def load_state_dict(self, state_dict, strict=True):
    #     _IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])
    #     missing_keys = []
    #     unexpected_keys = []
    #     error_msgs = []

    #     # copy state_dict so _load_from_state_dict can modify it
    #     metadata = getattr(state_dict, '_metadata', None)
    #     state_dict = state_dict.copy()
    #     if metadata is not None:
    #         state_dict._metadata = metadata

    #     def load(module, prefix=''):
    #         local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    #         module._load_from_state_dict(
    #             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    #         for name, child in module._modules.items():
    #             if child is not None:
    #                 load(child, prefix + name + '.')

    #     load(self)

    #     if strict:
    #         if len(unexpected_keys) > 0:
    #             error_msgs.insert(
    #                 0, 'Unexpected key(s) in state_dict: {}. '.format(
    #                     ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    #         if len(missing_keys) > 0:
    #             error_msgs.insert(
    #                 0, 'Missing key(s) in state_dict: {}. '.format(
    #                     ', '.join('"{}"'.format(k) for k in missing_keys)))

    #     if len(error_msgs) > 0:
    #         raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #             self.__class__.__name__, "\n\t".join(error_msgs)))
    #     return _IncompatibleKeys(missing_keys, unexpected_keys)

    # parameterized layers
    # @abstractmethod
    # def param_layers(self) -> list:
    #     pass
    #
    # @abstractmethod
    # def param_layer_prefixes(self) -> list:
    #     pass
    def traverse(self, criterion, layers: list, names: list):
        traverse_module(self, criterion, layers, names)

    def get_param_layers(self, layers: list, names: list, criterion=None):
        self.traverse(lambda x: len(list(x.parameters())) != 0, layers, names)

    @abstractmethod
    def collect_layers(self):
        pass

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    def loss(self, inputs, labels: torch.IntTensor) -> torch.FloatTensor:
        return self.loss_func(self(inputs), labels)

    @torch.no_grad()
    def evaluate(self, test_loader, mode="mean"):
        """
        给定数据集，计算 loss 和 准确率
        :param test_loader:
        :param mode:
        :return:
        """
        assert mode in ["sum", "mean"], "mode must be sum or mean"
        self.eval()
        test_loss = 0
        n_correct = 0
        n_total = 0
        device = next(self.parameters()).device

        idx = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self(inputs)
            # new_labels = torch.argmax(labels, 1)
            batch_loss = self.loss_func(outputs, labels)
            test_loss += batch_loss.item()

            labels_predicted = torch.argmax(outputs, dim=1)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)

            n_total += labels.size(0)
            n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()
            idx += 1

        if mode == "mean":
            test_loss /= idx
        self.train()
        return test_loss, n_correct / n_total

    # @torch.no_grad()
    # def apply_grad(self):
    #     for param in self.parameters():
    #         param.add_(param.grad, alpha=-self.lr)  # includes both sparse and dense

    # def step(self, inputs, labels):
    #     self.zero_grad()
    #     loss = self.loss(inputs, labels)
    #     loss.backward()
    #     self.apply_grad()

    def prune_by_threshold(self, thr_arg: Union[int, float, Sized]):
        """
        根据每一层的阈值，对可裁剪的层进行裁剪
        :param thr_arg:
        :return:
        """
        prunable_layers = self.prunable_layers
        if isinstance(thr_arg, Sized):
            assert len(prunable_layers) == len(thr_arg)
        else:
            thr_arg = [thr_arg] * len(prunable_layers)
        for thr, layer in zip(thr_arg, prunable_layers):
            if thr is not None:
                layer.prune_by_threshold(thr)

        return self

    def prune_by_rank(self, rank_arg: Union[int, float, Sized]):
        """
        根据每一层的rank，对可裁剪的层进行裁剪
        :param rank_arg:
        :return:
        """
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.prune_by_rank(rank)

        return self

    def retain_by_rank(self, rank_arg: Union[int, float, Sized]):
        """
        根据每一层的rank，对可裁剪层进行保留
        :param rank_arg:
        :return:
        """
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.retain_by_rank(rank)

        return self

    def prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        """
        根据每一层的比例，对可裁剪层进行裁剪
        :param pct_arg:
        :return:
        """
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.prune_by_pct(pct)

        return self

    @torch.no_grad()
    def structured_by_rank(self, rank_arg: Union[int, float, Sized], rate_arg: Union[int, float, Sized], idx: int,
                           device: torch.device):
        """
        给定 rank，对某一层进行结构化剪枝
        :param rank_arg:
        :param rate_arg:
        :param idx:
        :param device:
        :return:
        """
        assert len(self.prunable_layers) > idx
        if isinstance(rank_arg, Sized):
            assert len(rank_arg) > idx
            rank = rank_arg[idx]
        else:
            rank = rank_arg

        if isinstance(rate_arg, Sized):
            assert len(rate_arg) > idx
            rate = rate_arg[idx]
        else:
            rate = rate_arg

        layer = self.prunable_layers[idx]
        out_channels = layer.out_channels

        layer.structured_by_rank(rank, rate, device)
        ind = layer.ind

        if idx < len(self.prunable_layers) - 1:
            prunable_next_layer = self.prunable_layers[idx + 1]
            if is_fc(prunable_next_layer):
                num_each = int(prunable_next_layer.in_features / out_channels)
                new_ind = torch.ones((num_each * ind.shape[0]))
                for i_idx, i in enumerate(ind):
                    for j in range(num_each):
                        new_ind[i_idx * num_each + j] = int(i * num_each + j)
                ind = new_ind.long()
                ind = ind.to(prunable_next_layer.mask.device)

            prunable_next_layer.prune_in_channels(ind)

        key = self.prunable_layer_prefixes[idx]
        layer_prefix_idx = self.param_layer_prefixes.index(key)

        if layer_prefix_idx < len(self.param_layers) - 1:
            next_layer = self.param_layers[layer_prefix_idx + 1]

        return self, ind

    # @torch.no_grad()
    # def structured_after_unstructured(self, idx, ind, device):

    #     layer = self.prunable_layers[idx]

    #     layer.prune_out_channels(ind)

    #     if idx < len(self.prunable_layers) - 1:
    #         prunable_next_layer = self.prunable_layers[idx + 1]
    #         prunable_next_layer.prune_in_channels(ind)

    #     key = self.prunable_layer_prefixes[idx]
    #     layer_prefix_idx = self.param_layer_prefixes.index(key)
    #     if layer_prefix_idx < len(self.param_layers) - 1:
    #         next_layer = self.param_layers[idx + 1]
    #         if isinstance(next_layer, nn.BatchNorm2d):
    #             next_layer = nn.BatchNorm2d(len(ind))

    #     return self

    def recovery_layer(self, idx):
        """
        对某一层进行恢复
        :param idx:
        :return:
        """
        layer = self.prunable_layers[idx]

        layer.recovery_info()

        if idx < len(self.prunable_layers) - 1:
            prunable_next_layer = self.prunable_layers[idx + 1]
            prunable_next_layer.recovery_info()

        key = self.prunable_layer_prefixes[idx]
        layer_prefix_idx = self.param_layer_prefixes.index(key)

        if layer_prefix_idx < len(self.param_layers) - 1:
            next_layer = self.param_layers[layer_prefix_idx + 1]

        return self

    @torch.no_grad()
    def unstructured_by_rank(self, rank_arg: Union[int, float, Sized], rate_arg: Union[int, float, Sized], idx: int,
                             device: torch.device, baselinename: str):
        """
        给定 rank，对某一层进行非结构化剪枝
        :param rank_arg:
        :param rate_arg:
        :param idx:
        :param device:
        :return:
        """
        assert len(self.prunable_layers) > idx
        if isinstance(rank_arg, Sized):
            assert len(rank_arg) > idx
            rank = rank_arg[idx]
        else:
            rank = rank_arg

        if isinstance(rate_arg, Sized):
            assert len(rate_arg) > idx
            rate = rate_arg[idx]
        else:
            rate = rate_arg

        layer = self.prunable_layers[idx]

        layer.unstructured_by_rank(rank, rate, device, baselinename=baselinename)
        ind = layer.ind

        # if idx < len(self.prunable_layers) - 1:
        #     prunable_next_layer = self.prunable_layers[idx + 1]
        #     prunable_next_layer.uprune_in_channels(ind)

        # key = self.prunable_layer_prefixes[idx]
        # layer_prefix_idx = self.param_layer_prefixes.index(key)
        # if layer_prefix_idx < len(self.param_layers) - 1:
        #     next_layer = self.param_layers[idx + 1]
        #     if isinstance(next_layer, nn.BatchNorm2d):
        #         next_layer = nn.BatchNorm2d(len(ind))

        return self, ind

    def random_prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        """
        给定每一层的剪枝比例，对每一层进行随机剪枝
        :param pct_arg:
        :return:
        """
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.random_prune_by_pct(pct)

        return self

    @torch.no_grad()
    def reinit_from_model(self, final_model):
        """
        从 final_model 中恢复当前模型的 mask
        :param final_model:
        :return:
        """
        assert isinstance(final_model, self.__class__)
        for self_layer, layer in zip(self.prunable_layers, final_model.prunable_layers):
            self_layer.mask = layer.mask.clone().to(self_layer.mask.device)

    def calc_num_prunable_params(self, count_bias):
        """
        计算可剪枝层的 实际使用的参数量 和 全部参数量
        :param count_bias:
        :return:
        """
        total_param_in_use = 0
        total_param = 0
        for layer in self.prunable_layers:
            num_bias = layer.bias.nelement() if layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight
            num_params_in_use = num_weight + num_bias
            num_params = layer.weight.nelement() + num_bias
            total_param_in_use += num_params_in_use
            total_param += num_params

        return total_param_in_use, total_param

    def calc_num_all_active_params(self):
        """
        计算总得实际使用的参数量
        :param count_bias:
        :return:
        """
        total_param = 0
        for layer in self.param_layers:
            # num_bias = layer.bias.nelement() if layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight if hasattr(layer, "num_weight") else layer.weight.nelement()
            # num_params = num_weight + num_bias
            total_param += num_weight

        return total_param

    def nnz(self, count_bias=False):
        # number of parameters in use in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[0]

    def nelement(self, count_bias=False):
        # number of all parameters in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[1]

    def density(self, count_bias=False):
        total_param_in_use, total_param = self.calc_num_prunable_params(count_bias=count_bias)
        return total_param_in_use / total_param

    def _get_module_by_name_list(self, module_names: list):
        module = self
        for name in module_names:
            module = getattr(module, name)
        return module

    def get_module_by_name(self, module_name: str):
        return self._get_module_by_name_list(module_name.split('.'))

    def get_mask_by_name(self, param_name: str):
        if param_name.endswith("bias"):
            return None
        module = self._get_module_by_name_list(param_name.split('.')[:-1])
        return module.mask if hasattr(module, "mask") else None

    # @abstractmethod
    # def to_sparse(self):
    #     pass

    def get_channels(self):
        channels = []
        for layer in self.prunable_layers:
            if is_conv(layer):
                if layer.ind is None:
                    min_filters = min(layer.out_channels, layer.rank_mask.sum().item())
                    channels.append(min_filters)
                else:
                    channels.append(layer.ind.shape[0])
        return channels
    def get_nearons(self):
        nearons = []
        for layer in self.prunable_layers:
            # TODO:记录剪枝后神经元的个数
            if is_fc(layer):
                if layer.ind is None:
                    min_feature = min(layer.out_features, layer.rank_mask.sum().item())
                    nearons.append(min_feature)
                else:
                    nearons.append(layer.ind.shape[0])
        return nearons
    def recovery_model(self):
        for layer in self.prunable_layers:
            layer.recovery_mask()

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            # move data to device
            for m in self.param_layers:
                if is_conv(m) or is_fc(m):
                    m.move_data(device)
                else:
                    m.to(device)
        return super(BaseModel, self).to(*args, **kwargs)