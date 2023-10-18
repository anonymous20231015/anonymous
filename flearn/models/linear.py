import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.nn.parameter import Parameter
import math

class DenseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, use_bias=True, use_mask=False, **kwargs):
        super(DenseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if use_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(**kwargs)

        # self._initial_weight = self.weight.data.clone()
        # self._initial_bias = self.bias.data.clone() if use_bias else None
        self.use_mask = use_mask
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.use_bias = use_bias
        self.bias_mask = torch.ones_like(self.bias, dtype=torch.bool)
        self.rank_mask = torch.ones_like(self.bias).unsqueeze(-1)
        self.mask_dp = torch.ones_like(self.bias).unsqueeze(-1)
        self.ind = None
        self.prune_rate = 0

        self.save_weight_data = None
        self.save_weight_grad = None

        self.save_bias_data = None
        self.save_bias_grad = None

        self.save_mask = None
        self.save_bias_mask = None

    def reset_parameters(self, **kwargs):
        if len(kwargs.keys()) == 0:
            # default init, see https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            init.kaiming_uniform_(self.weight, **kwargs)

        if self.bias is not None:
            # default init, see https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp: torch.Tensor):
        """
        前向传播前，将对应 mask 位置的权重设置为 0
        :param inp:
        :return:
        """
        # print("size:",self.weight.dtype, self.mask.dtype, self.mask_dp.dtype)
        # print("size:",self.weight.size(), self.mask.size(), self.mask_dp.size())
        # print("size:", self.weight.shape, self.mask.shape, self.mask_dp.shape)
        masked_weight = self.weight * self.mask * self.mask_dp if self.use_mask else self.weight
        masked_bias = self.bias
        # print(masked_bias.shape, masked_weight.shape)
        if self.use_bias and self.bias is not None and self.bias_mask is not None:
            masked_bias = masked_bias * self.bias_mask * (self.mask_dp.squeeze())
        return nn.functional.linear(inp, masked_weight, masked_bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def prune_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weight_val = self.weight[self.mask == 1.]
        sorted_abs_weight = weight_val.abs().sort()[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def retain_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def retain_by_rank(self, rank):
        weights_val = self.weight[self.mask == 1.]
        sorted_abs_weights = weights_val.abs().sort(descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand(size=self.mask.size(), device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = rand_val.sort()[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    def recovery_mask(self):
        """
        恢复 mask
        :return:
        """
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.bias_mask = torch.ones_like(self.bias, dtype=torch.bool)

    # def reinitialize(self):
    #     self.weight = Parameter(self._initial_weight)
    #     if self._initial_bias is not None:
    #         self.bias = Parameter(self._initial_bias)

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)
        self.bias_mask = self.bias_mask.to(device)
        self.rank_mask = self.rank_mask.to(device)
        self.mask_dp = self.mask_dp.to(device)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]

        if device is not None:
            self.move_data(device)

        return super(DenseLinear, self).to(*args, **kwargs)

    def unstructured_by_rank(self, rank, rate, device: torch.device, baselinename = None, mode=1):
        """
        根据 rank 进行非结构化剪枝
        :param rank:
        :param rate:
        :param device:
        :return:
        """
        #TODO：修改mask数值为1/(1-p)
        ones_re, zeros_re = torch.ones_like(rank), torch.zeros_like(rank)
        if baselinename == "AdapDrop":
            rank = rank * rate
            rank_list = rank.tolist()
            rank_symbol = torch.zeros_like(rank)
            mask_dp = torch.zeros_like(rank)
        else:
            rank_list = rank.tolist()
            rank_symbol = torch.zeros_like(rank)
            mask_dp = torch.zeros_like(rank)
        for i in range(len(rank_list)):
            prune_symbol = [0, 1]
            symbol_rate = [rank_list[i], 1-rank_list[i]]
            if rank_list[i] >= 1:
                rank_symbol[i] = 0
            elif rank_list[i] <= 0:
                rank_symbol[i] = 1
                mask_dp[i] = 1
            else:
                rank_symbol[i] = torch.from_numpy(np.random.choice(a=prune_symbol, size=1, replace=False, p=symbol_rate))
                mask_dp[i] = 1 / (1 - rank_list[i])
        pruned_count = (rank_symbol <= 0).sum()
        ind = torch.argsort(rank)[pruned_count:].to(device)  # preserved filter id
        ind, _ = torch.sort(ind)
        self.ind = ind
        self.prune_rate = 1 - (ind.shape[0] / rank_symbol.shape[0])
        rank_mask = torch.where(rank_symbol > 0, ones_re, zeros_re).to(device)
        rank_mask = rank_mask > 0

        rank_mask = rank_mask.unsqueeze(-1)
        mask_dp = mask_dp.unsqueeze(-1).to(device)
        self.mask_dp = mask_dp
        self.rank_mask = rank_mask
        self.uprune_out_channels(self.rank_mask, device)

    def uprune_out_channels(self, rank_mask, device: torch.device):
        """
        实际的非结构化剪枝，更新 mask，以及对应位置的权重置 0
        :param rank_mask:
        :param device:
        :return:
        """
        print(self.mask.shape, rank_mask.shape)

        if rank_mask is not None:
            self.mask = self.mask * rank_mask

        # self.weight.data *= self.mask
        #
        # if self.weight.grad is not None:
        #     self.weight.grad *= self.mask

        if self.bias is not None and rank_mask is not None:
            if self.bias_mask is None:
                self.bias_mask = torch.zeros_like(self.bias, dtype=torch.bool).to(device)
            self.bias_mask = self.bias_mask * (rank_mask.squeeze())
            # self.bias.data *= self.bias_mask

            # if self.bias.grad is not None:
            #     self.bias.grad *= self.bias_mask

    def structured_by_rank(self, rank, rate, device: torch.device):
        f, _ = self.weight.size()
        pruned_num = int(rate * f)
        ind = torch.argsort(rank)[pruned_num:].to(device)  # preserved filter id
        ind, _ = torch.sort(ind)

        self.ind = ind

        self.prune_out_channels(ind)

    def save_layer_info(self):
        self.save_weight_data = self.weight.data.clone()
        self.save_weight_grad = self.weight.grad.clone() if self.weight.grad is not None else None

        self.save_bias_data = self.bias.data.clone() if self.bias is not None else None
        self.save_bias_grad = self.bias.grad.clone() if self.bias is not None and self.bias.grad is not None else None

        self.save_mask = self.mask.clone()
        self.save_bias_mask = self.bias_mask.clone()

    def recovery_info(self):
        self.weight.data = self.save_weight_data if self.save_weight_data is not None else self.weight.data
        self.weight.grad = self.save_weight_grad if self.save_weight_grad is not None else self.weight.grad

        self.mask = self.save_mask

        if self.bias is not None:
            self.bias.data = self.save_bias_data if self.save_bias_data is not None else self.bias.data
            self.bias.grad = self.save_bias_grad if self.save_bias_grad is not None else self.bias.grad
            self.bias_mask = self.save_bias_mask if self.save_bias_mask is not None else self.bias_mask

        self.out_features = self.weight.data.shape[0]
        self.in_features = self.weight.data.shape[1]

    def prune_out_channels(self, ind):
        self.save_layer_info()

        self.weight.data = torch.index_select(self.weight.data, 0, ind)

        if self.weight.bias is not None:
            self.weight.grad = torch.index_select(self.weight.grad, 0, ind)

        self.mask = torch.index_select(self.mask, 0, ind)

        self.weight *= self.mask

        if self.bias is not None:
            self.bias.data = torch.index_select(self.bias.data, 0, ind)
            if self.bias.grad is not None:
                self.bias.grad = torch.index_select(self.bias.grad, 0, ind)

        self.out_features = len(ind)

    def prune_in_channels(self, ind):
        self.save_layer_info()

        self.weight.data = torch.index_select(self.weight.data, 1, ind)

        if self.weight.grad is not None:
            self.weight.grad = torch.index_select(self.weight.grad, 1, ind)

        self.mask = torch.index_select(self.mask, 1, ind)
        self.in_features = len(ind)

        self.weight *= self.mask

    @property
    def num_weight(self) -> int:
        return self.mask.sum().item() + self.bias_mask.sum().item()

if __name__=="__main__":
    a = DenseLinear(512, 512, use_mask=True)
    rank = torch.linspace(1, 64, steps=64)
