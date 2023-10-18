import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

class DenseConv2d(torch.nn.Conv2d):
    """
    为卷积层包装上一层 mask 变量，让其方便进行剪枝
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True,
                 padding_mode='zeros', mask: torch.FloatTensor = None, use_mask=False):
        super(DenseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                          dilation, groups, use_bias, padding_mode)
        if mask is None:
            self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        else:
            self.mask = mask
            assert self.mask.size() == self.weight.size()

        self.use_mask = use_mask
        self.dropout_rate = 0

        self.ind = None
        self.bias_mask = torch.ones_like(self.bias, dtype=torch.bool)

        self.rank_mask = torch.ones_like(self.bias_mask, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.mask_dp = torch.ones_like(self.bias_mask, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.use_bias = use_bias

        self.save_weight_data = None
        self.save_weight_grad = None

        self.save_bias_data = None
        self.save_bias_grad = None

        self.save_mask = None
        self.save_bias_mask = None


    """
    def conv2d_forward(self, inp, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            inp = F.pad(inp, expanded_padding, mode='circular')
            padding = _pair(0)
        else:
            padding = self.padding

        return DenseConv2dFunction.apply(inp, weight, self.kernel_size, self.bias, self.stride, padding)
    """

    def conv2d_forward(self, input, weight, bias):
        """
        进行前向传播
        :param input:
        :param weight:
        :param bias:
        :return:
        """
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


    def forward(self, inp):
        """
        前向传播前，将对应 mask 位置的权重设置为 0
        :param inp:
        :return:
        """
        masked_weight = self.weight * self.mask * self.mask_dp if self.use_mask else self.weight
        masked_bias = self.bias

        if self.use_bias and self.bias is not None and self.bias_mask is not None:
            masked_bias = masked_bias * self.bias_mask * (self.mask_dp.squeeze())

        return self.conv2d_forward(inp, masked_weight, masked_bias)

    def prune_by_threshold(self, thr):
        """
        小于某个阈值的 mask 全都设置为 0
        :param thr:
        :return:
        """
        self.mask *= (torch.abs(self.weight) >= thr)

    def retain_by_threshold(self, thr):
        """
        这不是一样吗
        :param thr:
        :return:
        """
        self.mask *= (torch.abs(self.weight) >= thr)

    def prune_by_rank(self, rank):
        """
        根据 rank 值来进行裁剪，将权重从小到大排序，第 rank 个的值作为裁剪的阈值
        :param rank:
        :return:
        """
        if rank == 0:
            return
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def retain_by_rank(self, rank):
        """
        降序排列，第 rank 个的值作为保留的阈值
        :param rank:
        :return:
        """
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val), descending=True)[0]
        thr = sorted_abs_weights[rank]
        self.retain_by_threshold(thr)

    def prune_by_pct(self, pct):
        """
        根据一定的百分比，计算出 rank，之后裁剪
        :param pct:
        :return:
        """
        if pct == 0:
            return
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        """
        根据比例，随机裁剪
        :param pct:
        :return:
        """
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand_like(self.mask, device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)
        self.bias_mask = self.bias_mask.to(device)
        self.rank_mask = self.bias_mask.to(device)
        self.mask_dp = self.mask_dp.to(device)

    def unstructured_by_rank(self, rank, rate, device: torch.device, baselinename = None, mode=1):
        """
        根据 rank 进行非结构化剪枝
        :param rank:
        :param rate:
        :param device:
        :return:
        """
        f, _, _, _ = self.weight.size()
        ones_re, zeros_re = torch.ones_like(rank), torch.zeros_like(rank)
        mask_dp = torch.zeros_like(rank)
        if baselinename == "AdapDrop":
            rank = rank * rate
            rank_list = rank.tolist()
            rank_symbol = torch.zeros_like(rank)
        else:
            rank_list = rank.tolist()
            rank_symbol = torch.zeros_like(rank)

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
        ind = torch.argsort(rank_symbol)[pruned_count:].to(device)
        ind, _ = torch.sort(ind)
        self.ind = ind
        rank_mask = torch.where(rank_symbol > 0, ones_re, zeros_re).to(device)
        rank_mask = rank_mask > 0
        rank_mask = rank_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask_dp = mask_dp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        self.mask_dp = mask_dp
        self.rank_mask = rank_mask
        self.uprune_out_channels(self.rank_mask, device)

    def structured_by_rank(self, rank, rate, device: torch.device, mode=1):
        """
        根据 rank 进行结构化剪枝
        :param rank:
        :param rate:
        :param device:
        :param mode: 0 表示以最大值和最小值的之间某个百分比进行阈值裁剪， 1 表示直接裁剪百分比数量的通道
        :return:
        """
        f, _, _, _ = self.weight.size()
        min_rank, max_rank = torch.min(rank), torch.max(rank)
        pruned_num = (max_rank - min_rank) * rate + min_rank
        pruned_count = (rank <= pruned_num).sum()
        if mode == 1:
            pruned_count = int(rate * f)

        print("该层剪枝率：{}".format(pruned_count / f))
        ind = torch.argsort(rank)[pruned_count:].to(device)  # preserved filter id
        ind, _ = torch.sort(ind)

        self.ind = ind

        self.prune_out_channels(ind)

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

    def recovery_mask(self):
        """
        恢复 mask
        :return:
        """
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.bias_mask = torch.ones_like(self.bias, dtype=torch.bool)


    def save_layer_info(self):
        """
        保存当前层的权重以及 mask
        :return:
        """
        self.save_weight_data = self.weight.data.clone()
        self.save_weight_grad = self.weight.grad.clone() if self.weight.grad is not None else None

        self.save_bias_data = self.bias.data.clone() if self.bias is not None else None
        self.save_bias_grad = self.bias.grad.clone() if self.bias is not None and self.bias.grad is not None else None

        self.save_mask = self.mask.clone()
        self.save_bias_mask = self.bias_mask.clone()

    def recovery_info(self):
        """
        恢复当前层的权重以及 mask
        :return:
        """
        self.weight.data = self.save_weight_data if self.save_weight_data is not None else self.weight.data
        self.weight.grad = self.save_weight_grad if self.save_weight_grad is not None else self.weight.grad

        self.mask = self.save_mask

        if self.bias is not None:
            self.bias.data = self.save_bias_data if self.save_bias_data is not None else self.bias.data
            self.bias.grad = self.save_bias_grad if self.save_bias_grad is not None else self.bias.grad
            self.bias_mask = self.save_bias_mask if self.save_bias_mask is not None else self.bias_mask

        self.out_channels = self.weight.data.shape[0]
        self.in_channels = self.weight.data.shape[1]

    def prune_out_channels(self, ind):
        """
        根据卷积层的索引，裁剪掉对应的出口 filter, 权重和 mask 都会进行实际裁剪
        :param ind:
        :return:
        """
        self.save_layer_info()

        self.weight.data = torch.index_select(self.weight.data, 0, ind)

        if self.weight.grad is not None:
            self.weight.grad = torch.index_select(self.weight.grad, 0, ind)

        self.mask = torch.index_select(self.mask, 0, ind)

        self.weight *= self.mask

        if self.bias is not None:
            self.bias.data = torch.index_select(self.bias.data, 0, ind)
            if self.bias_mask is not None:
                self.bias_mask = torch.index_select(self.bias_mask, 0, ind)
            if self.bias.grad is not None:
                self.bias.grad = torch.index_select(self.bias.grad, 0, ind)

        self.out_channels = len(ind)

    def prune_in_channels(self, ind):
        """
        根据卷积层的索引，裁剪掉对应的入口 filter，权重和 mask 都会进行实际裁剪
        :param ind:
        :return:
        """
        self.save_layer_info()

        self.weight.data = torch.index_select(self.weight.data, 1, ind)

        if self.weight.grad is not None:
            self.weight.grad = torch.index_select(self.weight.grad, 1, ind)

        self.mask = torch.index_select(self.mask, 1, ind)
        self.in_channels = len(ind)

        self.weight *= self.mask

    @property
    def num_weight(self):
        return torch.sum(self.mask).int().item() + torch.sum(self.bias_mask).int().item()

if __name__=="__main__":
    f = [[[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]],
         [[1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]],
         [[1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]]
         ]
    a = DenseConv2d(3, 64, kernel_size=3, padding=1, use_mask=True)
    print(a.num_weight)
    a.random_prune_by_pct(0.1)
    print(a)
    # rank = torch.linspace(1, 64, steps=64)
    a.unstructured_by_rank(rank, 0.6, "cpu")
    # a.recovery_mask()
    # ind = a.ind
    # a.prune_in_channels(ind)
    # print(a.ind.shape[0])
    # a.structured_by_rank(rank, 0.6, "cpu")
    # e = np.array(f)
    # g = torch.index_select(torch.tensor(e), 1, torch.tensor([1, 2]))
    # print(g)
    # print(g.shape)
    # mask = torch.ones_like(g)
    # print(mask)
    # mask *= (g > 3)
    # print(mask)
    # g_val = g[mask == 1]
    # print(g_val)
    # g_val_sort = torch.sort(g_val)
    # print(g_val_sort)
    # print(g_val_sort[0])
    # print(g_val.unsqueeze(-1).unsqueeze(-1).squeeze())
