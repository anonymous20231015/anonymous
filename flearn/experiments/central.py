import torch
import numpy as np
import random
from tqdm import tqdm
from flearn.utils.update import LocalUpdate
from flearn.utils.model_utils import average_weights, test_inference, is_conv, is_fc,ratio_combine, ratio_minus
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.utils.prune import hrank_prune, get_ratio
from flearn.models import cnn, vgg, resnet, lenet
from data.util import get_global_distribution, get_target_users_distribution, \
    get_noniid_degree, get_distribution, get_client_noniid_degree
import copy
import os
import time
import math


class CentralTraining(object):
    """
    对于聚合后的模型，进行中心化的训练，share_percent 是共享数据集的大小
    """

    def __init__(self, args, equal=False, unequal=False, unbalance=False, dirichlet=False, l=2,
                prune_rate=0.6, auto_rate=False, result_dir="FedAD",
                 ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

        # 设置随机种子
        self.reset_seed()

        # 数据集划分信息
        self.num_data = 50000
        self.l = l  # noniid的程度， l越小 noniid 程度越大， 当 l = 1 时， 就是将数据按照顺序分成 clients 份，每个设备得到一份， 基本上只包含一个数字

        # 定义FedAvg的一些参数
        self.m = 10  # 每次取m个客户端进行平均
        self.equal = equal
        self.unequal = unequal
        self.unbalance = unbalance
        self.dirichlet = dirichlet
        self.decay = self.args.decay

        # 剪枝的参数
        self.channels_list = []
        self.nearons_list = []
        self.conv_layer_idx = []
        self.fc_layer_idx = []
        self.current_prune_idx = 0
        self.pre_prune_round = 0
        self.recovery_round = None
        self.valid_loss_list = []
        self.valid_acc_list = []
        self.prune_rate = self.args.prune_rate
        self.auto_rate = auto_rate
        self.init_compress_rate = [prune_rate] * 3
        self.compress_rate = [prune_rate] * 3
        self.init_fc_compress_rate = [prune_rate] * 2
        self.fc_compress_rate = [prune_rate] * 2
        self.pre_acc = 1
        self.num_weights = []
        self.log_weight = []
        self.log_weights = []
        #filter剪枝概率
        self.dropout_rate = []

        self.lr_decay = 0.99
        self.init_lr = self.args.lr

        #Adaptive Dropout参数
        self.scoremap = [[[],[]]]*self.args.num_users
        self.latest_loss = [0]*self.args.num_users
        self.Recorded = False
        self.Ac = [[], []]

        #FedAD参数
        self.dropout_rates = [None] *100
        self.drop_epoch_users = None

    def reset_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def init_data(self):
        if self.args.dataset == "cifar10":
            from data.cifar10.cifar10_data import get_dataset
            self.num_data = 50000
            train_dataset, test_dataset, user_groups = get_dataset(num_data=self.num_data, num_users=self.args.num_users, equal=self.equal, unequal = self.unequal, unbalance=self.unbalance, dirichlet=self.dirichlet, l=self.l)
        elif self.args.dataset == "mnist":
            from data.mnist.mnist_data import get_dataset
            self.num_data = 50000
            self.num_share = int(self.num_data * self.share_percent / 100)
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            num_share=self.num_share,
                            l=self.l, unequal=self.unequal, share_l=self.args.share_l)
        elif self.args.dataset == "fashionmnist":
            from data.fashionmnist.fashionmnist_data import get_dataset
            self.num_data = 60000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users,
                            dirichlet=self.dirichlet)
        elif self.args.dataset == "svhn":
            from data.svhn.svhn_data import get_dataset
            self.num_data = 73257
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users,
                            dirichlet=self.dirichlet)
        elif self.args.dataset == "cifar100":
            from data.cifar100.cifar100_data import get_dataset
            self.num_data = 50000
            self.l = self.l * 10
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, equal=self.equal,
                            unequal=self.unequal, dirichlet=self.dirichlet,
                            l=self.l)
        elif self.args.dataset == "tinyimagenet":
            from data.tinyImagenet.tinyimagenet_data import get_dataset
            self.num_data = 100000
            train_dataset, test_dataset, user_groups = get_dataset(num_data=100000, num_users=self.args.num_users, dirichlet=self.dirichlet)
        else:
            exit('Error: unrecognized dataset')

        #self.num_share = len(user_groups[self.args.num_users])
        #self.share_percent = math.ceil(self.num_share / self.num_data * 100)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_groups = user_groups
        self.global_distribution = get_global_distribution(train_dataset, user_groups)

        return train_dataset, test_dataset, user_groups

    def load_model(self):
        # BUILD MODEL
        # Convolutional neural network
        if self.args.dataset == "cifar10" or self.args.dataset == "svhn":
            in_channel = 3
            num_classes = 10
        elif self.args.dataset == "mnist" or self.args.dataset == "fashionmnist":
            self.init_lr = 0.01
            self.args.lr = 0.01
            in_channel = 1
            num_classes = 10
        elif self.args.dataset == "cifar100":
            in_channel = 3
            num_classes = 100
        elif self.args.dataset == "tinyimagenet":
            in_channel = 3
            num_classes = 200
        else:
            exit('Error: unrecognized dataset')
        if self.args.model == "vgg":
            global_model = vgg.VGG11(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "resnet":
            global_model = resnet.resnet18(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "lenet":
            global_model = lenet.LENET(in_channel=in_channel, num_classes=num_classes)
        else:
            global_model = cnn.CNN(in_channel=in_channel, num_classes=num_classes)

        self.load_model_info(global_model)
        self.num_weights.append(global_model.calc_num_all_active_params())
        self.channels_list.append(global_model.get_channels())
        self.log_weights.append([0, self.num_weights[-1], self.channels_list[-1]])

        # Set the model to train and send it to device.
        global_model.to(self.device)
        global_model.train()
        return global_model

    def load_model_info(self, model):
        """
        加载模型信息，确定可剪枝层的索引，并按照参数量进行降序排序
        :param model:
        :return:
        """
        num_trainables = []
        for i, layer in enumerate(model.prunable_layers):
            if is_conv(layer):
                self.conv_layer_idx.append(i)
                num_trainables.append(layer.num_weight)
            if is_fc(layer):
                self.fc_layer_idx.append(i)
        if self.auto_rate:
            self.init_compress_rate = [0] * (len(self.conv_layer_idx))
            self.compress_rate = [0] * (len(self.conv_layer_idx))
        else:
            self.init_compress_rate = [self.prune_rate] * len(self.conv_layer_idx)
            self.compress_rate = [self.prune_rate] * len(self.conv_layer_idx)
        self.sorted_conv_layers = self.conv_layer_idx
        self.sorted_fc_layers = self.fc_layer_idx
        # self.sorted_conv_layers = np.argsort(num_trainables)[::-1]
        self.num_trainables = num_trainables
        print(f"prunable layer idx: {self.conv_layer_idx}")
        print(f"sorted_layers: {self.sorted_conv_layers} according: {num_trainables}")

    def record_base_message(self, log_path):
        record_log(self.args, log_path, "=== " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " ===\n")
        record_log(self.args, log_path, f"=== model: {self.args.model} ===\n")
        record_log(self.args, log_path, f"=== dataset: {self.args.dataset} ===\n")
        client_noniid_degree = get_client_noniid_degree(self.train_dataset, self.user_groups)
        record_log(self.args, log_path,
                   f"=== noniid_degree client: {client_noniid_degree}  ===\n")
        record_log(self.args, log_path,
                   f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===\n")
        record_log(self.args, log_path,
                   f"=== prune_ratio: {self.prune_rate if not self.auto_rate else 'auto rate'} ===\n")
        record_log(self.args, log_path, f"=== prunable_layer_idx: {self.conv_layer_idx} ===\n")
        record_log(self.args, log_path,
                   f"=== sorted_layers: {self.sorted_conv_layers} according: {self.num_trainables} ===\n")

    def print_info(self, user_groups=None):
        if user_groups is None:
            user_groups = [[]]
        print(f"data name: {self.args.dataset}")
        print(f"seed: {self.args.seed}")
        print(f"=== model: {self.args.model} ===\n")
        print(f"user nums: {self.args.num_users}")
        print(f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===")
        print(f"=== prune_ratio: {self.prune_rate if not self.auto_rate else 'auto rate'} ===\n")
        print(f"=== using device {self.device} optim {self.args.optim} ===")

    def get_loss(self, all_trian_data, train_dataset, global_model):
        """
        获取所有训练数据的 loss
        :param user_groups:
        :param train_dataset:
        :param global_model:
        :return:
        """

        # losses = []
        # for idx in range(len(user_groups)):
        #     if user_groups[idx].shape[0] == 0:
        #         continue
        #     local_model = LocalUpdate(args=self.args, dataset=train_dataset,
        #                               idxs=user_groups[idx], device=self.device)
        #     acc, loss = local_model.inference(global_model)
        #     losses.append(loss)
        # loss = sum(losses) / len(losses)
        local_model = LocalUpdate(args=self.args, local_bs=128, dataset=train_dataset,
                                  idxs=all_trian_data, device=self.device)
        acc, loss = local_model.inference(global_model)
        return round(loss, 4)

    def client_train(self, idxs_users, users_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses):
        """
        进行客户端训练
        :param local_v:
        :param local_P:
        :param idxs_users:
        :param global_model:
        :param user_groups:
        :param epoch:
        :param train_dataset:
        :param train_losses:
        :param local_weights:
        :param local_losses:
        :return:
        """
        idxs_users_list = idxs_users.tolist()
        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)
        #TODO:获取最长训练时间的设备和训练时间
        # 计算 mu 值
        for idx in idxs_users:
            global_model = copy.deepcopy(users_model[idxs_users_list.index(idx)])
            local_model = LocalUpdate(args=self.args, dataset=train_dataset, idxs=user_groups[idx], device=self.device)
            w, loss, grad = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            if loss < train_losses[0] * 3:
                local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                local_losses.append(copy.deepcopy(loss))
            print("{}:{:.4f}".format(idx, loss), end=" ")
        print("本轮设备总用时：{:.4f}".format(time.time() - start))
        print()

    def adapdrop_train(self, idxs_users, users_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses):
        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)
        for idx in idxs_users:
            global_model = copy.deepcopy(users_model)
            global_model.train()
            local_model = LocalUpdate(args=self.args, dataset=train_dataset, idxs=user_groups[idx], device=self.device)

            w, loss, grad = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch
                                                 )
            if self.latest_loss[idx] == 0:
                self.latest_loss[idx] = loss
            else:
                if loss < self.latest_loss[idx]:
                    self.Ac = [[], []]
                    score = (self.latest_loss[idx] - loss) / self.latest_loss[idx]
                    if len(self.scoremap[idx][0]) == 0:
                        for layer in self.sorted_conv_layers:
                            self.Ac[0].append(global_model.prunable_layers[layer].rank_mask)
                            self.scoremap[idx][0].append(global_model.prunable_layers[layer].rank_mask * score)
                        for layer in self.sorted_fc_layers:
                            self.Ac[1].append(global_model.prunable_layers[layer].rank_mask)
                            self.scoremap[idx][1].append(global_model.prunable_layers[layer].rank_mask * score)
                        self.Recorded = True
                    else:
                        for ind, layer in enumerate(self.sorted_conv_layers):
                            self.Ac[0].append(global_model.prunable_layers[layer].rank_mask)
                            self.scoremap[idx][0][ind] += global_model.prunable_layers[layer].rank_mask * score
                        for ind, layer in enumerate(self.sorted_fc_layers):
                            self.Ac[1].append(global_model.prunable_layers[layer].rank_mask)
                            self.scoremap[idx][1][ind] += global_model.prunable_layers[layer].rank_mask * score
                        self.Recorded = True
                else:
                    self.Recorded = False
                self.latest_loss[idx] = loss
            if loss < train_losses[0] * 3:
                local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                local_losses.append(copy.deepcopy(loss))
            print("{}:{:.4f}".format(idx, loss), end=" ")
        print("本轮设备总用时：{:.4f}".format(time.time() - start))
        print()

    def check_pruning_round(self, cur_round):
        return self.pre_prune_round + self.prune_interval == cur_round and \
               self.current_prune_idx < len(self.conv_layer_idx)

    def check_recovery_round(self, cur_round):
        return self.recovery_round is not None and \
               cur_round == self.recovery_round and self.current_prune_idx < len(self.conv_layer_idx)


if __name__ == "__main__":
    args = args_parser()
    t = CentralTraining(args, unequal=False, prune_rate=0.6, auto_rate=True)
