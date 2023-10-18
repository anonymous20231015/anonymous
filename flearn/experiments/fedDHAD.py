import copy
import math
import os
import random
import time
import torch
from torch import nn
import sys
import numpy as np

from data.util import get_global_distribution, get_client_noniid_degree, get_target_users_distribution, \
    get_noniid_degree
from flearn.utils.model_utils import test_inference, gradient_norm, is_conv, is_fc
from flearn.utils.options import args_parser
from flearn.utils.prune import get_ratio, hrank_prune, get_all_feature, get_weight
from flearn.utils.update import LocalUpdate
from flearn.utils.util import record_log, save_result
from flearn.models import cnn, vgg, resnet, lenet
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class FedDHAD(object):
    """"
    Estimate Non-IID degree Based on Gradients
    """
    def __init__(self, args, iid, equal=True, unequal=False, unbalance=False, dirichlet=False, l=2, result_dir="fedDHAD"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

        # 设置随机种子
        self.reset_seed()

        # 数据集划分
        self.num_data = 50000
        self.l = l  # noniid的程度， l越小 noniid 程度越大， 当 l = 1 时， 就是将数据按照顺序分成 clients 份，每个设备得到一份， 基本上只包含一个数字

        # 定义FedDH参数
        self.m = 10 #10
        self.iid = iid
        self.equal = equal
        self.unequal = unequal
        self.unbalance = unbalance
        self.dirichlet = dirichlet
        self.decay = self.args.decay
        self.lambda_decay = self.args.lambda_decay
        self.b_decay = self.args.b_decay
        self.lambdas = [(float(1.0)) for i in range(self.args.num_users)]
        self.b = [(float(0.0)) for i in range(self.args.num_users)]
        self.result_dir = result_dir

        # 定义FedAD参数
        self.conv_rank = []
        self.fc_weight = []
        self.avg_rank_weight = []
        self.rank_weight = []
        self.pre_model = None
        self.pre_diff = None
        self.start_dropout = False
        self.find_first_min_diff = False
        self.get_dropout_tatios = []
        self.channels_list = []
        self.nearons_list = []
        self.conv_layer_idx = []
        self.fc_layer_idx = []
        self.current_prune_idx = 0
        self.pre_prune_round = 0
        self.recovery_round = None
        self.valid_loss_list = []
        self.valid_acc_list = []
        self.pre_acc = 1
        self.num_weights = []
        self.log_weight = []
        self.log_weights = []
        # filter剪枝概率
        self.dropout_rate = []
        self.dropout_rates = [None] * 100
        self.drop_epoch_users = None

        self.lr_decay = 0.99
        self.init_lr = self.args.lr
        self.lr_lambda = self.args.lr_lambda
        self.lr_b = self.args.lr_b

        self.client_losses = [list() for i in range(self.args.num_users)]
        self.niid_degree = []

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
            train_dataset, test_dataset, user_groups = get_dataset(num_data=self.num_data,
                                                    num_users=self.args.num_users, equal=self.equal, unequal=self.unequal, unbalance=self.unbalance, dirichlet=self.dirichlet,  l=self.l)
        elif self.args.dataset == "mnist":
            from data.mnist.mnist_data import get_dataset
            self.num_data = 50000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            l=self.l, unequal=self.unequal)
        elif self.args.dataset == "svhn":
            from data.svhn.svhn_data import get_dataset
            self.num_data = 73257
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users,
                            dirichlet=self.dirichlet)
        elif self.args.dataset == "fashionmnist":
            from data.fashionmnist.fashionmnist_data import get_dataset
            self.num_data = 60000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users,
                            dirichlet=self.dirichlet)
        elif self.args.dataset == "cifar100":
            from data.cifar100.cifar100_data import get_dataset
            self.num_data = 50000
            self.l = self.l * 10
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, equal=self.equal, unequal=self.unequal, dirichlet=self.dirichlet,
                            l=self.l)
        elif self.args.dataset == "tinyimagenet":
            from data.tinyImagenet.tinyimagenet_data import get_dataset
            self.num_data = 100000
            train_dataset, test_dataset, user_groups = get_dataset(num_data=100000, num_users=self.args.num_users, dirichlet=self.dirichlet)
        else:
            exit('Error: unrecognized dataset')
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_groups = user_groups
        self.global_distribution = get_global_distribution(train_dataset, user_groups)
        return train_dataset, test_dataset, user_groups

    def load_model(self):
        # BUILD MODEL
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
        elif self.args.model == "alexnet":
            global_model = alexnet.Alexnet(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "lenet":
            global_model = lenet.LENET(in_channel=in_channel, num_classes=num_classes)
        else:
            global_model = cnn.CNN(in_channel=in_channel, num_classes=num_classes)
        self.load_model_info(global_model)
        self.num_weights.append(global_model.calc_num_all_active_params())
        self.channels_list.append(global_model.get_channels())
        self.log_weights.append([0, self.num_weights[-1], self.channels_list[-1]])

        # SET THE MODEL TO TRAIN AND SEND IT TO DEVICE
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
                   f"=== noniid_degree client: {client_noniid_degree} ===\n")
        record_log(self.args, log_path,
                   f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===\n")
        record_log(self.args, log_path, f"=== update_lambda: {self.args.update_lambda} ===\n")
        record_log(self.args, log_path, f"=== regularization: {self.args.regular} ===\n")
        record_log(self.args, log_path, f"=== lr_lambda: {self.args.lr_lambda} ===\n")
        record_log(self.args, log_path, f"=== prunable_layer_idx: {self.conv_layer_idx} ===\n")
        record_log(self.args, log_path,
                   f"=== sorted_layers: {self.sorted_conv_layers} according: {self.num_trainables} ===\n")

    def print_info(self, user_groups=None):
        if user_groups is None:
            user_groups = [[]]
        print(f"data name: {self.args.dataset}")
        print(f"=== model: {self.args.model} ===\n")
        print(f"regularization: {self.args.regular==1}")
        print(f"update_lambda: {self.args.update_lambda==1}")
        print(f"lr_lambda: {self.args.lr_lambda}")
        print(f"user nums: {self.args.num_users}")
        print(f"{'iid' if self.iid else 'noniid'} user sample nums: {len(user_groups[0])}")
        print(f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===")
        print(f"=== using device {self.device} optim {self.args.optim} ===")

    def eucliDist(self, A, B):
        return np.sqrt(sum(np.power((A - B), 2)))

    def load_balance(self, dropout_rates, user_groups, max_device_dprate, p_device):
        P = max_device_dprate
        for i in range(len(dropout_rates)):
            rate = 0.0
            for device in p_device:
                r = 1 - (len(user_groups[device]) * (1 - P) / len(user_groups[i]))
                rate += r
            rate = rate / len(p_device)
            if rate <= 0:
                dropout_rates[i] = 0.05
            else:
                dropout_rates[i] = rate

    def get_layers_dropout_rate(self, feature_result_list, idx_users):
        global_prune_rate_list = [list() for i in range(100)]
        filter_nums = 0
        nearon_nums = 0
        sum_feature_result = 0
        sum_weight_result = 0
        feature_result = feature_result_list
        for loc, idx in enumerate(self.sorted_conv_layers):
            filter_nums += len(feature_result[idx])
            sum_feature_result += torch.exp(-feature_result[idx]).sum()
        for loc, idx in enumerate(self.sorted_fc_layers):
            sum_weight_result += torch.exp(-feature_result[idx]).sum()
            nearon_nums += len(feature_result[idx])
        for user in idx_users:
            for loc, idx in enumerate(self.sorted_conv_layers):
                global_prune_rate_list[user].append(self.dropout_rates[user] * filter_nums * (
                        torch.exp(-feature_result_list[idx]) / sum_feature_result))
            for loc, idx in enumerate(self.sorted_fc_layers):
                global_prune_rate_list[user].append(self.dropout_rates[user] * nearon_nums * (
                        torch.exp(-feature_result_list[idx]) / sum_weight_result))
        return feature_result_list, global_prune_rate_list

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
        return loss

    def client_train(self, idx_users, users_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses, local_grads, local_delta):
        """"
        进行客户端训练
        """

        idxs_users_list = idx_users.tolist()
        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)
        # 计算 mu 值
        for idx in idx_users:
            global_model = copy.deepcopy(users_model[idxs_users_list.index(idx)])
            local_model = LocalUpdate(args=self.args, dataset=train_dataset, idxs=user_groups[idx], device= self.device)
            w, loss, grad = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            grad_lengthened = []
            for key in w.keys():
                if 'weight' not in key and 'bias' not in key:
                    grad_lengthened.append(-1)
                else:
                    grad_lengthened.append(grad.pop(0))
            if loss < train_losses[0] * 3:
                local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                local_losses.append(copy.deepcopy(loss))
                local_grads.append(grad_lengthened)
            print("{}:{:.4f}".format(idx, loss), end="   ")
        print("本轮设备总用时: {:.4f}".format(time.time() - start))
        print()

        return local_losses, local_grads

    def niid_aggregation(self, w, idx):
        """
        Returns the average of the weights with non-iid degree aggregating.
        """
        print("start non-iid aggregation")
        w_avg = copy.deepcopy(w[0][1])
        total = 0

        for i in range(0, len(w)):
            print("client {} | Non-iid degree: {} | lambda {}; b {}".format(idx[i], self.niid_degree[idx[i]],
                                                                            self.lambdas[idx[i]], self.b[idx[i]]))
            total += (w[i][0] / ( (self.lambdas[idx[i]] * self.niid_degree[idx[i]]) + self.b[idx[i]]) )
        for key in w_avg.keys():
            # print("the size of w: ", w_avg[key].size())
            if 'weight' not in key and 'bias' not in key:
                continue
            w_avg[key] *= (w[0][0] / ((self.lambdas[idx[0]] * self.niid_degree[idx[0]]) + self.b[idx[0]]) )
            for i in range(1, len(w)):
                w_avg[key] += (w[i][1][key] * (w[i][0] / ((self.lambdas[idx[i]] * self.niid_degree[idx[i]]) + self.b[idx[i]])))
            w_avg[key] = torch.div(w_avg[key], total)
        return w_avg

    def b_update(self, lr, local_weights, idxs_users, local_grads):
        result = []
        for n, k in enumerate(idxs_users):
            a = 0
            k_index = 0
            for i in range(0, len(local_weights)):
                if idxs_users[i] != k:
                    a += (local_weights[i][0] / (self.lambdas[idxs_users[i]] * self.niid_degree[idxs_users[i]] + self.b[idxs_users[i]]))
                else:
                    k_index = i
            total = pow(a + (local_weights[k_index][0] / (self.lambdas[k] * self.niid_degree[k] + self.b[k])), 2) * pow(self.b[k] + self.lambdas[k]*self.niid_degree[k], 2)

            w = copy.deepcopy(local_weights[k_index][1])
            for key in w.keys():
                if 'weight' not in key and 'bias' not in key:
                    continue
                w[key] *= a * (-local_weights[k_index][0])
                w[key] = torch.div(w[key], total)

            for key in w.keys():
                if 'weight' not in key and 'bias' not in key:
                    continue
                for i in range(0, len(local_weights)):
                    if idxs_users[i] != k:
                        w[key] += (local_weights[i][1][key] * local_weights[k_index][0] * (local_weights[i][0] / (
                                self.lambdas[idxs_users[i]] * self.niid_degree[idxs_users[i]] + self.b[idxs_users[i]])))
                    w[key] = torch.div(w[key], total)

            grad_inner_result = 0
            n_total = 0
            # for each i in C*K ( Note: It should be \sum^K_{i=1}, but the grad w_{t+1} of lambda_k of {i \notin C*K} is zero )
            for i in range(len(local_grads)):
                sum_result = []
                for key, grad in zip(w.keys(), local_grads[i]):
                    if 'weight' not in key and 'bias' not in key:
                        continue
                    if "classifier" in key and "weight" in key or "fc" in key and "weight" in key:
                        inner_res = torch.matmul(w[key], torch.transpose(grad, dim0=0, dim1=1))
                    else:
                        inner_res = torch.matmul(w[key], grad)
                    sum_res = torch.sum(inner_res)
                    sum_result.append(sum_res)
                grad_inner_result += sum(sum_result) * local_weights[i][0]
                n_total += local_weights[i][0]
            r = torch.div(grad_inner_result, n_total)
            # print(r)
            result.append(r)

        normalized_res = []
        # Normalization
        for i in range(len(result)):
            # print("Before Normalization: {:5f}".format(result[i]))
            # torch.div(torch.add(result[i], -min(result)), torch.add(max(result), -min(result)))
            r = (result[i] - min(result)) / (max(result) - min(result))
            normalized_res.append(r)
            # print("After Normalization: {:5f}".format(normalized_res[i]))

        for i, idx in enumerate(idxs_users):
            print("Before: client {} | b {}".format(idx, self.b[idx]))
            # bi_loss_grad = f + self.lambdas[idx]
            bi_loss_grad = normalized_res[i]
            # print(bi_loss_grad)
            self.b[idx] = abs((self.b[idx] - lr * bi_loss_grad))
            print("After: client {} | b {}".format(idx, self.b[idx]))
            print()

    def lambda_update(self, lr, local_weights, idxs_users, local_grads):
        result = []
        for n, k in enumerate(idxs_users):
            a = 0
            k_index = 0
            for i in range(0, len(local_weights)):
                if idxs_users[i] != k:
                    a += (local_weights[i][0] / (
                                self.lambdas[idxs_users[i]] * self.niid_degree[idxs_users[i]] + self.b[idxs_users[i]]))
                else:
                    k_index = i
            total = pow(a + (local_weights[k_index][0] / (self.lambdas[k] * self.niid_degree[k] + self.b[k])), 2) * pow(
                self.b[k] + self.lambdas[k] * self.niid_degree[k], 2)
            w = copy.deepcopy(local_weights[k_index][1])
            for key in w.keys():
                if 'weight' not in key and 'bias' not in key:
                    continue
                w[key] *= a * (-local_weights[k_index][0]) * self.niid_degree[k_index]
                w[key] = torch.div(w[key], total)
            for key in w.keys():
                if 'weight' not in key and 'bias' not in key:
                    continue
                for i in range(0, len(local_weights)):
                    if idxs_users[i] != k:
                        w[key] += (local_weights[i][1][key] * local_weights[k_index][0] * self.niid_degree[k_index] * (local_weights[i][0] / (
                                self.lambdas[idxs_users[i]] * self.niid_degree[idxs_users[i]] + self.b[idxs_users[i]])))
                    w[key] = torch.div(w[key], total)
            grad_inner_result = 0
            n_total = 0
            # for each i in C*K ( Note: It should be \sum^K_{i=1}, but the grad w_{t+1} of lambda_k of {i \notin C*K} is zero )
            for i in range(len(local_grads)):
                sum_result = []
                for key, grad in zip(w.keys(), local_grads[i]):
                    if 'weight' not in key and 'bias' not in key:
                        continue
                    # inner_res = torch.tensordot(w[key], grad, dims=([-1], [-1]))
                    if "classifier" in key and "weight" in key or "fc" in key and "weight" in key:
                        inner_res = torch.matmul(w[key], torch.transpose(grad,dim0=0,dim1=1))
                    else:
                        inner_res = torch.matmul(w[key], grad)
                    # print("inner_res",  inner_res)
                    sum_res = torch.sum(inner_res)
                    sum_result.append(sum_res)
                grad_inner_result += sum(sum_result) * local_weights[i][0]
                n_total += local_weights[i][0]
            r = torch.div(grad_inner_result, n_total)
            # print(r)
            result.append(r)
        normalized_res = []
        # Normalization
        for i in range(len(result)):
            # print("Before Normalization: {:5f}".format(result[i]))
            # torch.div(torch.add(result[i], -min(result)), torch.add(max(result), -min(result)))
            r = (result[i] - min(result)) / (max(result) - min(result))
            normalized_res.append(r)
            # print("After Normalization: {:5f}".format(normalized_res[i]))

        for i, idx in enumerate(idxs_users):
            print("Before: client {} | lambda {}".format(idx, self.lambdas[idx]))
            # bi_loss_grad = f + self.lambdas[idx]
            bi_loss_grad = normalized_res[i]
            # print(bi_loss_grad)
            self.lambdas[idx] = abs((self.lambdas[idx] - lr * bi_loss_grad))
            print("After: client {} | lambda {}".format(idx, self.lambdas[idx]))
            print()

    def train(self):
        result_path = os.path.join(self.result_dir, str(self.args.lr_lambda))
        log_path = os.path.join(self.result_dir, str(self.args.lr_lambda), "log", "log.txt")
        dropout_ratios_path = os.path.join(result_path, "log", f"dropout_ratios.txt")

        global_model = self.load_model()
        print(global_model)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        # 计算niid-degree (JS-divergence)
        print("niid-degree")
        for i in range(self.args.num_users):
            client_distribution = get_target_users_distribution(train_dataset, user_groups, [i])
            client_JS_divergence = get_noniid_degree(client_distribution, self.global_distribution)
            print("client {} : {:4f}".format(i, client_JS_divergence), end="  |  ")
            if (i+1) % 5 == 0:
                print()
            self.niid_degree.append(client_JS_divergence)
        print()

        self.print_info(user_groups)
        self.record_base_message(log_path)

        # Training
        global_losses = []
        train_losses = []
        test_accuracys = []

        all_train_data = np.array([])
        for k, v in user_groups.items():
            all_train_data = np.concatenate(
                (all_train_data, v), axis=0)

        record_rank = []
        dropout_step = 1
        max_device = 0
        min_device = 0
        max_sample_nums = len(user_groups[0])
        min_sample_nums = len(user_groups[0])
        feature_result_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
        update_ratio_rank_list = None

        self.reset_seed()

        # 第一次评估
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
        test_accuracys.append(test_acc)
        train_losses.append(test_loss)
        print("-train loss:{:.4f} -val acc:{:.4f}".format(test_loss, test_acc))

        for i in range(self.args.num_users):
            if max_sample_nums < len(user_groups[i]):
                max_sample_nums = len(user_groups[i])
                max_device = i
        for i in range(self.args.num_users):
            if min_sample_nums < len(user_groups[i]):
                min_sample_nums = len(user_groups[i])
                min_device = i
        rand_device = random.randint(0, 100)
        rand_sample_nums = len(user_groups[rand_device])
        total_sample_nums = max_sample_nums + min_sample_nums + rand_sample_nums
        p_devices = [max_device, min_device, rand_device]

        for epoch in range(self.args.epochs):
            local_weights, local_losses, local_grads, local_delta = [], [], [], []
            self.fc_weight = []
            self.conv_rank = []
            if self.args.model == 'lenet':
                filter_rank_list = [[], []]
                filter_rank = [[], []]
                nearon_weight_list = [[], [], []]
                nearon_weight = [[], [], []]
            elif self.args.model == 'cnn':
                filter_rank_list = [[], [], []]
                filter_rank = [[], [], []]
                nearon_weight_list = [[], []]
                nearon_weight = [[], []]
            elif self.args.model == 'vgg':
                filter_rank_list = [list() for i in range(8)]
                filter_rank = [list() for i in range(8)]
                nearon_weight_list = [list() for i in range(3)]
                nearon_weight = [list() for i in range(3)]
            elif self.args.model == 'resnet':
                filter_rank_list = [list() for i in range(20)]
                filter_rank = [list() for i in range(20)]
                nearon_weight_list = [list() for i in range(1)]
                nearon_weight = [list() for i in range(1)]
            start = time.time()
            users_model = [None] * 10
            print(f"\n | Global Training Round : {epoch} |\n")

            # 选择设备，进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)

            # adaptive dropout
            if (epoch > 0 and epoch % dropout_step == 0) and self.start_dropout is False:
                print("finding the minimum rank diff")
                mean_feature_result_list = feature_result_list
                for device in p_devices:
                    for loc, idx in enumerate(self.sorted_conv_layers):
                        feature_result = get_all_feature(global_model, train_dataset, user_groups[device], idx)
                        feature_result_list[idx] = feature_result[idx]
                        filter_rank_list[loc].append(feature_result[idx])
                        filter_rank[loc].append(feature_result[idx].cpu().tolist())
                    for loc, idx in enumerate(self.sorted_fc_layers):
                        layer_weight = get_weight(global_model, idx)
                        feature_result_list[idx] = layer_weight
                        nearon_weight_list[loc].append(layer_weight)
                        nearon_weight[loc].append(layer_weight.cpu().tolist())
                    for i in range(len(feature_result_list)):
                        mean_feature_result_list[i] = torch.add(mean_feature_result_list[i],
                                                                torch.mul(feature_result_list[i],
                                                                          len(user_groups[device])))
                for i in range(len(feature_result_list)):
                    mean_feature_result_list[i] = torch.div(mean_feature_result_list[i],
                                                            len(p_devices) * total_sample_nums)
                for i in range(len(self.sorted_conv_layers)):
                    self.conv_rank.append(torch.stack(filter_rank_list[i]).mean(dim=0).tolist())
                for i in range(len(self.sorted_fc_layers)):
                    self.fc_weight.append(torch.stack(nearon_weight_list[i]).mean(dim=0).tolist())

                self.rank_weight.append([epoch, filter_rank, nearon_weight])
                self.avg_rank_weight.append([epoch, self.conv_rank, self.fc_weight])

                if len(record_rank) > 0:
                    tmp = []
                    for i in range(len(self.avg_rank_weight[-1][1])):
                        conv_right = np.array(self.avg_rank_weight[-1][1][i])
                        conv_left = np.array(record_rank[-1][1][i])
                        tmp.append(self.eucliDist(conv_left, conv_right))
                    tmp1 = np.array(tmp)
                    cur_diff = round(math.sqrt(np.sum(tmp1 ** 2)), 3)
                    if len(record_rank) == 1:
                        self.pre_diff = cur_diff
                        self.pre_model = copy.deepcopy(global_model)
                    if cur_diff < self.pre_diff and self.find_first_min_diff is False:
                        self.pre_diff = cur_diff
                        self.pre_model = copy.deepcopy(global_model)
                    elif cur_diff >= self.pre_diff and epoch >= self.args.start_epoch:
                        global_model = copy.deepcopy(self.pre_model)
                        mean_ratio = []
                        for device in p_devices:
                            device_ret = get_ratio(global_model, train_dataset, user_groups[device], self.device)
                            mean_ratio.append(device_ret * len(user_groups[device]))
                        device_ratio = sum(mean_ratio) / (len(mean_ratio) * total_sample_nums)
                        update_ratio_rank_list = mean_feature_result_list
                        self.load_balance(self.dropout_rates, user_groups, device_ratio, p_devices)
                        self.start_dropout = True
                        self.pre_diff = cur_diff
                        self.get_dropout_tatios.append([epoch, self.pre_diff, self.dropout_rates])

                    record_rank.append([epoch, self.conv_rank])
                else:
                    record_rank.append([epoch, self.conv_rank])
                    self.pre_diff = 0

            for i in range(len(idxs_users)):
                users_model[i] = copy.deepcopy(global_model)

            if self.start_dropout:
                print("start dropout")
                feature_result_list, global_prune_rate_list = self.get_layers_dropout_rate(update_ratio_rank_list, idxs_users)
                for i, user_idx in enumerate(idxs_users):
                    prune_rate = self.dropout_rates[user_idx]
                    for idx in self.sorted_conv_layers:
                        hrank_prune(users_model[i], global_prune_rate_list[user_idx], prune_rate, layer_idx=idx, device=self.device)
                    for idx in self.sorted_fc_layers:
                        hrank_prune(users_model[i], global_prune_rate_list[user_idx], prune_rate, layer_idx=idx, device=self.device)
                    users_model[i].train()
                    self.num_weights.append(users_model[i].calc_num_all_active_params())
                    self.channels_list.append(users_model[i].get_channels())
                    self.nearons_list.append(users_model[i].get_nearons())
                    self.log_weight.append([self.num_weights[-1], self.channels_list[-1], self.nearons_list[-1]])
                self.log_weights.append([epoch, self.log_weight[-10:]])


            local_losses, local_grads = self.client_train(idxs_users, users_model, user_groups, epoch, train_dataset, train_losses, local_weights, local_losses, local_grads, local_delta)

            # 无效轮
            if len(local_weights) == 0:
                train_losses.append(train_losses[-1])
                test_accuracys.append(test_accuracys[-1])
                continue

            if self.args.gradient_norm:
                print("gradient normalization")
                num_steps = []
                for idx in idxs_users:
                    if len(user_groups[idx]) % self.args.local_bs==0:
                        num_step = (len(user_groups[idx]) * self.args.local_ep) / self.args.local_bs
                    else:
                        num_step = ((len(user_groups[idx]) / self.args.local_bs) + 1) * self.args.local_ep
                    num_steps.append(num_step)
                gradient_norm(global_model, local_weights, num_steps)

            # non-iid aggregation
            global_weights = self.niid_aggregation(local_weights, idxs_users)
            # update control parameters
            self.lr_lambda = self.lr_lambda * pow(self.lambda_decay, epoch)
            self.lr_b = self.lr_b * pow(self.b_decay, epoch)

            self.lambda_update(lr=self.lr_lambda, local_weights=local_weights, idxs_users=idxs_users,
                                             local_grads=local_grads)
            self.b_update(lr=self.lr_b, local_weights=local_weights, idxs_users=idxs_users,
                          local_grads=local_grads)


            # update global weights
            pre_model = copy.deepcopy(global_model.state_dict())
            global_model.load_state_dict(global_weights)

            # Test inference after completion of training
            test_acc, test_loss = test_inference(global_model, test_dataset, self.device)

            if test_loss < train_losses[0] * 100:
                test_accuracys.append(test_acc)
                train_losses.append(test_loss)
            else:
                print("recover model test_loss/test_acc : {}/{}".format(test_loss, test_acc))
                train_losses.append(train_losses[-1])
                test_accuracys.append(test_accuracys[-1])
                global_model.load_state_dict(pre_model)

            if (epoch + 11) % 10 == 0 or epoch == self.args.epochs - 1:
                save_result(self.args, os.path.join(result_path, str(self.args.lr_b), "_train_loss.txt"), str(train_losses)[1:-1])
                save_result(self.args, os.path.join(result_path, str(self.args.lr_b), "_test_accuracy.txt"), str(test_accuracys)[1:-1])

            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}".format(epoch, test_loss, test_acc, self.args.lr, time.time() - start))
            print()
        record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        weight_path = os.path.join(result_path, str(self.args.lr_b), "log", f"weights_{self.args.seed}.txt")
        rank_weight_path = os.path.join(result_path, str(self.args.lr_b), "log", f"rank_weight_{self.args.seed}.txt")
        avg_rank_weight_path = os.path.join(result_path, str(self.args.lr_b), "log", f"avg_rank_weight_{self.args.seed}.txt")
        record_log(self.args, weight_path, str(self.log_weights)[1:-1])
        record_log(self.args, rank_weight_path, str(self.rank_weight)[1: -1])
        record_log(self.args, avg_rank_weight_path, str(self.avg_rank_weight)[1: -1])
        record_log(self.args, dropout_ratios_path, str(self.get_dropout_tatios)[1: -1])

if __name__ == '__main__':
    args = args_parser()
    t = FedDHAD(args, iid=False, equal=False, unequal=False, unbalance=False, dirichlet=True)
    t.train()
