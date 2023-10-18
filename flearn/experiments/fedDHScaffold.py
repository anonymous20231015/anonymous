import copy
import math
import os
import random
import time
import torch
from torch import nn
import sys
# sys.path.append('/home/zhouchendi/mbc/FedHAVG/')
# sys.path.append("/Users/mabeichen/PycharmProjects/FedHAVG")
import numpy as np

from data.util import get_global_distribution, get_client_noniid_degree, get_target_users_distribution, \
    get_noniid_degree
from flearn.utils.model_utils import test_inference, gradient_norm
from flearn.utils.options import args_parser
from flearn.utils.update import LocalUpdate
from flearn.utils.util import record_log, save_result
from flearn.models import cnn, vgg, resnet, lenet, alexnet
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class FedDHScaffold(object):
    """"
    Estimate Non-IID degree Based on Gradients
    """
    def __init__(self, args, iid, equal=True, unequal=False, unbalance=False, dirichlet=False, l=2, result_dir="fedhavg"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

        # 设置随机种子
        self.reset_seed()

        # 数据集划分
        self.num_data = 50000
        self.l = l  # noniid的程度， l越小 noniid 程度越大， 当 l = 1 时， 就是将数据按照顺序分成 clients 份，每个设备得到一份， 基本上只包含一个数字

        # 定义FedHAVG参数
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
        elif self.args.dataset == "fashionmnist":
            from data.fashionmnist.fashionmnist_data import get_dataset
            self.num_data = 50000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            l=self.l, unequal=self.unequal)
        elif self.args.dataset == "cifar100":
            from data.cifar100.cifar100_data import get_dataset
            self.num_data = 50000
            self.l = self.l * 10
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, equal=self.equal, unequal=self.unequal, dirichlet=self.dirichlet,
                            l=self.l)
        else:
            exit('Error: unrecognized dataset')
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_groups = user_groups
        self.global_distribution = get_global_distribution(train_dataset, user_groups)
        return train_dataset, test_dataset, user_groups

    def load_model(self):
        # BUILD MODEL
        if self.args.dataset == "cifar10":
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
        # SET THE MODEL TO TRAIN AND SEND IT TO DEVICE
        global_model.to(self.device)
        global_model.train()
        return global_model

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

    def init_c(self, model):
        self.server_c = {}
        self.clients_c = [{} for i in range(self.args.num_users)]

        for k, v in model.named_parameters():
            self.server_c[k] = torch.zeros_like(v.data)

        for i in range(self.args.num_users):
            for k, v in model.named_parameters():
                self.clients_c[i][k] = torch.zeros_like(v.data)

    def client_train(self, idx_users, global_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses, local_grads, local_delta):
        """"
        进行客户端训练
        """

        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)
        global_par = copy.deepcopy(global_model.state_dict())
        # 计算 mu 值
        for idx in idx_users:
            local_model = LocalUpdate(args=self.args, dataset=train_dataset, idxs=user_groups[idx], device= self.device)

            w, loss, grad, cnt = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, c=self.server_c, ci=self.clients_c[idx])
            c_i = copy.deepcopy(self.clients_c[idx])
            delta_i = copy.deepcopy(c_i)
            for key in w.keys():
                self.clients_c[idx][key] = c_i[key] - self.server_c[key] + (
                            (global_par[key] - w[key]) / (cnt * self.args.lr))
                delta_i[key] = self.clients_c[idx][key] - c_i[key]
            if loss < train_losses[0] * 3:
                local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                local_losses.append(copy.deepcopy(loss))
                local_grads.append(grad)
                local_delta.append(delta_i)
            print("{}:{:.4f}".format(idx, loss), end="   ")
        print("本轮设备总用时: {:.4f}".format(time.time() - start))
        print()

        return local_losses, local_grads, local_delta

    def niid_aggregation(self, w, idx):
        """
        Returns the average of the weights with non-iid degree aggregating.
        """
        print("start non-iid aggregation")
        w_avg = copy.deepcopy(w[0][1])
        total = 0

        for i in range(0, len(w)):
            print("client {} | Non-iid degree: {} | lambda {}".format(idx[i], self.niid_degree[idx[i]], self.lambdas[idx[i]]))
            total += (w[i][0] / ( (self.lambdas[idx[i]] * self.niid_degree[idx[i]]) + self.b[idx[i]]) )
        for key in w_avg.keys():
            # print("the size of w: ", w_avg[key].size())
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
                w[key] *= a * (-local_weights[k_index][0])
                w[key] = torch.div(w[key], total)

            for key in w.keys():
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
                    if "classifier" in key and "weight" in key:
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
                w[key] *= a * (-local_weights[k_index][0]) * self.niid_degree[k_index]
                w[key] = torch.div(w[key], total)

            for key in w.keys():
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
                    # inner_res = torch.tensordot(w[key], grad, dims=([-1], [-1]))
                    if "classifier" in key and "weight" in key:
                        inner_res = torch.matmul(w[key], torch.transpose(grad,dim0=0,dim1=1))
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

        self.reset_seed()

        # init server and client control
        self.init_c(global_model)

        # 第一次评估
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
        test_accuracys.append(test_acc)
        train_losses.append(test_loss)
        print("-train loss:{:.4f} -val acc:{:.4f}".format(test_loss, test_acc))

        for epoch in range(self.args.epochs):
            start = time.time()
            local_weights, local_losses, local_grads, local_delta = [], [], [], []

            print(f"\n | Global Training Round : {epoch} |\n")

            total_delta = copy.deepcopy(global_model.state_dict())
            for key in total_delta:
                total_delta[key] = 0.0

            # 选择设备，进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)
            local_losses, local_grads, local_delta = self.client_train(idxs_users, global_model, user_groups, epoch, train_dataset, train_losses, local_weights, local_losses, local_grads, local_delta)

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

            # update server control
            for key in total_delta:
                for i in range(len(local_weights)):
                    total_delta[key] += local_delta[i][key] / len(idxs_users)

            for key in self.server_c:
                self.server_c[key] += total_delta[key] * len(idxs_users) / self.args.num_users

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
if __name__ == '__main__':
    args = args_parser()
    t = FedDHScaffold(args, iid=False, equal=False, unequal=False, unbalance=False, dirichlet=True)
    t.train()
