import copy
import math
import os
import random
import time
import torch
from torch import nn
import sys
sys.path.append('/home/zhouchendi/mbc/FedHAVG/')
# sys.path.append("/Users/mabeichen/PycharmProjects/FedHAVG")
import numpy as np

from data.util import get_global_distribution, get_client_noniid_degree, get_target_users_distribution, \
    get_noniid_degree
from flearn.utils.model_utils import test_inference, average_weights, gradient_norm
from flearn.utils.options import args_parser
from flearn.utils.update import LocalUpdate
from flearn.utils.util import record_log, save_result
from flearn.models import cnn, vgg, resnet, lenet,alexnet
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Base(object):
    """"
    Estimate Non-IID degree Based on Gradients
    """
    def __init__(self, args, iid, equal=True, unequal=False, unbalance=False, dirichlet=False, l=2, result_dir="fedavg", server_mu=0.0, client_mu=0.0):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

        # 设置随机种子
        self.reset_seed()

        # 数据集划分
        self.num_data = 50000
        self.l = l  # noniid的程度， l越小 noniid 程度越大， 当 l = 1 时， 就是将数据按照顺序分成 clients 份，每个设备得到一份， 基本上只包含一个数字

        # 定义FedAVG参数
        self.m = 10 #10
        self.iid = iid
        self.equal = equal
        self.unequal = unequal
        self.unbalance = unbalance
        self.dirichlet = dirichlet
        self.decay = self.args.decay
        self.result_dir = result_dir
        self.client_mu = client_mu

        self.lr_decay = 0.99
        self.init_lr = self.args.lr


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
                                                    num_users=self.args.num_users, equal=self.equal, unequal=self.unequal, unbalance=self.unbalance, dirichlet=self.dirichlet, l=self.l)
        elif self.args.dataset == "mnist":
            from data.mnist.mnist_data import get_dataset
            self.num_data = 50000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            l=self.l, unequal=self.unequal)
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
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, equal=self.equal, unequal=self.unequal, dirichlet=self.dirichlet,
                            l=self.l)
        elif self.args.dataset == "tinyimagenet":
            from data.tinyImagenet.tinyimagenet_data import get_dataset
            self.num_data = 100000
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users,
                            dirichlet=self.dirichlet)
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
            global_model = resnet.resnet50(num_classes=num_classes)
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
        client_noniid_degree = get_client_noniid_degree(self.train_dataset, self.user_groups)
        record_log(self.args, log_path,
                   f"=== noniid_degree client: {client_noniid_degree} ===\n")
        record_log(self.args, log_path,
                   f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===\n")


    def print_info(self, user_groups=None):
        if user_groups is None:
            user_groups = [[]]
        print(f"data name: {self.args.dataset}")
        print(f"=== model: {self.args.model} ===\n")
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

    def client_train(self, idx_users, global_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses, local_grads):
        """"
        进行客户端训练
        """

        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)

        for idx in idx_users:
            local_model = LocalUpdate(args=self.args, dataset=train_dataset, idxs=user_groups[idx], device= self.device)
            w, loss, grad = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, mu=self.args.client_mu)
            if loss < train_losses[0] * 3:
                local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                local_losses.append(copy.deepcopy(loss))
                local_grads.append(grad)
            print("{}:{:.4f}".format(idx, loss), end="   ")
        print("本轮设备总用时: {:.4f}".format(time.time() - start))
        print()

    def train(self):
        log_path = os.path.join(self.result_dir, "log.txt")
        result_path = os.path.join(self.result_dir)
        global_model = self.load_model()
        print(global_model)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

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

        # 第一次评估
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
        test_accuracys.append(test_acc)
        train_losses.append(test_loss)
        print("-train loss:{:.4f} -val acc:{:.4f}".format(test_loss, test_acc))

        for epoch in range(self.args.epochs):
            start = time.time()
            local_weights, local_losses, local_grads = [], [], []

            print(f"\n | Global Training Round : {epoch} |\n")

            # 选择设备，进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)

            self.client_train(idxs_users, global_model, user_groups, epoch, train_dataset,train_losses, local_weights, local_losses, local_grads)

            # 无效轮
            if len(local_weights) == 0:
                train_losses.append(train_losses[-1])
                test_accuracys.append(test_accuracys[-1])
                continue

            if self.args.gradient_norm:
                print("gradient normalization")
                num_steps = []
                for idx in idxs_users:
                    # print(len(user_groups[idx]))
                    if len(user_groups[idx]) % self.args.local_bs==0:
                        num_step = (len(user_groups[idx]) * self.args.local_ep) / self.args.local_bs
                    else:
                        num_step = (math.floor(len(user_groups[idx]) / self.args.local_bs) + 1) * self.args.local_ep
                    # print(num_step)
                    num_steps.append(num_step)
                gradient_norm(global_model, local_weights, num_steps)

            # aggregation
            global_weights = average_weights(local_weights)

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
                save_result(self.args, os.path.join(result_path,  "_train_loss.txt"), str(train_losses)[1:-1])
                save_result(self.args, os.path.join(result_path,  "_test_accuracy.txt"), str(test_accuracys)[1:-1])


            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}".format(epoch, test_loss, test_acc, self.args.lr, time.time() - start))
            print()

if __name__ == '__main__':
    args = args_parser()
    t = Base(args, iid=False, equal=False, unequal=False, unbalance=0, dirichlet=True)
    t.train()
