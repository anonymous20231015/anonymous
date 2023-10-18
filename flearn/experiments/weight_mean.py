import math
import random
import sys
import numpy
# sys.path.append('/home/zhouchendi/lishuo/new/FedGMS_pth')

import numpy as np
import torch
from utils.get_flops import dropout_get_flops
from flearn.utils.model_utils import average_weights, test_inference, is_conv, ratio_minus
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.utils.prune import hrank_prune, get_ratio, get_rate_for_each_layers, lt_prune,get_all_feature,get_weight
import copy
import os
import time
from flearn.experiments.central import CentralTraining
device_speed = 500 / 8  # MB/s
server_speed = 1000 / 8  # MB/s
cmp_speed = 500000000

class HRank(CentralTraining):
    """
    按照特征图的秩，一次减去一定百分比的卷积层
    """

    def __init__(self, args, equal=False, unequal=False, unbalance=False, dirichlet=False, l=2,
                 result_dir="hrank",
                 ):

        super(HRank, self).__init__(args, equal, unequal, unbalance,  dirichlet, l,
                                    result_dir)
        self.conv_rank = []
        self.fc_weight = []
        self.avg_rank_weight = []
        self.rank_weight = []
        self.pre_model = None
        self.pre_diff = None
        self.start_dropout = False
        self.get_dropout_tatios = []
        self.result_dir = result_dir

    def eucliDist(self, A, B):
        return np.sqrt(sum(np.power((A - B), 2)))

    def load_balance(self, dropout_rates, user_groups, mean_device_dprate, p_device):
        P = mean_device_dprate
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

    def train(self):
        # 记录日志和结果
        log_path = os.path.join(self.result_dir, "log.txt")
        result_path = os.path.join(self.result_dir, "FedAD")
        dropout_ratios_path = os.path.join(result_path, "log", f"dropout_ratios.txt")

        # 加载模型
        global_model = self.load_model()
        print(global_model)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        self.print_info(user_groups)
        self.record_base_message(log_path)
        # Training
        train_losses = []
        test_accs = []
        record_rank = []
        dropout_step = 10
        max_device = 0
        min_device = 0
        max_sample_nums = len(user_groups[0])
        min_sample_nums = len(user_groups[0])
        feature_result_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
        update_ratio_rank_list = None
        self.reset_seed()

        # 第一次评估
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
        test_accs.append(test_acc)
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
        rand_device = random.randint(0,100)
        rand_sample_nums = len(user_groups[rand_device])
        total_sample_nums = max_sample_nums + min_sample_nums + rand_sample_nums
        p_devices = [max_device, min_device, rand_device]
        for epoch in range(self.args.epochs):
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
            local_weights, local_losses = [], []
            local_P, local_v = [], []
            users_model = [None]*10

            print(f'\n | Global Training Round : {epoch} |\n')
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)

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
                        mean_feature_result_list[i] = torch.add(mean_feature_result_list[i], torch.mul(feature_result_list[i], len(user_groups[device])))
                for i in range(len(feature_result_list)):
                    mean_feature_result_list[i] = torch.div(mean_feature_result_list[i], len(p_devices) * total_sample_nums)
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
                    if cur_diff < self.pre_diff:
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

            self.client_train(idxs_users, users_model, user_groups, epoch,
                                                      train_dataset, train_losses, local_weights, local_losses)

            # 无效轮
            if len(local_weights) == 0 and len(local_P) == 0:
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                continue

            # update global weights

            global_weights = average_weights(local_weights)
            pre_model = copy.deepcopy(global_model.state_dict())
            global_model.load_state_dict(global_weights)

            # Test inference after completion of training
            test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
            if test_loss < train_losses[0] * 3:
                test_accs.append(test_acc)
                train_losses.append(test_loss)
            else:
                print("recover model test_loss/test_acc : {}/{}".format(test_loss, test_acc))
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                global_model.load_state_dict(pre_model)

            if (epoch + 11) % 10 == 0 or epoch == self.args.epochs - 1:
                save_result(self.args, os.path.join(result_path, str(self.args.seed) + "_train_loss.txt"),
                            str(train_losses)[1:-1])
                save_result(self.args, os.path.join(result_path, str(self.args.seed) + "_test_accuracy.txt"),
                            str(test_accs)[1:-1])

            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}".
                  format(epoch, test_loss, test_acc, self.args.lr, time.time() - start))
        record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        weight_path = os.path.join(result_path, "log", f"weights_{self.args.seed}.txt")
        rank_weight_path = os.path.join(result_path, "log", f"rank_weight_{self.args.seed}.txt")
        avg_rank_weight_path = os.path.join(result_path, "log", f"avg_rank_weight_{self.args.seed}.txt")
        record_log(self.args, weight_path, str(self.log_weights)[1:-1])
        record_log(self.args, rank_weight_path, str(self.rank_weight)[1: -1])
        record_log(self.args, avg_rank_weight_path, str(self.avg_rank_weight)[1: -1])
        record_log(self.args, dropout_ratios_path, str(self.get_dropout_tatios)[1: -1])


if __name__ == "__main__":
    args = args_parser()
    t = HRank(args, equal=False, unequal=False, unbalance=False, dirichlet=True, l=2)
    t.train()
