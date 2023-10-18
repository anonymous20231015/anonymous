import sys
sys.path.append('/home/zhouchendi/lishuo/Drop/FedGMS_pth')
import numpy as np
import torch
from utils.get_flops import dropout_get_flops
from flearn.utils.model_utils import average_weights, test_inference, is_conv, ratio_minus
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.utils.prune import hrank_prune, get_ratio, get_rate_for_each_layers, lt_prune,get_all_feature,get_weight
from data.util import get_global_distribution, get_target_users_distribution, \
    get_noniid_degree
import copy
import os
import time
from flearn.experiments.central import CentralTraining
device_speed = 50 / 8  # MB/s
server_speed = 100 / 8  # MB/s
cmp_speed = 500000000

class DropOut(CentralTraining):
    """
    按照特征图的秩，一次减去一定百分比的卷积层
    """

    def __init__(self, args, equal=False, unequal=False, unbalance=False, dirichlet=False, l=2,
                prune_rate=0.6, result_dir="hrank",
                 ):

        super(DropOut, self).__init__(args, equal, unequal, unbalance,  dirichlet, l, prune_rate,
                                    result_dir)

        self.result_dir = result_dir

    def train(self):
        # 记录日志和结果
        log_path = os.path.join(self.result_dir, "log.txt")
        result_path = os.path.join(self.result_dir, "FedDrop")

        # 加载模型
        global_model = self.load_model()

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        self.print_info(user_groups)
        self.record_base_message(log_path)
        # Training
        train_losses = []
        test_accs = []

        all_train_data = np.array([])
        for k, v in user_groups.items():
            all_train_data = np.concatenate(
                (all_train_data, v), axis=0)

        degrees = []
        for i in range(self.args.num_users):
            distribution = get_target_users_distribution(train_dataset, user_groups, [i])
            degree = get_noniid_degree(distribution, self.global_distribution)
            degrees.append(degree)
        print(degrees)
        degrees = [1 / (i + 0.0001) for i in degrees]
        for i in range(self.args.num_users):
            degrees[i] = degrees[i] * len(self.user_groups[i])
        rates = [i / sum(degrees) for i in degrees]
        print(rates)
        print(sum(rates))

        self.reset_seed()

        # 第一次评估
        # loss = self.get_loss(all_train_data, train_dataset, global_model)
        # Test inference after completion of training
        test_acc, test_loss = test_inference(global_model, test_dataset, self.device)
        test_accs.append(test_acc)
        train_losses.append(test_loss)
        print("-train loss:{:.4f} -val acc:{:.4f}".format(test_loss, test_acc))

        for epoch in range(self.args.epochs):
            start = time.time()
            local_weights, local_losses = [], []
            local_P, local_v = [], []
            users_model = [None]*10
            conv_cmp_nums, conv_par_nums, fc_cmp_nums, fc_par_nums = [], [], [], []
            dropout_rates = []
            time_totals = []
            print(f'\n | Global Training Round : {epoch} |\n')
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)
            for i in range(len(idxs_users)):
                users_model[i] = copy.deepcopy(global_model)
            # 进行剪枝操作
            if epoch >= 0:
                sum_feature_result = torch.tensor(0.)
                feature_result_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
                prune_rate_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
                for i in range(len(idxs_users)):
                    conv_cmp, conv_par = dropout_get_flops(model=self.args.model, dataset=self.args.dataset,
                                                           input_num=len(user_groups[idxs_users[i]]),
                                                           layer_cls='conv')
                    fc_cmp, fc_par = dropout_get_flops(model=self.args.model, dataset=self.args.dataset,
                                                       input_num=len(user_groups[idxs_users[i]]), layer_cls='fc')
                    m_k = conv_par + (1 - self.prune_rate) ** 2 * fc_par
                    c_k = conv_cmp + (1 - self.prune_rate) ** 2 * fc_cmp
                    time_comm = m_k * 4 / 1024 / 1024 / device_speed + m_k * 4 / 1024 / 1024 / server_speed
                    time_cmp = c_k / cmp_speed
                    time_total = time_comm + time_cmp
                    conv_cmp_nums.append(conv_cmp)
                    conv_par_nums.append(conv_par)
                    fc_cmp_nums.append(fc_cmp)
                    fc_par_nums.append(fc_par)
                    time_totals.append(time_total)

                for i in range(len(idxs_users)):
                    T = max(time_totals)
                    conv_par = conv_par_nums[i]
                    conv_cmp = conv_cmp_nums[i]
                    fc_par = fc_par_nums[i]
                    fc_cmp = fc_cmp_nums[i]
                    t_k_conv = conv_par * 4 / 1024 / 1024 * (1 / device_speed + 1 / server_speed) + conv_cmp / cmp_speed
                    t_k_full = fc_par * 4 / 1024 / 1024 * (1 / device_speed + 1 / server_speed) + fc_cmp / cmp_speed
                    dropout_rate = 1 - (abs(T - t_k_conv) / t_k_full) ** (1/2)
                    if dropout_rate<0:
                        dropout_rate=0.2
                    dropout_rates.append(dropout_rate)
                for i in range(len(idxs_users)):
                    for idx in self.sorted_fc_layers:
                        print(dropout_rates)
                        prune_rate = dropout_rates[i]
                        layer = users_model[i].prunable_layers[idx]
                        nearon_nums = len(layer.weight)
                        layer_weight = torch.ones(nearon_nums)
                        prune_rate_list[idx] = prune_rate * layer_weight
                        hrank_prune(users_model[i], prune_rate_list, prune_rate, layer_idx=idx, device=self.device)
                    self.num_weights.append(users_model[i].calc_num_all_active_params())
                    self.channels_list.append(users_model[i].get_channels())
                    self.nearons_list.append(users_model[i].get_nearons())
                    self.log_weight.append([self.num_weights[-1], self.channels_list[-1], self.nearons_list[-1]])
                self.log_weights.append([epoch, self.log_weight[-10:]])
                # print("=======剪枝后通道结果=======", self.log_weights[-1])
            # 选择设备，并进行训练
            global_model.train()
            self.client_train(idxs_users, users_model, user_groups, epoch,
                                                      train_dataset, train_losses, local_weights, local_losses)
            if len(local_weights) == 0 and len(local_P) == 0:
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                continue

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            pre_model = copy.deepcopy(global_model.state_dict())
            global_model.load_state_dict(global_weights)
            # loss = self.get_loss(all_train_data, train_dataset, global_model)
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
            print()

        print("prune result:" + str(self.num_weights) + " " + str(self.channels_list))
        record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        weight_path = os.path.join(result_path, "log", f"weights_{self.args.seed}.txt")
        record_log(self.args, weight_path, str(self.log_weights)[1:-1])


if __name__ == "__main__":
    args = args_parser()
    t = DropOut(args, equal=False, unequal=False, unbalance=False, dirichlet=True, l=2, prune_rate=0.5)
    t.train()

