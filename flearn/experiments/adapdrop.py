import sys
sys.path.append('/home/zhouchendi/lishuo/AFD/FedGMS_pth')
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

class AdapDrop(CentralTraining):
    """
    按照特征图的秩，一次减去一定百分比的卷积层
    """

    def __init__(self, args, equal=False, unequal=False, unbalance=False, dirichlet=False, l=2,
                  prune_rate=0.6,  result_dir="adapdrop",
                 ):

        super(AdapDrop, self).__init__(args, equal, unequal, unbalance,  dirichlet, l, prune_rate,
                                    result_dir)
        self.result_dir = result_dir
    def train(self):
        # 记录日志和结果
        log_path = os.path.join(self.result_dir, "log.txt")
        result_path = os.path.join(self.result_dir, "AdapDrop")
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

        all_train_data = np.array([])
        for k, v in user_groups.items():
            all_train_data = np.concatenate(
                (all_train_data, v), axis=0)

        degrees = []
        for i in range(self.args.num_users):
            distribution = get_target_users_distribution(train_dataset, user_groups, [i])
            degree = get_noniid_degree(distribution, self.global_distribution)
            degrees.append(degree)
        degrees = [1 / (i + 0.0001) for i in degrees]
        for i in range(self.args.num_users):
            degrees[i] = degrees[i] * len(self.user_groups[i])
        rates = [i / sum(degrees) for i in degrees]

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

            print(f'\n | Global Training Round : {epoch} |\n')
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)
            print("epoch:", epoch, idxs_users)
            for i in range(len(idxs_users)):
                users_model[i] = copy.deepcopy(global_model)
                if epoch > 0:
                    if self.Recorded:
                        print("Recorded:", idxs_users[i])
                        for layer in range(len(self.sorted_conv_layers)):
                            prune_layer = users_model[i].prunable_layers[self.sorted_conv_layers[layer]]
                            rank_mask = self.Ac[0][layer]
                            prune_layer.rank_mask = rank_mask
                            prune_layer.dropout_rate = self.prune_rate
                            prune_layer.uprune_out_channels(rank_mask, self.device)
                        for layer in range(len(self.sorted_fc_layers)):
                            prune_layer = users_model[i].prunable_layers[self.sorted_fc_layers[layer]]
                            rank_mask = self.Ac[1][layer]
                            prune_layer.rank_mask = rank_mask
                            prune_layer.prune_rate = self.prune_rate
                            prune_layer.uprune_out_channels(rank_mask, self.device)
                    else:
                        if len(self.scoremap[idxs_users[i]][0]) == 0:
                            for idx in self.sorted_conv_layers:
                                feature_result_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
                                prune_rate = self.prune_rate
                                feature_result = get_all_feature(users_model[i], train_dataset,
                                                                 user_groups[idxs_users[i]], idx)
                                feature_result_list[idx] = torch.ones_like(feature_result[idx])
                                hrank_prune(users_model[i], feature_result_list, prune_rate, layer_idx=idx, device=self.device, baselinename="AdapDrop")
                            for idx in self.sorted_fc_layers:
                                prune_rate_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
                                prune_rate = self.prune_rate
                                layer_weight = get_weight(users_model[i], idx)
                                prune_rate_list[idx] = torch.ones_like(layer_weight)
                                hrank_prune(users_model[i], prune_rate_list, prune_rate, layer_idx=idx, device=self.device, baselinename="AdapDrop")
                        else:
                            for layer in range(len(self.sorted_conv_layers)):
                                prune_layer = users_model[i].prunable_layers[self.sorted_conv_layers[layer]]
                                tmp = self.scoremap[idxs_users[i]][0][layer].squeeze()
                                sorted, ind = torch.sort(tmp)
                                prune_ind = int(sorted.shape[0] * self.prune_rate) + 1
                                rank_mask = self.scoremap[idxs_users[i]][0][layer] >= sorted[prune_ind]
                                prune_layer.rank_mask = rank_mask
                                prune_layer.dropout_rate = self.prune_rate
                                prune_layer.uprune_out_channels(rank_mask, self.device)
                            for layer in range(len(self.sorted_fc_layers)):
                                prune_layer = users_model[i].prunable_layers[self.sorted_fc_layers[layer]]
                                tmp = self.scoremap[idxs_users[i]][1][layer].squeeze(1)
                                sorted, ind = torch.sort(tmp)
                                prune_ind = int(sorted.shape[0] * self.prune_rate) + 1
                                rank_mask = self.scoremap[idxs_users[i]][1][layer] >= sorted[prune_ind]
                                prune_layer.rank_mask = rank_mask
                                prune_layer.prune_rate = self.prune_rate
                                prune_layer.uprune_out_channels(rank_mask, self.device)

            # 进行剪枝操作
                else:
                    for idx in self.sorted_conv_layers:
                        feature_result_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
                        prune_rate = self.prune_rate
                        feature_result = get_all_feature(users_model[i], train_dataset, user_groups[idxs_users[i]], idx)
                        feature_result_list[idx] = torch.ones_like(feature_result[idx])
                        hrank_prune(users_model[i], feature_result_list, prune_rate, layer_idx=idx, device=self.device, baselinename="AdapDrop")
                    for idx in self.sorted_fc_layers:
                        prune_rate_list = [torch.tensor(0.)] * len(global_model.prunable_layers)
                        prune_rate = self.prune_rate
                        layer_weight = get_weight(users_model[i], idx)
                        prune_rate_list[idx] = torch.ones_like(layer_weight)
                        hrank_prune(users_model[i], prune_rate_list, prune_rate, layer_idx=idx, device=self.device, baselinename="AdapDrop")
                users_model[i].train()
                self.adapdrop_train([idxs_users[i]], users_model[i], user_groups, epoch,
                                                          train_dataset, train_losses, local_weights, local_losses,
                                                          )
                self.num_weights.append(users_model[i].calc_num_all_active_params())
                self.channels_list.append(users_model[i].get_channels())
                self.nearons_list.append(users_model[i].get_nearons())
                self.log_weight.append([self.num_weights[-1], self.channels_list[-1], self.nearons_list[-1]])
            self.log_weights.append([epoch, self.log_weight[-10:]])
            # 无效轮
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
    t = AdapDrop(args,equal=False, unequal=False, unbalance=False, dirichlet=True, l=2, prune_rate=0.5)
    t.train()

