import matplotlib.pyplot as plt
import matplotlib
import os
import re
from utils.get_flops import get_flops
from scipy import signal
from numpy import mean
import numpy as np
pdf = 0
root_path = "FedDHAD"
model_name = "lenet"
distribution = "dirichlet"
cwd = os.getcwd()
root_path = cwd[:cwd.find(root_path) + len(root_path)]
acc_round=20 # 计算倒数round轮的平均精度
epoch = 500
location = 500
# 获取当前目录
cwd = os.getcwd()
root_name = "FedDHAD"
# 获取根目录
root_path = cwd[:cwd.find(root_name) + len(root_name)]
drop_prune_rate = 0.5
adap_prune_rate = 0.25
seed = 777
data_name = "svhn"
dropout_rate = []

# # cifar10-lenet-776
# ###################################
# # FedDH best result
# DH_lam_decay = 0.999
# DH_b_decay = 0.99
# DH_lam_lr = 0.0001
# DH_b_lr = 0.0001
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# ###################################

# # cifar10-lenet-777
# ###################################
# # FedDH best result
# DH_lam_decay = 0.999
# DH_b_decay = 0.99
# DH_lam_lr = 0.1
# DH_b_lr = 0.0001
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# ###################################

# # cifar100-lenet
# ###################################
# # FedDH best result
# DH_lam_decay = 0.99
# DH_b_decay = 0.99
# DH_lam_lr = 0.01
# DH_b_lr = 0.1
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# ###################################

# # cifar10-cnn-776
# ###################################
# # FedDH best result
# DH_lam_decay = 0.999
# DH_b_decay = 0.99
# DH_lam_lr = 0.01
# DH_b_lr = 0.01
#
# # FedDHAD best result
# DHAD_lam_decay = 0.999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.0001
# DHAD_b_lr = 0.1
#
# ###################################

# # cifar100-cnn
# ###################################
# # FedDH best result
# DH_lam_decay = 0.9999
# DH_b_decay = 0.99
# DH_lam_lr = 0.001
# DH_b_lr = 0.1
#
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# ###################################

# # svhn-cnn
# ###################################
# # FedDH best result
# DH_lam_decay = 0.999
# DH_b_decay = 0.99
# DH_lam_lr = 0.01
# DH_b_lr = 0.001
#
# # FedDHAD best result
# DHAD_lam_decay = 0.99
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.1
# DHAD_b_lr = 0.01
# ###################################

# svhn-lenet
###################################
# FedDH best result
DH_lam_decay = 0.999
DH_b_decay = 0.99
DH_lam_lr = 0.0001
DH_b_lr = 0.001

# FedDHAD best result
DHAD_lam_decay = 0.999
DHAD_b_decay = 0.99
DHAD_lam_lr = 0.0001
DHAD_b_lr = 0.001
###################################

model_time = {
    "cnn": {"training_time": 13 / 10},
    "vgg": {"training_time": 80 / 10},
    "resnet": {"training_time": 101 / 10},
    "lenet": {"training_time": 13 / 10}
}

mode1 = {
    "tick": 17,
    "fig": [5, 4],
    "legend": 14,
    "label": 18
}
mode =mode1

###########################################################################################################
# DH
prox_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedprox{seed}_{distribution}", f"_test_accuracy.txt")
nova_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fednova{seed}_{distribution}", f"_test_accuracy.txt")
dyn_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedDyn{seed}_{distribution}", f"_test_accuracy.txt")
moon_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/moon{seed}_{distribution}", f"_test_accuracy.txt")
DH_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedDH{seed}_{distribution}_lam-decay{DH_lam_decay}_b-decay{DH_b_decay}/{DH_lam_lr}/{DH_b_lr}", f"_test_accuracy.txt")
fedavg_acc_fill_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedavg{seed}_{distribution}", f"_test_accuracy.txt")

DHAD_weight_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedDHAD{seed}_{distribution}_lam-decay{DHAD_lam_decay}_b-decay{DHAD_b_decay}11_27/{DHAD_lam_lr}/{DHAD_b_lr}/log", f"weights_{seed}.txt")
DHAD_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedDHAD{seed}_{distribution}_lam-decay{DHAD_lam_decay}_b-decay{DHAD_b_decay}11_27/{DHAD_lam_lr}/{DHAD_b_lr}", f"_test_accuracy.txt")

with open(fedavg_acc_fill_path, "r") as f:
    fedavg_acc = eval(f.read())
with open(prox_acc_path, "r") as f:
    prox_acc = eval(f.read())
with open(nova_acc_file_path, "r") as f:
    nova_acc = eval(f.read())
with open(dyn_acc_file_path, "r") as f:
    dyn_acc = eval(f.read())
with open(moon_acc_file_path, "r") as f:
    moon_acc = eval(f.read())
with open(DH_acc_file_path, "r") as f:
    DH_acc = eval(f.read())
with open(DHAD_weight_path, "r") as f:
    DHAD_weight = eval(f.read())
with open(DHAD_acc_path, "r") as f:
    DHAD_acc = eval(f.read())

device_speed = 500 / 8  # MB/s
server_speed = 1000 / 8  # MB/s

base_flop, base_size = get_flops(model_name, dataset=data_name)
training_time = model_time[model_name]["training_time"]
comm_time = base_size * 4 * 10 / 1024 / 1024 / device_speed + base_size * 4 * 10 / 1024 / 1024 / server_speed

dropout_epoch = DHAD_weight[1][0]

def computeDHAD_time():
    tmp = 0
    cnn_conv_test = [32, "M", 64, "M", 64]
    lenet_conv_test = [6, "M", 16, "M"]
    vgg_conv_test = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    resnet_conv_test = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    base_train_time = []
    train_time = [0]
    dropout_time = [0]
    fedavg_time = [0]
    prune_epoch = []
    prune_epoch_train_time = []

    epoch_weight = []
    filter_dic = []
    feature_dic = []

    baseline_dropepoch = []
    baseline_dropweight = []
    baseline_dropfilter = []
    baseline_dropfeature = []

    total_fedAD = 0
    total_fedDrop = 0
    total_fedAvg = 0
    FedAD_conv_filter = [0, 0]
    FedAD_fc_nearon = [0, 0, 0]
    FedDrop_fc_nearon = [0, 0, 0]
    for i in range(1, len(DHAD_weight), 1):
        prune_epoch.append(DHAD_weight[i][0])
        for j in range(len(DHAD_weight[i][1])):
            epoch_weight.append(DHAD_weight[i][1][j][0])
            filter_dic.append(DHAD_weight[i][1][j][1])
            feature_dic.append(DHAD_weight[i][1][j][2])

    # for i in range(1, len(baselinedrop_list), 1):
    #     baseline_dropepoch.append(baselinedrop_list[i][0])
    #     for j in range(len(baselinedrop_list[i][1])):
    #         baseline_dropweight.append(baselinedrop_list[i][1][j][0])
    #         if model_name == 'cnn':
    #             baseline_dropfilter.append([32, 64, 64])
    #             baseline_dropfeature.append(baselinedrop_list[i][1][j][2])
    #         elif model_name == 'lenet':
    #             baseline_dropfilter.append([6, 16])
    #             baseline_dropfeature.append(baselinedrop_list[i][1][j][2])
    #         elif model_name == "vgg":
    #             baseline_dropfilter.append([64, 128, 256, 256, 512, 512, 512, 512])
    #             baseline_dropfeature.append(baselinedrop_list[i][1][j][2])

    # baselinedropout
    baseline_drop_time = [0]
    add_drop_time = [0]

    # baselineadapdropout
    baseline_adapdrop_time = [0]
    add_adapdrop_time = [0]
    if model_name == "cnn":
        for i in range(len(filter_dic)):
            cnn_conv_test[0] = filter_dic[i][0]
            cnn_conv_test[2] = filter_dic[i][1]
            cnn_conv_test[4] = filter_dic[i][2]
            flop, weight = get_flops(config=cnn_conv_test, linear_config=feature_dic[i],dataset=data_name)
            weight_min = min(weight, epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i+1) % 10 == 0:
                prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(baseline_dropfilter)):
            cnn_conv_test[0] = baseline_dropfilter[i][0]
            cnn_conv_test[2] = baseline_dropfilter[i][1]
            cnn_conv_test[4] = baseline_dropfilter[i][2]
            flop, weight = get_flops(config=cnn_conv_test, linear_config=baseline_dropfilter[i],dataset=data_name)
            weight_min = min(weight, baseline_dropweight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                baseline_drop_time.append(tmp)
                tmp = 0
    if model_name == "lenet":
        for i in range(len(filter_dic)):
            lenet_conv_test[0] = filter_dic[i][0]
            lenet_conv_test[2] = filter_dic[i][1]
            flop, weight = get_flops(model = "lenet", dataset= data_name, config=lenet_conv_test, linear_config=feature_dic[i])
            # print("FedAD:", flop)
            weight_min = min(weight, epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i+1) % 10 == 0:
                prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(baseline_dropfilter)):
            lenet_conv_test[0] = baseline_dropfilter[i][0]
            lenet_conv_test[2] = baseline_dropfilter[i][1]
            FedDrop_fc_nearon = np.sum([FedDrop_fc_nearon, baseline_dropfeature[i]], axis=0)
            flop, weight = get_flops(model="lenet",dataset= data_name, config=lenet_conv_test, linear_config=baseline_dropfeature[i])
            # print("FedDrop:",flop)
            # print("Fedavg:",base_flop)
            weight_min = min(weight, baseline_dropweight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                baseline_drop_time.append(tmp)
                tmp = 0
    if model_name == "resnet":
        for i in range(len(filter_dic)):
            resnet_conv_test[0] = filter_dic[i][0]
            resnet_conv_test[1] = filter_dic[i][1]
            resnet_conv_test[2] = filter_dic[i][2]
            resnet_conv_test[3] = filter_dic[i][3]
            resnet_conv_test[4] = filter_dic[i][4]
            resnet_conv_test[5] = filter_dic[i][5]
            resnet_conv_test[6] = filter_dic[i][6]
            resnet_conv_test[7] = filter_dic[i][7]
            resnet_conv_test[8] = filter_dic[i][8]
            resnet_conv_test[9] = filter_dic[i][9]
            resnet_conv_test[10] = filter_dic[i][10]
            resnet_conv_test[11] = filter_dic[i][11]
            resnet_conv_test[12] = filter_dic[i][12]
            resnet_conv_test[13] = filter_dic[i][13]
            resnet_conv_test[14] = filter_dic[i][14]
            resnet_conv_test[15] = filter_dic[i][15]
            resnet_conv_test[16] = filter_dic[i][16]
            flop, weight = get_flops(config=resnet_conv_test, linear_config=feature_dic[i],dataset=data_name)
            weight_min = min(weight, epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i+1) % 10 == 0:
                prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(baseline_dropfilter)):
            resnet_conv_test[0] = filter_dic[i][0]
            resnet_conv_test[1] = filter_dic[i][1]
            resnet_conv_test[2] = filter_dic[i][2]
            resnet_conv_test[3] = filter_dic[i][3]
            resnet_conv_test[4] = filter_dic[i][4]
            resnet_conv_test[5] = filter_dic[i][5]
            resnet_conv_test[6] = filter_dic[i][6]
            resnet_conv_test[7] = filter_dic[i][7]
            resnet_conv_test[8] = filter_dic[i][8]
            resnet_conv_test[9] = filter_dic[i][9]
            resnet_conv_test[10] = filter_dic[i][10]
            resnet_conv_test[11] = filter_dic[i][11]
            resnet_conv_test[12] = filter_dic[i][12]
            resnet_conv_test[13] = filter_dic[i][13]
            resnet_conv_test[14] = filter_dic[i][14]
            resnet_conv_test[15] = filter_dic[i][15]
            resnet_conv_test[16] = filter_dic[i][16]
            flop, weight = get_flops(config=resnet_conv_test, linear_config=baseline_dropfilter[i],dataset=data_name)
            weight_min = min(weight, baseline_dropweight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                baseline_drop_time.append(tmp)
                tmp = 0
    for i in range(dropout_epoch):
        train_time.append((training_time+comm_time) * 10)
    for i in range(1, len(prune_epoch_train_time)):
        train_time.append(prune_epoch_train_time[i])
    for i in range(epoch):
        base_train_time.append((training_time + comm_time)*10)
    for i in range(epoch):
        dropout_time.append(dropout_time[i]+train_time[i])
        fedavg_time.append(fedavg_time[i]+base_train_time[i])
        # add_drop_time.append(add_drop_time[i] + baseline_drop_time[i])
        # add_adapdrop_time.append((add_adapdrop_time[i] + base_train_time[i] * 0.75))
        total_fedAvg += base_flop
    total_fedAD += base_flop
    print("FedDHAD计算量：", total_fedAD * 50 / 2)
    print("FedAvg计算量：", 10 * total_fedAvg / 2)
    return dropout_time

def test():
    fedavg = signal.savgol_filter(fedavg_acc, 39, 3)
    prox = signal.savgol_filter(prox_acc, 39, 3)
    nova = signal.savgol_filter(nova_acc, 39, 3)
    dyn = signal.savgol_filter(dyn_acc, 39, 3)
    moon = signal.savgol_filter(moon_acc, 39, 3)
    DH = signal.savgol_filter(DH_acc, 39, 3)
    DHAD = signal.savgol_filter(DHAD_acc, 39, 3)

    fedavg = [round(i, 3) for i in fedavg]
    prox = [round(i, 3) for i in prox]
    nova = [round(i, 3) for i in nova]
    dyn = [round(i, 3) for i in dyn]
    moon = [round(i, 3) for i in moon]
    DH = [round(i, 3) for i in DH]
    DHAD = [round(i, 3) for i in DHAD]

    base_train_time = []
    fedavg_time = [0]
    total_fedAvg = 0

    DHAD_time = computeDHAD_time()

    for i in range(epoch):
        base_train_time.append((training_time + comm_time)*10)
    for i in range(epoch):
        fedavg_time.append(fedavg_time[i]+base_train_time[i])
        total_fedAvg += base_flop

    print("FedDHAD Acc:", mean(DHAD_acc[:location][-acc_round:]))
    print("FedDH Acc:", mean(DH_acc[:location][-acc_round:]))
    print("FedAVG Acc:",mean(fedavg_acc[:location][-acc_round:]))
    print("FedProx Acc:",mean(prox_acc[:location][-acc_round:]))
    print("FedNova Acc:",mean(nova_acc[:location][-acc_round:]))
    print("FedDyn Acc:",mean(dyn_acc[:location][-acc_round:]))
    print("MOON Acc:",mean(moon_acc[:location][-acc_round:]))
    print("相比FedAvg时间提升了：", (fedavg_time[:500][-1] - DHAD_time[:500][-1]) / fedavg_time[:500][-1])
    plt.figure(figsize=[mode["fig"][0], mode["fig"][1]])
    plt.plot([i / 1000 for i in fedavg_time[:location]], fedavg[:location], label='FedAvg')
    plt.plot([i / 1000 for i in fedavg_time[:location]], prox[:location], label='FedProx')
    plt.plot([i / 1000 for i in fedavg_time[:location]], nova[:location], label='FedNova')
    plt.plot([i / 1000 for i in fedavg_time[:location]], moon[:location], label='MOON')
    plt.plot([i / 1000 for i in fedavg_time[:location]], dyn[:location], label='FedDyn')
    plt.plot([i / 1000 for i in fedavg_time[:location]], DH[:location], label='FedDH')
    plt.plot([i / 1000 for i in DHAD_time[:location]], DHAD[:location], label='FedDHAD')
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=mode["legend"])
    plt.xlabel("Training time (10$^{3}$ s)", fontsize=mode["label"])
    plt.ylabel("Accuracy", fontsize=mode["label"])
    plt.tight_layout()

    if pdf:
        plt.savefig(f'FedDHAD_{data_name}_{model_name}.pdf')
    else:
        plt.show()

if __name__ == '__main__':
    test()


