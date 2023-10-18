import matplotlib.pyplot as plt
import matplotlib
import os
import re
from utils.get_flops import get_flops
from scipy import signal
from numpy import mean
import numpy as np

root_path = "FedDHAD"
data_name = "cifar10"
model_name = "lenet"
target_acc = 0.43
distribution = "dirichlet"
suffix="num200"
cwd = os.getcwd()
root_path = cwd[:cwd.find(root_path) + len(root_path)]
epoch = 500
location = 500
# 获取当前目录
cwd = os.getcwd()
root_name = "FedDHAD"
# 获取根目录
root_path = cwd[:cwd.find(root_name) + len(root_name)]
drop_prune_rate = 0.56
adap_prune_rate = 0.25
seed = 777
acc_round=20
method = "fedAD"

if data_name=="cifar10" and model_name=="lenet":
    # cifar10-lenet
    ###################################
    # FedDH best result
    DH_lam_decay = 0.999
    DH_b_decay = 0.99
    DH_lam_lr = 0.0001
    DH_b_lr = 0.0001

    # FedDHAD best result
    DHAD_lam_decay = 0.9999
    DHAD_b_decay = 0.99
    DHAD_lam_lr = 0.01
    DHAD_b_lr = 0.0001

    ###################################

# fedavg
fedavg_acc_fill_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedavg{seed}_{distribution}{suffix}", f"_test_accuracy.txt")
prox_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedprox{seed}_{distribution}{suffix}",
                                 f"_test_accuracy.txt")
nova_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fednova{seed}_{distribution}{suffix}",
                                  f"_test_accuracy.txt")
dyn_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedDyn{seed}_{distribution}{suffix}",
                                 f"_test_accuracy.txt")
moon_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/moon{seed}_{distribution}{suffix}",
                              f"_test_accuracy.txt")
DH_acc_file_path = os.path.join(root_path,
                                f"result/{data_name}/{model_name}/fedDH{seed}_{distribution}_lam-decay{DH_lam_decay}_b-decay{DH_b_decay}{suffix}/{DH_lam_lr}/{DH_b_lr}",
                                f"_test_accuracy.txt")
# DHAD
DHAD_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/fedDHAD{seed}_{distribution}_lam-decay{DHAD_lam_decay}_b-decay{DHAD_b_decay}{suffix}/{DHAD_lam_lr}/{DHAD_b_lr}",f"_test_accuracy.txt")
# FedAD
AD_weight_path = os.path.join(root_path, f"result/{data_name}/{model_name}/{method}{seed}_{distribution}{suffix}/FedAD/log",f"weights_{seed}.txt")
AD_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/{method}{seed}_{distribution}{suffix}/FedAD", f"{seed}_test_accuracy.txt")
# AFD
afd_weight_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/adapDrop{seed}_{distribution}{suffix}/AdapDrop/log", f"weights_{seed}.txt")
afd_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/adapDrop{seed}_{distribution}{suffix}/AdapDrop", f"{seed}_test_accuracy.txt")
# FedDrop
fedDrop_weight_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/FedDrop{seed}_{distribution}{suffix}/FedDrop/log", f"weights_{seed}.txt")
fedDrop_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/FedDrop{seed}_{distribution}{suffix}/FedDrop", f"{seed}_test_accuracy.txt")
# FjORD
fjord_weight_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/FjORD{seed}_{distribution}{suffix}/log", f"weights_{seed}.txt")
fjord_acc_file_path = os.path.join(root_path,
                                     f"result/{data_name}/{model_name}/FjORD{seed}_{distribution}{suffix}",
                                     f"_test_accuracy.txt")

with open(fedavg_acc_fill_path, "r") as f:
    fedavg_acc = eval(f.read())

with open(DHAD_acc_path, "r") as f:
    DHAD_acc = eval(f.read())

with open(AD_weight_path, "r") as f:
    AD_weight = eval(f.read())
with open(AD_acc_path, "r") as f:
    AD_acc = eval(f.read())

with open(fedDrop_weight_file_path, "r") as f:
    fedDrop_weight = eval(f.read())
with open(fedDrop_acc_file_path, "r") as f:
    fedDrop_acc = eval(f.read())

with open(afd_weight_file_path, "r") as f:
    afd_weight = eval(f.read())
with open(afd_acc_file_path, "r") as f:
    afd_acc = eval(f.read())
try:
    with open(fjord_weight_file_path, "r") as f:
        fjord_weight = eval(f.read())
    with open(fjord_acc_file_path, "r") as f:
        fjord_acc = eval(f.read())
except:
    input("FjORD npath not found!")
    fjord_acc = None
try:
    with open(prox_acc_path, "r") as f:
        prox_acc = eval(f.read())
except:
    prox_acc = None
try:
    with open(nova_acc_file_path, "r") as f:
        nova_acc = eval(f.read())
except:
    nova_acc = None
try:
    with open(dyn_acc_file_path, "r") as f:
        dyn_acc = eval(f.read())
except:
    input("FedDyn path not found!")
    dyn_acc = None
try:
    with open(moon_acc_file_path, "r") as f:
        moon_acc = eval(f.read())
except:
    input("MOON path not found!")
    moon_acc = None
with open(DH_acc_file_path, "r") as f:
    DH_acc = eval(f.read())

model_time = {
    "cnn": {"training_time": 13 / 10},
    "vgg": {"training_time": 80 / 10},
    "resnet": {"training_time": 101 / 10},
    "lenet": {"training_time": 13 / 10}
}

device_speed = 500 / 8  # MB/s
server_speed = 1000 / 8  # MB/s

base_flop, base_size = get_flops(model_name, dataset=data_name)
training_time = model_time[model_name]["training_time"]
comm_time = base_size * 4 * 10 / 1024 / 1024 / device_speed + base_size * 4 * 10 / 1024 / 1024 / server_speed

AD_first_prune_epoch = AD_weight[1][0]

def cal_MFLOPs_and_time(method):
    tmp = 0
    cnn_conv_test = [32, "M", 64, "M", 64]
    lenet_conv_test = [6, "M", 16, "M"]
    vgg_conv_test = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    resnet_conv_test = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]

    cnn_linear_test = [64, 10]
    lenet_linear_test = [120, 84, 10]
    vgg_linear_test = [512, 512, 10]
    resnet_linear_test = [120, 84, 10]

    AD_prune_epoch_train_time = [0]
    fedDrop_drop_epoch_train_time = [0]
    afd_drop_epoch_train_time = [0]
    fjord_drop_epoch_train_time = [0]

    # fedAD
    AD_prune_epoch = []
    AD_prune_epoch_weight = []
    AD_filter = []
    AD_feature = []

    # fedDrop
    fedDrop_drop_epoch = []  # 做dropout的epoch
    fedDrop_drop_epoch_weight = []  # 模型实际的参数量
    fedDrop_filter = []  # filter的数量
    fedDrop_nearon = []  # neuron的数量

    # AFD
    afd_drop_epoch = []
    afd_drop_epoch_weight = []
    afd_filter = []
    afd_feature = []

    # fjord
    fjord_drop_epoch = []
    fjord_filter = []
    fjord_feature = []

    total_fedAvg = 0
    total_fedAD = 0
    total_fedDrop = 0
    total_afd = 0
    total_fjord = 0

    for i in range(1, len(AD_weight), 1):
        AD_prune_epoch.append(AD_weight[i][0])
        for j in range(len(AD_weight[i][1])):
            AD_prune_epoch_weight.append(AD_weight[i][1][j][0])
            AD_filter.append(AD_weight[i][1][j][1])
            AD_feature.append(AD_weight[i][1][j][2])

    if method == "fedAD":
        for i in range(1, len(fedDrop_weight), 1):
            fedDrop_drop_epoch.append(fedDrop_weight[i][0])

            for j in range(len(fedDrop_weight[i][1])):
                fedDrop_drop_epoch_weight.append(fedDrop_weight[i][1][j][0])
                if model_name == 'cnn':
                    fedDrop_filter.append([32, 64, 64])
                    fedDrop_nearon.append(fedDrop_weight[i][1][j][2])
                elif model_name == 'lenet':
                    fedDrop_filter.append([6, 16])
                    fedDrop_nearon.append(fedDrop_weight[i][1][j][2])
                elif model_name == "vgg":
                    fedDrop_filter.append([64, 128, 256, 256, 512, 512, 512, 512])
                    fedDrop_nearon.append(fedDrop_weight[i][1][j][2])
                elif model_name == 'resnet':
                    fedDrop_filter.append(
                        [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512])
                    fedDrop_nearon.append(fedDrop_weight[i][1][j][2])

        for i in range(1, len(afd_weight), 1):
            afd_drop_epoch.append(afd_weight[i][0])
            for j in range(len(afd_weight[i][1])):
                afd_drop_epoch_weight.append(afd_weight[i][1][j][0])
                afd_filter.append(afd_weight[i][1][j][1])
                afd_feature.append(afd_weight[i][1][j][2])

        for i in range(0, len(fjord_weight), 1):
            fjord_drop_epoch.append(fjord_weight[i][0])
            for j in range(len(fjord_weight[i][1])):
                avg_prune_rate = np.average(fjord_weight[i][1][j])
                if model_name == "cnn":
                    fjord_filter.append([int(avg_prune_rate * cnn_conv_test[0]), int(avg_prune_rate * cnn_conv_test[2]), int(avg_prune_rate * cnn_conv_test[4])])
                    fjord_feature.append([int(avg_prune_rate * cnn_linear_test[0]), cnn_linear_test[1]])
                if model_name == "lenet":
                    fjord_filter.append([int(avg_prune_rate * lenet_conv_test[0]), int(avg_prune_rate * lenet_conv_test[2])])
                    fjord_feature.append([int(avg_prune_rate * lenet_linear_test[0]), int(avg_prune_rate * lenet_linear_test[1]), lenet_linear_test[2]])
                if model_name == "vgg":
                    fjord_filter.append([int(avg_prune_rate * vgg_conv_test[0]), int(avg_prune_rate * vgg_conv_test[2]), int(avg_prune_rate * vgg_conv_test[4]), int(avg_prune_rate * vgg_conv_test[5]), int(avg_prune_rate * vgg_conv_test[7]), int(avg_prune_rate * vgg_conv_test[8]), int(avg_prune_rate * vgg_conv_test[10]), int(avg_prune_rate * vgg_conv_test[11])])
                    fjord_feature.append([int(avg_prune_rate * vgg_linear_test[0]), int(avg_prune_rate * vgg_linear_test[1]), vgg_linear_test[2]])
                if model_name == "resnet":
                    temp=[]
                    for i in resnet_conv_test:
                        temp.append(int(avg_prune_rate * resnet_conv_test[i]))
                    fjord_filter.append(temp)
                    fjord_feature.append(
                        [int(avg_prune_rate * resnet_linear_test[0]), int(avg_prune_rate * resnet_linear_test[1]),
                         resnet_linear_test[2]])
    if model_name == "cnn":
        for i in range(len(AD_filter)):
            cnn_conv_test[0] = AD_filter[i][0]
            cnn_conv_test[2] = AD_filter[i][1]
            cnn_conv_test[4] = AD_filter[i][2]
            flop, weight = get_flops(model="cnn", config=cnn_conv_test, linear_config=AD_feature[i], dataset=data_name)
            total_fedAD += flop
            weight_min = min(weight, AD_prune_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                AD_prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fedDrop_filter)):
            if method != "fedAD":
                break
            cnn_conv_test[0] = fedDrop_filter[i][0]
            cnn_conv_test[2] = fedDrop_filter[i][1]
            cnn_conv_test[4] = fedDrop_filter[i][2]
            flop, weight = get_flops(model="cnn", config=cnn_conv_test, linear_config=fedDrop_nearon[i], dataset=data_name)
            total_fedDrop += flop
            weight_min = min(weight, fedDrop_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                fedDrop_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(afd_filter)):
            if method != "fedAD":
                break
            cnn_conv_test[0] = afd_filter[i][0]
            cnn_conv_test[2] = afd_filter[i][1]
            cnn_conv_test[4] = afd_filter[i][2]
            flop, weight = get_flops(model="cnn", config=cnn_conv_test, linear_config=afd_feature[i], dataset=data_name)
            total_afd += flop
            weight_min = min(weight, afd_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                afd_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fjord_filter)):
            if method != "fedAD":
                break
            cnn_conv_test[0] = fjord_filter[i][0]
            cnn_conv_test[2] = fjord_filter[i][1]
            cnn_conv_test[4] = fjord_filter[i][2]
            flop, weight = get_flops(model="cnn", config=cnn_conv_test, linear_config=fjord_feature[i], dataset=data_name)
            total_fjord += flop
            tmp += training_time * flop / base_flop + comm_time * weight / base_size
            if i != 0 and (i + 1) % 10 == 0:
                fjord_drop_epoch_train_time.append(tmp)
                tmp = 0
    if model_name == "vgg":
        for i in range(len(AD_filter)):
            vgg_conv_test[0] = AD_filter[i][0]
            vgg_conv_test[2] = AD_filter[i][1]
            vgg_conv_test[4] = AD_filter[i][2]
            vgg_conv_test[5] = AD_filter[i][3]
            vgg_conv_test[7] = AD_filter[i][4]
            vgg_conv_test[8] = AD_filter[i][5]
            vgg_conv_test[10] = AD_filter[i][6]
            vgg_conv_test[11] = AD_filter[i][7]
            flop, weight = get_flops(model="vgg", config=vgg_conv_test, linear_config=AD_feature[i], dataset=data_name)
            total_fedAD += flop
            weight_min = min(weight, AD_prune_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                AD_prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fedDrop_filter)):
            if method != "fedAD":
                break
            vgg_conv_test[0] = fedDrop_filter[i][0]
            vgg_conv_test[2] = fedDrop_filter[i][1]
            vgg_conv_test[4] = fedDrop_filter[i][2]
            vgg_conv_test[5] = fedDrop_filter[i][3]
            vgg_conv_test[7] = fedDrop_filter[i][4]
            vgg_conv_test[8] = fedDrop_filter[i][5]
            vgg_conv_test[10] = fedDrop_filter[i][6]
            vgg_conv_test[11] = fedDrop_filter[i][7]
            flop, weight = get_flops(model="vgg", config=vgg_conv_test, linear_config=fedDrop_nearon[i], dataset=data_name)
            total_fedDrop += flop
            weight_min = min(weight, fedDrop_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                fedDrop_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(afd_filter)):
            if method != "fedAD":
                break
            vgg_conv_test[0] = afd_filter[i][0]
            vgg_conv_test[2] = afd_filter[i][1]
            vgg_conv_test[4] = afd_filter[i][2]
            vgg_conv_test[5] = afd_filter[i][3]
            vgg_conv_test[7] = afd_filter[i][4]
            vgg_conv_test[8] = afd_filter[i][5]
            vgg_conv_test[10] = afd_filter[i][6]
            vgg_conv_test[11] = afd_filter[i][7]
            flop, weight = get_flops(model="vgg", config=vgg_conv_test, linear_config=afd_feature[i], dataset=data_name)
            total_afd += flop
            weight_min = min(weight, afd_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                afd_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fjord_filter)):
            if method != "fedAD":
                break
            vgg_conv_test[0] = fjord_filter[i][0]
            vgg_conv_test[2] = fjord_filter[i][1]
            vgg_conv_test[4] = fjord_filter[i][2]
            vgg_conv_test[5] = fjord_filter[i][3]
            vgg_conv_test[7] = fjord_filter[i][4]
            vgg_conv_test[8] = fjord_filter[i][5]
            vgg_conv_test[10] = fjord_filter[i][6]
            vgg_conv_test[11] = fjord_filter[i][7]
            flop, weight = get_flops(model="vgg", config=vgg_conv_test, linear_config=fjord_feature[i], dataset=data_name)
            total_fjord += flop
            tmp += training_time * flop / base_flop + comm_time * weight / base_size
            if i != 0 and (i + 1) % 10 == 0:
                fjord_drop_epoch_train_time.append(tmp)
                tmp = 0
    if model_name == "lenet":
        for i in range(len(AD_filter)):
            lenet_conv_test[0] = AD_filter[i][0]
            lenet_conv_test[2] = AD_filter[i][1]
            flop, weight = get_flops(model="lenet", dataset=data_name, config=lenet_conv_test,
                                     linear_config=AD_feature[i])
            total_fedAD += flop
            weight_min = min(weight, AD_prune_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 1 == 0:
                AD_prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fedDrop_filter)):
            if method != "fedAD":
                break
            lenet_conv_test[0] = fedDrop_filter[i][0]
            lenet_conv_test[2] = fedDrop_filter[i][1]
            flop, weight = get_flops(model="lenet", dataset=data_name, config=lenet_conv_test,
                                     linear_config=fedDrop_nearon[i])
            total_fedDrop += flop
            weight_min = min(weight, fedDrop_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 1 == 0:
                fedDrop_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(afd_filter)):
            if method != "fedAD":
                break
            lenet_conv_test[0] = afd_filter[i][0]
            lenet_conv_test[2] = afd_filter[i][1]
            flop, weight = get_flops(model="lenet", dataset=data_name, config=lenet_conv_test,
                                     linear_config=afd_feature[i])
            total_afd += flop
            weight_min = min(weight, afd_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 1 == 0:
                afd_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fjord_filter)):
            if method != "fedAD":
                break
            lenet_conv_test[0] = fjord_filter[i][0]
            lenet_conv_test[2] = fjord_filter[i][1]
            flop, weight = get_flops(model="lenet", config=lenet_conv_test, linear_config=fjord_feature[i], dataset=data_name)
            total_fjord += flop
            tmp += training_time * flop / base_flop + comm_time * weight / base_size
            if i != 0 and (i + 1) % 1 == 0:
                fjord_drop_epoch_train_time.append(tmp)
                tmp = 0
    if model_name == "resnet":
        for i in range(len(AD_filter)):
            resnet_conv_test[0] = AD_filter[i][0]
            resnet_conv_test[1] = AD_filter[i][1]
            resnet_conv_test[2] = AD_filter[i][2]
            resnet_conv_test[3] = AD_filter[i][3]
            resnet_conv_test[4] = AD_filter[i][4]
            resnet_conv_test[5] = AD_filter[i][5]
            resnet_conv_test[6] = AD_filter[i][6]
            resnet_conv_test[7] = AD_filter[i][7]
            resnet_conv_test[8] = AD_filter[i][8]
            resnet_conv_test[9] = AD_filter[i][9]
            resnet_conv_test[10] = AD_filter[i][10]
            resnet_conv_test[11] = AD_filter[i][11]
            resnet_conv_test[12] = AD_filter[i][12]
            resnet_conv_test[13] = AD_filter[i][13]
            resnet_conv_test[14] = AD_filter[i][14]
            resnet_conv_test[15] = AD_filter[i][15]
            resnet_conv_test[16] = AD_filter[i][16]
            flop, weight = get_flops(model="resnet", config=resnet_conv_test, linear_config=AD_feature[i], dataset=data_name)
            total_fedAD += flop
            weight_min = min(weight, AD_prune_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i+1) % 10 == 0:
                AD_prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fedDrop_filter)):
            if method != "fedAD":
                break
            resnet_conv_test[0] = fedDrop_filter[i][0]
            resnet_conv_test[1] = fedDrop_filter[i][1]
            resnet_conv_test[2] = fedDrop_filter[i][2]
            resnet_conv_test[3] = fedDrop_filter[i][3]
            resnet_conv_test[4] = fedDrop_filter[i][4]
            resnet_conv_test[5] = fedDrop_filter[i][5]
            resnet_conv_test[6] = fedDrop_filter[i][6]
            resnet_conv_test[7] = fedDrop_filter[i][7]
            resnet_conv_test[8] = fedDrop_filter[i][8]
            resnet_conv_test[9] = fedDrop_filter[i][9]
            resnet_conv_test[10] = fedDrop_filter[i][10]
            resnet_conv_test[11] = fedDrop_filter[i][11]
            resnet_conv_test[12] = fedDrop_filter[i][12]
            resnet_conv_test[13] = fedDrop_filter[i][13]
            resnet_conv_test[14] = fedDrop_filter[i][14]
            resnet_conv_test[15] = fedDrop_filter[i][15]
            resnet_conv_test[16] = fedDrop_filter[i][16]
            flop, weight = get_flops(model="resnet", config=resnet_conv_test, linear_config=fedDrop_nearon[i],dataset=data_name)
            total_fedDrop += flop
            weight_min = min(weight, fedDrop_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                fedDrop_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(afd_filter)):
            if method != "fedAD":
                break
            resnet_conv_test[0] = afd_filter[i][0]
            resnet_conv_test[1] = afd_filter[i][1]
            resnet_conv_test[2] = afd_filter[i][2]
            resnet_conv_test[3] = afd_filter[i][3]
            resnet_conv_test[4] = afd_filter[i][4]
            resnet_conv_test[5] = afd_filter[i][5]
            resnet_conv_test[6] = afd_filter[i][6]
            resnet_conv_test[7] = afd_filter[i][7]
            resnet_conv_test[8] = afd_filter[i][8]
            resnet_conv_test[9] = afd_filter[i][9]
            resnet_conv_test[10] = afd_filter[i][10]
            resnet_conv_test[11] = afd_filter[i][11]
            resnet_conv_test[12] = afd_filter[i][12]
            resnet_conv_test[13] = afd_filter[i][13]
            resnet_conv_test[14] = afd_filter[i][14]
            resnet_conv_test[15] = afd_filter[i][15]
            resnet_conv_test[16] = afd_filter[i][16]
            flop, weight = get_flops(model="resnet", config=resnet_conv_test, linear_config=afd_feature[i],dataset=data_name)
            total_afd += flop
            weight_min = min(weight, afd_drop_epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i + 1) % 10 == 0:
                afd_drop_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(fjord_filter)):
            if method != "fedAD":
                break
            resnet_conv_test[0] = fjord_filter[i][0]
            resnet_conv_test[1] = fjord_filter[i][1]
            resnet_conv_test[2] = fjord_filter[i][2]
            resnet_conv_test[3] = fjord_filter[i][3]
            resnet_conv_test[4] = fjord_filter[i][4]
            resnet_conv_test[5] = fjord_filter[i][5]
            resnet_conv_test[6] = fjord_filter[i][6]
            resnet_conv_test[7] = fjord_filter[i][7]
            resnet_conv_test[8] = fjord_filter[i][8]
            resnet_conv_test[9] = fjord_filter[i][9]
            resnet_conv_test[10] = fjord_filter[i][10]
            resnet_conv_test[11] = fjord_filter[i][11]
            resnet_conv_test[12] = fjord_filter[i][12]
            resnet_conv_test[13] = fjord_filter[i][13]
            resnet_conv_test[14] = fjord_filter[i][14]
            resnet_conv_test[15] = fjord_filter[i][15]
            resnet_conv_test[16] = fjord_filter[i][16]
            flop, weight = get_flops(model="resnet", config=resnet_conv_test, linear_config=fjord_feature[i], dataset=data_name)
            total_fjord += flop
            tmp += training_time * flop / base_flop + comm_time * weight / base_size
            if i != 0 and (i + 1) % 10 == 0:
                fjord_drop_epoch_train_time.append(tmp)
                tmp = 0
    base_train_time = [0]
    AD_train_time = [0]
    AD_time = [0]
    fedavg_time = [0]
    fedDrop_time = [0]
    afd_time = [0]
    fjord_time = [0]

    # FedAD time
    for i in range(AD_first_prune_epoch):  # FedAD开始剪枝前的时间
        AD_train_time.append((training_time + comm_time) * 10)
        total_fedAD += base_flop*10
    for i in range(1, len(AD_prune_epoch_train_time)):  # FedAD开始剪枝后的时间
        AD_train_time.append(AD_prune_epoch_train_time[i])

    for i in range(epoch):
        base_train_time.append((training_time + comm_time) * 10)

    for i in range(epoch):
        total_fedAvg += base_flop * 10
        AD_time.append(AD_time[i] + AD_train_time[i])
        fedavg_time.append(fedavg_time[i] + base_train_time[i])
        if method=="fedAD":
            fedDrop_time.append(fedDrop_time[i] + fedDrop_drop_epoch_train_time[i])
            # afd_time.append((afd_time[i] + afd_drop_epoch_train_time[i] * 0.75))
            afd_time.append((afd_time[i] + afd_drop_epoch_train_time[i]))
            fjord_time.append((fjord_time[i] + fjord_drop_epoch_train_time[i]))

    if method == "fedAD":
        return  fedavg_time, fedDrop_time, afd_time, AD_time, fjord_time, total_fedAvg, total_fedAD, total_afd, total_fedDrop, total_fjord
    else:
        return fedavg_time, AD_time, total_fedAvg, total_fedAD

def test():
    if method=="fedAD":
        fedAD = signal.savgol_filter(AD_acc, 39, 3)
        fedDrop = signal.savgol_filter(fedDrop_acc, 39, 3)
        fedavg = signal.savgol_filter(fedavg_acc, 39, 3)
        AFD = signal.savgol_filter(afd_acc, 39, 3)
        DHAD = signal.savgol_filter(DHAD_acc, 39, 3)
        fjord = signal.savgol_filter(fjord_acc, 39, 3)

        if prox_acc is not None:
            prox = signal.savgol_filter(prox_acc, 39, 3)
        if nova_acc is not None:
            nova = signal.savgol_filter(nova_acc, 39, 3)
        if dyn_acc is not None:
            dyn = signal.savgol_filter(dyn_acc, 39, 3)
        if moon_acc is not None:
            moon = signal.savgol_filter(moon_acc, 39, 3)
        DH = signal.savgol_filter(DH_acc, 39, 3)


        if prox_acc is not None:
            prox = [round(i, 3) for i in prox]
        if nova_acc is not None:
            nova = [round(i, 3) for i in nova]
        if dyn_acc is not None:
            dyn = [round(i, 3) for i in dyn]
        if moon_acc is not None:
            moon = [round(i, 3) for i in moon]
        DH = [round(i, 3) for i in DH]
        fedAD = [round(i, 3) for i in fedAD]
        fedDrop = [round(i, 3) for i in fedDrop]
        fedavg = [round(i, 3) for i in fedavg]
        AFD = [round(i, 3) for i in AFD]
        DHAD = [round(i, 3) for i in DHAD]
        fjord = [round(i,3) for i in fjord]

        fedavg_time, fedDrop_time, afd_time, AD_time, fjord_time, total_fedAvg, total_fedAD, total_afd, total_fedDrop, total_fjord=cal_MFLOPs_and_time(method)

        for i in fedavg:
            if i >= target_acc:
                y_fedavg = fedavg.index(i)
                break
        print("到达目标精度：{} FedAVG所用时间：{}".format(target_acc, fedavg_time[y_fedavg]))
        if prox_acc is not None:
            for i in prox:
                if i >= target_acc:
                    y_fedprox = prox.index(i)
                    break
            print("到达目标精度：{} FedProx所用时间：{}".format(target_acc, fedavg_time[y_fedprox]))
        if nova_acc is not None:
            for i in nova:
                if i >= target_acc:
                    y_fednova = nova.index(i)
                    break
            print("到达目标精度：{} FedNova所用时间：{}".format(target_acc, fedavg_time[y_fednova]))
        if dyn_acc is not None:
            for i in dyn:
                if i >= target_acc:
                    y_fedDyn = dyn.index(i)
                    break
            try:
                print("到达目标精度：{} FedDyn所用时间：{}".format(target_acc, fedavg_time[y_fedDyn]))
            except:
                print("到达目标精度：{} FedDyn所用时间：NaN".format(target_acc))
        if moon_acc is not None:
            for i in moon:
                if i >= target_acc:
                    y_moon = moon.index(i)
                    break
            try:
                print("到达目标精度：{} MOON所用时间：{}".format(target_acc, fedavg_time[y_moon]))
            except:
                print("到达目标精度：{} MOON所用时间：NaN".format(target_acc))
        for i in DH:
            if i >= target_acc:
                y_fedDH = DH.index(i)
                break
        print("到达目标精度：{} FedDH所用时间：{}".format(target_acc, fedavg_time[y_fedDH]))
        for i in fedDrop:
            if i >= target_acc:
                y_feddrop = fedDrop.index(i)
                break
        try:
            print("到达目标精度：{} FedDrop所用时间：{}".format(target_acc, fedDrop_time[y_feddrop]))
        except:
            print("到达目标精度：{} FedDrop所用时间：NaN".format(target_acc))
        for i in AFD:
            if i >= target_acc:
                y_afd = AFD.index(i)
                break
        try:
            print("到达目标精度：{} AFD所用时间：{}".format(target_acc, afd_time[y_afd]))
        except:
            print("到达目标精度：{} AFD所用时间：NaN".format(target_acc))
        for i in fjord:
            if i >= target_acc:
                y_fjord = fjord.index(i)
                break
        try:
            print("到达目标精度：{} FjORD所用时间：{}".format(target_acc, fjord_time[y_fjord]))
        except:
            print("到达目标精度：{} FjORD所用时间：NaN".format(target_acc))
        for i in fedAD:
            if i >= target_acc:
                y_fedAD = fedAD.index(i)
                break
        print("到达目标精度：{} FedAD所用时间：{}".format(target_acc, AD_time[y_fedAD]))
        for i in DHAD:
            if i >= target_acc:
                y_fedDHAD = DHAD.index(i)
                break
        print("到达目标精度：{} FedDHAD所用时间：{}".format(target_acc, AD_time[y_fedDHAD]))

        print("FedAD MFLOPs:", total_fedAD / (epoch*10*10**6))
        print("FedAvg MFLOPs:", total_fedAvg / (epoch*10*10**6))
        print("FedDrop MFLOPs:", total_fedDrop / (epoch*10*10**6))
        print("AFD MFLOPs:", total_afd / (epoch*10*10**6))
        print("FjORD MFLOPs:", total_fjord / (epoch*10*10**6))

        # print("FedDrop estimate MFLOPs:", total_fedAvg * (fedDrop_time[location - 1] / fedavg_time[location - 1])/ (5*10**9))
        # print("AFD estimate MFLOPs:", total_fedAvg * (afd_time[location - 1] / fedavg_time[location - 1])/ (5*10**9))

        print("FedDHAD Acc:", mean(DHAD_acc[:location][-20:]))
        print("FedDH Acc:", mean(DH_acc[:location][-20:]))
        print("FedAD Acc:", mean(AD_acc[:location][-20:]))
        print("Fedavg Acc:", mean(fedavg_acc[:location][-20:]))
        print("FedProx Acc:", mean(prox_acc[:location][-20:]))
        print("FedNova Acc:", mean(nova_acc[:location][-20:]))
        print("FedDyn Acc:", mean(dyn_acc[:location][-20:]))
        # print("MOON Acc:", mean(moon_acc[:location][-20:]))
        print("FedDrop Acc:", mean(fedDrop_acc[:location][-20:]))
        print("AFD Acc:", mean(afd_acc[:location][-20:]))
        print("FjORD Acc:", mean(fjord_acc[:location][-20:]))


        # print("相比FedDrop时间提升了：", (fedDrop_time[:500][-1] - AD_time[:500][-1]) / fedDrop_time[:500][-1])
        # print("相比FedAvg时间提升了：", (fedavg_time[:500][-1] - AD_time[:500][-1]) / fedavg_time[:500][-1])
        # print("相比AFD时间提升了：", (afd_time[:500][-1] - AD_time[:500][-1]) / afd_time[:500][-1])

        print(f"{model_name} FedAD {location}轮最终时间：", round(AD_time[location - 1], 2))
        print(f"{model_name} FedDrop {location}轮最终时间：", round(fedDrop_time[location - 1], 2))
        print(f"{model_name} FedAvg {location}轮最终时间", round(fedavg_time[location - 1], 2))
        print(f"{model_name} AFD {location}轮最终时间", round(afd_time[location - 1], 2))
        print(f"{model_name} FjORD {location}轮最终时间", round(fjord_time[location - 1], 2))
        print(training_time + comm_time)


        # plt.figure(figsize=[mode["fig"][0], mode["fig"][1]])
        # plt.plot([i / 1000 for i in fedavg_time[:location]], fedavg[:location], label='FedAvg')
        # plt.plot([i / 1000 for i in fedDrop_time[:location]], fedDrop[:location], label='FedDrop')
        # plt.plot([i / 1000 for i in afd_time[:location]], AFD[:location], label='AFD')
        # plt.plot([i / 1000 for i in fjord_time[:location]], fjord[:location], label='FjORD')
        # plt.plot([i / 1000 for i in AD_time[:location]], fedAD[:location], label='FedAD')
        # plt.plot([i / 1000 for i in AD_time[:location]], DHAD[:location], label='FedDHAD')
        # plt.tick_params(labelsize=14)
        # plt.legend(fontsize=mode["legend"])
        # plt.xlabel("Training time (10$^{3}$ s)", fontsize=mode["label"])
        # plt.ylabel("Accuracy", fontsize=mode["label"])
        # plt.tight_layout()
        # plt.show()

if __name__ == '__main__':

    test()
