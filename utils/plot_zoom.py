import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import matplotlib
import os
import re
from utils.get_flops import get_flops
from scipy import signal
from numpy import mean
import numpy as np

fig, ax = plt.subplots()
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
# # FedDHScaffold best result
# DHSca_lam_decay = 0.999
# DHSca_b_decay = 0.99
# DHSca_lam_lr = 0.01
# DHSca_b_lr = 0.01
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# # FedDHADScaffold best result
# DHADSca_lam_decay = 0.999
# DHADSca_b_decay = 0.99
# DHADSca_lam_lr = 0.01
# DHADSca_b_lr = 0.01
# ###################################

# # cifar10-lenet-777
# ###################################
# # FedDH best result
# DH_lam_decay = 0.999
# DH_b_decay = 0.99
# DH_lam_lr = 0.1
# DH_b_lr = 0.0001
#
# # FedDHScaffold best result
# DHSca_lam_decay = 0.999
# DHSca_b_decay = 0.99
# DHSca_lam_lr = 0.001
# DHSca_b_lr = 0.001
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# # FedDHADScaffold best result
# DHADSca_lam_decay = 0.999
# DHADSca_b_decay = 0.99
# DHADSca_lam_lr = 0.01
# DHADSca_b_lr = 0.01
# ###################################

# # cifar100-lenet
# ###################################
# # FedDH best result
# DH_lam_decay = 0.99
# DH_b_decay = 0.99
# DH_lam_lr = 0.01
# DH_b_lr = 0.1
#
# # FedDHScaffold best result
# DHSca_lam_decay = 0.9999
# DHSca_b_decay = 0.99
# DHSca_lam_lr = 0.1
# DHSca_b_lr = 0.1
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# # FedDHADScaffold best result
# DHADSca_lam_decay = 0.999
# DHADSca_b_decay = 0.99
# DHADSca_lam_lr = 0.01
# DHADSca_b_lr = 0.01
# ###################################

# # cifar10-cnn-776
# ###################################
# # FedDH best result
# DH_lam_decay = 0.999
# DH_b_decay = 0.99
# DH_lam_lr = 0.01
# DH_b_lr = 0.01
#
# # FedDHScaffold best result
# DHSca_lam_decay = 0.99
# DHSca_b_decay = 0.99
# DHSca_lam_lr = 0.1
# DHSca_b_lr = 0.01
#
# # FedDHAD best result
# DHAD_lam_decay = 0.999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.0001
# DHAD_b_lr = 0.1
#
# # FedDHADScaffold best result
# DHADSca_lam_decay = 0.999
# DHADSca_b_decay = 0.99
# DHADSca_lam_lr = 0.01
# DHADSca_b_lr = 0.01
# ###################################

# # cifar100-cnn
# ###################################
# # FedDH best result
# DH_lam_decay = 0.9999
# DH_b_decay = 0.99
# DH_lam_lr = 0.001
# DH_b_lr = 0.1
#
# # FedDHScaffold best result
# DHSca_lam_decay = 0.99
# DHSca_b_decay = 0.99
# DHSca_lam_lr = 0.1
# DHSca_b_lr = 0.01
#
# # FedDHAD best result
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
#
# # FedDHADScaffold best result
# DHADSca_lam_decay = 0.999
# DHADSca_b_decay = 0.99
# DHADSca_lam_lr = 0.01
# DHADSca_b_lr = 0.01
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
# DHAD_lam_decay = 0.9999
# DHAD_b_decay = 0.99
# DHAD_lam_lr = 0.01
# DHAD_b_lr = 0.0001
# ###################################

# svhn-lenet
###################################
# FedDH best result
DH_lam_decay = 0.999
DH_b_decay = 0.99
DH_lam_lr = 0.0001
DH_b_lr = 0.001

# FedDHAD best result
DHAD_lam_decay = 0.9999
DHAD_b_decay = 0.99
DHAD_lam_lr = 0.01
DHAD_b_lr = 0.0001
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
prox_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedprox{seed}_{distribution}", f"_test_accuracy.txt")
nova_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fednova{seed}_{distribution}", f"_test_accuracy.txt")
scaffold_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/scaffold{seed}_{distribution}", f"_test_accuracy.txt")
dyn_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDyn{seed}_{distribution}", f"_test_accuracy.txt")
moon_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/moon{seed}_{distribution}", f"_test_accuracy.txt")
DH_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDH{seed}_{distribution}_lam-decay{DH_lam_decay}_b-decay{DH_b_decay}/{DH_lam_lr}/{DH_b_lr}", f"_test_accuracy.txt")
# DHSca_acc_file_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDHScaffold{seed}_{distribution}_lam-decay{DHSca_lam_decay}_b-decay{DHSca_b_decay}/{DHSca_lam_lr}/{DHSca_b_lr}", f"_test_accuracy.txt")
fedavg_acc_fill_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedavg{seed}_{distribution}", f"_test_accuracy.txt")

# DHAD_weight_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDHAD{seed}_{distribution}_lam-decay{DHAD_lam_decay}_b-decay{DHAD_b_decay}/{DHAD_lam_lr}/{DHAD_b_lr}/log", f"weights_{seed}.txt")
# DHAD_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDHAD{seed}_{distribution}_lam-decay{DHAD_lam_decay}_b-decay{DHAD_b_decay}/{DHAD_lam_lr}/{DHAD_b_lr}", f"_test_accuracy.txt")

# DHADSca_weight_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDHADScaffold{seed}_{distribution}_lam-decay{DHADSca_lam_decay}_b-decay{DHADSca_b_decay}/{DHADSca_lam_lr}/{DHADSca_b_lr}/log", f"weights_{seed}.txt")
# DHADSca_acc_path = os.path.join(root_path, f"result/{data_name}/{model_name}/seed{seed}/fedDHADScaffold{seed}_{distribution}_lam-decay{DHADSca_lam_decay}_b-decay{DHADSca_b_decay}/{DHADSca_lam_lr}/{DHADSca_b_lr}", f"_test_accuracy.txt")

with open(fedavg_acc_fill_path, "r") as f:
    fedavg_acc = eval(f.read())
with open(prox_acc_path, "r") as f:
    prox_acc = eval(f.read())
with open(nova_acc_file_path, "r") as f:
    nova_acc = eval(f.read())
with open(scaffold_acc_file_path, "r") as f:
    scaffold_acc = eval(f.read())
with open(dyn_acc_file_path, "r") as f:
    dyn_acc = eval(f.read())
with open(moon_acc_file_path, "r") as f:
    moon_acc = eval(f.read())
with open(DH_acc_file_path, "r") as f:
    DH_acc = eval(f.read())
# with open(DHSca_acc_file_path, "r") as f:
#     DHSca_acc = eval(f.read())
# with open(DHAD_weight_path, "r") as f:
#     DHAD_weight = eval(f.read())
# with open(DHAD_acc_path, "r") as f:
#     DHAD_acc = eval(f.read())
# with open(DHADSca_weight_path, "r") as f:
#     DHADSca_weight = eval(f.read())
# with open(DHADSca_acc_path, "r") as f:
#     DHADSca_acc = eval(f.read())

device_speed = 500 / 8  # MB/s
server_speed = 1000 / 8  # MB/s

base_flop, base_size = get_flops(model_name, dataset=data_name)
training_time = model_time[model_name]["training_time"]
comm_time = base_size * 4 * 10 / 1024 / 1024 / device_speed + base_size * 4 * 10 / 1024 / 1024 / server_speed

# dropout_epoch = DHAD_weight[1][0]

def computeDHAD_time():
    tmp = 0
    cnn_conv_test = [32, "M", 64, "M", 64]
    lenet_conv_test = [6, "M", 16, "M"]
    vgg_conv_test = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
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
    if model_name == "vgg":
        for i in range(len(filter_dic)):
            vgg_conv_test[0] = filter_dic[i][0]
            vgg_conv_test[2] = filter_dic[i][1]
            vgg_conv_test[4] = filter_dic[i][2]
            vgg_conv_test[5] = filter_dic[i][3]
            vgg_conv_test[7] = filter_dic[i][4]
            vgg_conv_test[8] = filter_dic[i][5]
            vgg_conv_test[10] = filter_dic[i][6]
            vgg_conv_test[11] = filter_dic[i][7]
            flop, weight = get_flops(model = "vgg", dataset= data_name, config=vgg_conv_test, linear_config=feature_dic[i])
            # print("FedAD:", flop)
            weight_min = min(weight, epoch_weight[i])
            tmp += training_time * flop / base_flop + comm_time * weight_min / base_size
            if i != 0 and (i+1) % 10 == 0:
                prune_epoch_train_time.append(tmp)
                tmp = 0
        for i in range(len(baseline_dropfilter)):
            vgg_conv_test[0] = baseline_dropfilter[i][0]
            vgg_conv_test[2] = baseline_dropfilter[i][1]
            vgg_conv_test[4] = baseline_dropfilter[i][2]
            vgg_conv_test[5] = baseline_dropfilter[i][3]
            vgg_conv_test[7] = baseline_dropfilter[i][4]
            vgg_conv_test[8] = baseline_dropfilter[i][5]
            vgg_conv_test[10] = baseline_dropfilter[i][6]
            vgg_conv_test[11] = baseline_dropfilter[i][7]
            FedDrop_fc_nearon = np.sum([FedDrop_fc_nearon, baseline_dropfeature[i]], axis=0)
            flop, weight = get_flops(model="vgg",dataset= data_name, config=vgg_conv_test, linear_config=baseline_dropfeature[i])
            # print("FedDrop:",flop)
            # print("Fedavg:",base_flop)
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
    scaffold = signal.savgol_filter(scaffold_acc, 39, 3)
    dyn = signal.savgol_filter(dyn_acc, 39, 3)
    moon = signal.savgol_filter(moon_acc, 39, 3)
    DH = signal.savgol_filter(DH_acc, 39, 3)
    # DHSca = signal.savgol_filter(DHSca_acc, 39, 3)
    # DHAD = signal.savgol_filter(DHAD_acc, 39, 3)
    # DHADSca = signal.savgol_filter(DHADSca_acc, 39, 3)

    fedavg = [round(i, 3) for i in fedavg]
    prox = [round(i, 3) for i in prox]
    nova = [round(i, 3) for i in nova]
    scaffold = [round(i, 3) for i in scaffold]
    dyn = [round(i, 3) for i in dyn]
    moon = [round(i, 3) for i in moon]
    DH = [round(i, 3) for i in DH]
    # DHSca = [round(i, 3) for i in DHSca]
    # DHAD = [round(i, 3) for i in DHAD]
    # DHADSca = [round(i, 3) for i in DHADSca]

    base_train_time = []
    fedavg_time = [0]
    total_fedAvg = 0

    # DHAD_time = computeDHAD_time()
    # DHADSca_time = computeDHAD_time()

    for i in range(epoch):
        base_train_time.append((training_time + comm_time)*10)
    for i in range(epoch):
        fedavg_time.append(fedavg_time[i]+base_train_time[i])
        total_fedAvg += base_flop

    # print("FedDHAD Acc:", mean(DHAD_acc[:location][-acc_round:]))
    # print("FedDHADScaffold Acc:", mean(DHADSca_acc[:location][-acc_round:]))
    print("FedDH Acc:", mean(DH_acc[:location][-acc_round:]))
    # print("FedDHScaffold Acc:", mean(DHSca_acc[:location][-acc_round:]))
    print("FedAVG Acc:",mean(fedavg_acc[:location][-acc_round:]))
    print("FedProx Acc:",mean(prox_acc[:location][-acc_round:]))
    print("FedNova Acc:",mean(nova_acc[:location][-acc_round:]))
    print("Scaffold Acc:",mean(scaffold_acc[:location][-acc_round:]))
    print("FedDyn Acc:",mean(dyn_acc[:location][-acc_round:]))
    print("MOON Acc:",mean(moon_acc[:location][-acc_round:]))
    # print("相比FedAvg时间提升了：", (fedavg_time[:500][-1] - DHAD_time[:500][-1]) / fedavg_time[:500][-1])
    # plt.figure(figsize=[mode["fig"][0], mode["fig"][1]])
    # plt.plot(DHADSca_time[:location], DHADSca[:location], label='FedDHADScaffold')
    # plt.plot(fedavg_time[:location], DHSca[:location], label='FedDHScaffold')
    ax.plot([i / 1000 for i in fedavg_time[:location]], fedavg[:location], label='FedAvg')
    ax.plot([i / 1000 for i in fedavg_time[:location]], prox[:location], label='FedProx')
    ax.plot([i / 1000 for i in fedavg_time[:location]], nova[:location], label='FedNova')
    # plt.plot([i / 1000 for i in fedavg_time[:location]], scaffold[:location], label='Scaffold')
    ax.plot([i / 1000 for i in fedavg_time[:location]], moon[:location], label='MOON')
    ax.plot([i / 1000 for i in fedavg_time[:location]], dyn[:location], label='FedDyn')
    ax.plot([i / 1000 for i in fedavg_time[:location]], DH[:location], label='FedDH')
    # plt.plot([i / 1000 for i in DHAD_time[:location]], DHAD[:location], label='FedDHAD')
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=mode["legend"])
    plt.xlabel("Training time (10$^{3}$ s)", fontsize=mode["label"])
    plt.ylabel("Accuracy", fontsize=mode["label"])
    plt.tight_layout()

    s_loc = 400
    axins = zoomed_inset_axes(ax, 2.5, loc=1)  # zoom = 6
    axins.plot([i / 1000 for i in fedavg_time[s_loc:location]], fedavg[s_loc:location], label='FedAvg')
    axins.plot([i / 1000 for i in fedavg_time[s_loc:location]], prox[s_loc:location], label='FedProx')
    axins.plot([i / 1000 for i in fedavg_time[s_loc:location]], nova[s_loc:location], label='FedNova')
    # plt.plot([i / 1000 for i in fedavg_time[:location]], scaffold[:location], label='Scaffold')
    axins.plot([i / 1000 for i in fedavg_time[s_loc:location]], moon[s_loc:location], label='MOON')
    axins.plot([i / 1000 for i in fedavg_time[s_loc:location]], dyn[s_loc:location], label='FedDyn')
    axins.plot([i / 1000 for i in fedavg_time[s_loc:location]], DH[s_loc:location], label='FedDH')
    # plt.plot([i / 1000 for i in DHAD_time[:location]], DHAD[:location], label='FedDHAD')
    # 控制放大的区域
    axins.set_xlim(6, 6.8)  # Limit the region for zoom
    axins.set_ylim(0.80, 0.90)

    plt.xticks(visible=False)  # Not present ticks
    plt.yticks(visible=False)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=False)  # labels along the bottom edge are off
    #
    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    # 两条连接线
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="black", zorder=3)
    # ax.legend().get_frame().set_edgecolor('black')
    if pdf:
        plt.savefig(f'FedDHAD_{data_name}_{model_name}.pdf')
    else:
        plt.show()

if __name__ == '__main__':
    test()

