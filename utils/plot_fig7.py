import matplotlib.pyplot as plt

# # 创建一个包含两个子图的图（两行一列的布局）
# fig, ax = plt.subplots(2, 1)

# 设置阴影斜线的宽度
plt.rcParams['hatch.linewidth'] = 0.2
# xx = plt.rcParams.keys()
# plt.rcParams['hatch.size'] = 0.2
# 把grid放在最底层
plt.rcParams['axes.axisbelow'] = True

# 绘制第一组数据
# algo1 = ['FedAvg', 'FedProx', 'FedNova', 'SAFA', 'Sageflow', 'AD-PSGD', 'FedSA', 'ASO-Fed', 'FedBuff', 'Port', 'Hrank', 'FedAP', 'HAP', 'DisPFL', 'AEDFL-RL-DWU', 'AEDFL-P', 'AEDFL']
# algo1 = ['FedAvg', 'FedProx', 'FedNova', 'SAFA', 'Sageflow', 'FedSA', 'ASO-Fed', 'Port', 'FedBuff', 'AD-PSGD', 'DisPFL', 'Hrank', 'FedAP', 'HAP', 'AEDFL-RL-DWU', 'AEDFL-P', 'AEDFL']
# algo1 = ['FedAvg', 'FedProx', 'FedNova', 'SAFA', 'Sageflow', 'FedSA', 'ASO-Fed', 'Port', 'FedBuff', 'AD-PSGD', 'DisPFL', 'Hrank', 'FedAP', 'HAP', 'AEDFL']
algo1 = ['FedDHAD', 'FedDH', 'FedAD', 'FedAvg', 'FedProx', 'FedNova', 'MOON', 'FedDyn', 'FedDST', 'PruneFL', 'AFD', 'FedDrop', 'FjORD', 'MOON+FjORD']
x_pos1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# x_pos1 = [1,2,3,4,5,6,7,8,9,10,11]
# x_pos1 = [17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# hatch_list = ['', '', '', '', '', '', '', '', '', '', '//////////','//////////','//////////','//////////','//////////','//////////','//////////','//////////','//////////','//////////']
hatch_list = ['*', 'o', 'O', '.', 'xx', '..', 'x', '+', '++', '--', '-', '**', '//', '///', '']
times1 = [1371,     1426,     1394,   2360,  1994,  2197,  2871,  2256, 2087, 1354, 3909,  2021,  1203, 1785]
acc1 = [0.518, 0.503, 0.499, 0.489, 0.473, 0.497, 0.486, 0.447, 0.4785, 0.3223, 0.434, 0.481, 0.488, 0.4814]
# [0.518, 0.503, 0.499, 0.489, 0.473, 0.497, 0.486, 0.447, 0.434, 0.481, 0.488]
device1 = [0.518, 0.503, 0.499, 0.489, 0.473, 0.497, 0.486, 0.447, 0.4785, 0.3223, 0.434, 0.481, 0.488, 0.4814]
for idx in range(len(times1)):
    plt.bar(x_pos1[idx], times1[idx], width=0.7, label = algo1[idx], hatch=hatch_list[idx])
    # plt.bar(x_pos1[idx], acc1[idx], width=0.7, label = algo1[idx], hatch=hatch_list[idx])
    # plt.bar(x_pos1[idx], device1[idx], width=0.7, label = algo1[idx], hatch=hatch_list[idx])

# 重置颜色
plt.gca().set_prop_cycle(None)


# 绘制第二组数据
# algo2 = ['FedAvg1', 'FedProx1', 'FedNova1', 'SAFA1', 'Sageflow1', 'AD-PSGD1', 'FedSA1', 'ASO-Fed1', 'FedBuff1', 'Port1', 'Hrank1', 'FedAP1', 'HAP1', 'DisPFL1', 'AEDFL-RL-DWU1', 'AEDFL-P1', 'AEDFL1']
# algo2 = ['FedAvg1', 'FedProx1', 'FedNova1', 'SAFA1', 'Sageflow1', 'FedSA1', 'ASO-Fed1', 'Port1', 'FedBuff1', 'AD-PSGD1', 'DisPFL1', 'Hrank1', 'FedAP1', 'HAP1', 'AEDFL1']
algo2 = ['FedDHAD1', 'FedDH1', 'FedAD1', 'FedAvg1', 'FedProx1', 'FedNova1', 'MOON1', 'FedDyn1', 'FedDST1', 'PruneFL1', 'AFD1', 'FedDrop1', 'FjORD1', 'MOON+FjORD']
x_pos2 = [17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# x_pos2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
times2 = [ 1882, 1942, 1916, 3249, 2745, 3025, 3940, 3081, 2679, 1764,6530, 2615, 1653, 2164]
acc2 = [0.633, 0.621, 0.595, 0.566, 0.589, 0.582, 0.610, 0.543, 0.582, 0.392, 0.526, 0.585, 0.593, 0.608]
# times2 = [356, 1221, 395, 1655, 1519, 1573,1963, 2062, 297, 146, 81]
device2 = [0.507, 0.498, 0.476, 0.488, 0.487, 0.482, 0.478, 0.493, 0.427, 0.374,0.473, 0.487, 0.372, 0.367]
for idx in range(len(times2)):
    plt.bar(x_pos2[idx], times2[idx], width=0.7, hatch=hatch_list[idx])
    # plt.bar(x_pos2[idx], acc2[idx], width=0.7, hatch=hatch_list[idx])
    # plt.bar(x_pos2[idx], device2[idx], width=0.7, hatch=hatch_list[idx])
plt.subplots_adjust(right=0.7)
plt.legend(prop={'size': 11}, loc='center right',bbox_to_anchor=(1.5, 0.5)).get_frame().set_edgecolor('black')

plt.xticks([])
# plt.xticks([7, 23], ['Modest Het.', 'High Het.'], fontsize=14)
# plt.xticks([7, 23], ['100 devices', '200 devices'], fontsize=14)
plt.xticks([8, 24], ['Modest Com.', 'Poor Com.'], fontsize=14)
# plt.ylabel('Time (ms)', fontsize=14)
plt.ylabel('Time', fontsize=14)
plt.grid(visible=True, axis='y', linestyle='dashed')
# plt.title('Time to Accuracy with Diverse Bandwidth')
# plt.title('CIFAR10 on LeNet-5')
# plt.show()

# plt.savefig("HeterogeneityImpact.png")
# plt.savefig("HeterogeneityImpact.pdf")
# plt.savefig("DeviceImpact.pdf")
plt.savefig("NetworkImpact.pdf")