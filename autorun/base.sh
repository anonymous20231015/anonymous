export CUDA_VISIBLE_DEVICES=0
seed=777
dir_name="fedavg"
dir_suffix=""
models=("cnn" "lenet")
datasets=("fashionmnist")
dataset=${datasets[0]}
echo ${dataset}
lr=0.0001
decay=0.99
optim="sgd"

equal=0
unequal=0
unbalance=0
dirichlet=1
l=2

epochs=500
local_ep=5
local_bs=10

dir_name="${dir_name}${seed}"
if [ ${unequal} -eq "1" ];then
    dir_name="${dir_name}_unequal_l${l}"
elif [ ${equal} -eq "1" ]; then
    dir_name="${dir_name}_equal_l${l}"
elif [ ${unbalance} -eq "1" ]; then
    dir_name="${dir_name}_unbalance"
elif [ ${dirichlet} -eq "1" ]; then
    dir_name="${dir_name}_dirichlet"
else
    dir_name="${dir_name}_overlap"
fi
dir_name="${dir_name}${dir_suffix}"
echo ${dir_name}

if [ ! -d "./log/${dataset}" ];then
  mkdir -p ./log/${dataset}
else
  echo "文件夹已经存在"
fi

for model in ${models[*]}; do
echo ${model}
  nohup python ../basemain.py --iid 0 --equal ${equal} --unequal ${unequal} --unbalance ${unbalance} --dirichlet ${dirichlet} --dataset ${dataset} \
    --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
    --model ${model} --l ${l} --optim ${optim} --seed ${seed} --lr ${lr}\
      > ./log/${dataset}/"${model}_${dir_name}" 2>&1 &
done

#nohup python ../centralmain.py --iid 1 --share_percent 1 --unequal ${unequal} --result_dir ${dir_name} --epochs ${epochs} \
# --local_bs ${local_bs} --local_ep ${local_ep} --prune_interval ${prune_interval} --prune_rate ${prune_rate} \
# --auto_rate ${auto_rate} --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} &