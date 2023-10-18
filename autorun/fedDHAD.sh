export CUDA_VISIBLE_DEVICES=0
seed=777
dir_name="fedDHAD"
dir_suffix=""
models=("cnn")
datasets=("svhn")
dataset=${datasets[0]}
echo ${dataset}
decay=0.99
lambda_decay=0.99
b_decay=0.99
lr_lambdas=(0.1)
lr_bs=(0.1 0.01 0.001 0.0001)
start_epoch=0
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
dir_name="${dir_name}_lam-decay${lambda_decay}"
dir_name="${dir_name}_b-decay${b_decay}"
dir_name="${dir_name}${dir_suffix}"
echo ${dir_name}

if [ ! -d "./log/${dataset}" ];then
  mkdir -p ./log/${dataset}
else
  echo "文件夹已经存在"
fi

for model in ${models[*]}; do
echo ${model}
  for lr_lambda in ${lr_lambdas[*]}; do
    echo ${lr_lambda}
    for lr_b in ${lr_bs[*]}; do
      nohup python ../fedDHADmain.py --iid 0 --equal ${equal} --unequal ${unequal} --unbalance ${unbalance} --dirichlet ${dirichlet} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
      --lr_lambda ${lr_lambda}  --lr_b ${lr_b} --b_decay ${b_decay} --start_epoch ${start_epoch} \
      --model ${model} --l ${l} --optim ${optim} --lambda_decay ${lambda_decay} --seed ${seed}\
        > ./log/${dataset}/"${model}_${dir_name}_${lr_lambda}_${lr_b}" 2>&1 &
    done
  done
done

#nohup python ../centralmain.py --iid 1 --share_percent 1 --unequal ${unequal} --result_dir ${dir_name} --epochs ${epochs} \
# --local_bs ${local_bs} --local_ep ${local_ep} --prune_interval ${prune_interval} --prune_rate ${prune_rate} \
# --auto_rate ${auto_rate} --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} &