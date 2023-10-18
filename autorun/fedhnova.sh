export CUDA_VISIBLE_DEVICES=0
seed=777
dir_name="fedHnova"
dir_suffix=""
models=("cnn" "lenet")
datasets=("cifar10" "cifar100")
dataset=${datasets[0]}
echo ${dataset}
decay=0.99
lambda_decay=0.9999
gradient_norm=1
optim="sgd"

equal=0
unequal=0
unbalance=0
update_lambda=1
regular=1

l=2

lr_lambdas=(0.1 0.01 0.001)

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
else
    dir_name="${dir_name}_overlap"
fi
dir_name="${dir_name}_d${lambda_decay}"
dir_name="${dir_name}${dir_suffix}"
echo ${dir_name}

if [ ! -d "./log/${dataset}" ];then
  mkdir -p ./log/${dataset}
else
  echo "文件夹已经存在"
fi

for model in ${models[*]}; do
echo ${model}
  if [ ${update_lambda} -eq 1 ]; then
    for lr_lambda in ${lr_lambdas[*]}; do
      echo ${lr_lambda}
      nohup python ../fedhamain.py --iid 0 --equal ${equal} --unequal ${unequal} --unbalance ${unbalance} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
      --update_lambda ${update_lambda} --lr_lambda ${lr_lambda}  --regular ${regular} --gradient_norm ${gradient_norm}\
      --model ${model} --l ${l} --optim ${optim} --lambda_decay ${lambda_decay} --seed ${seed}\
        > ./log/${dataset}/"${model}_${dir_name}_${regular}_${lr_lambda}" 2>&1 &
    done
  else
    nohup python ../fedhamain.py --iid 0 --equal ${equal} --unequal ${unequal} --unbalance ${unbalance} --dataset ${dataset} \
      --result_dir ${dir_name} --epochs ${epochs} --local_bs ${local_bs} --local_ep ${local_ep} --decay ${decay}\
      --update_lambda ${update_lambda} --lr_lambda 0.0  --regular ${regular} --seed ${seed}\
      --model ${model} --l ${l} --optim ${optim} --gradient_norm ${gradient_norm}\
        > ./log/${dataset}/"${model}_${dir_name}" 2>&1 &
  fi
done

#nohup python ../centralmain.py --iid 1 --share_percent 1 --unequal ${unequal} --result_dir ${dir_name} --epochs ${epochs} \
# --local_bs ${local_bs} --local_ep ${local_ep} --prune_interval ${prune_interval} --prune_rate ${prune_rate} \
# --auto_rate ${auto_rate} --server_mu ${server_mu} --client_mu ${client_mu} --auto_mu ${auto_mu} --model ${model} &