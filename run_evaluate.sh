#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --output=/home/grandee/grandee/projects/TACL_v1/logs/eval.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

for model in "mlm" "clm"
do
  for seed in 0 7 11 31 41 51 101
  do
    sh evaluate.sh cached_2000_NT_2000_NT_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_2000_OT_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_2000_TED_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_2000_WIKI_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_2000_INV_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_2000_RND_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_500_NT_${model}_${seed} 2000_500 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_1000_NT_${model}_${seed} 2000_1000 cached_2000_NT_2000_NT ${model}
    sh evaluate.sh cached_2000_NT_4000_NT_${model}_${seed} 2000_4000 cached_2000_NT_2000_NT ${model}
  done
done