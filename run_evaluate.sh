#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --output=/home/grandee/grandee/projects/TACL_v1/logs/eval.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

#for model in "mlm" "clm"
#do
#  for seed in 0 7 11 31 41 51 101
#  do
#    sh evaluate.sh cached_2000_NT_2000_NT_${model}_${seed} 2000_2000 cached_2000_NT_2000_NT ${model}
#    sh evaluate.sh cached_2000_NT_2000_OT_${model}_${seed} 2000_2000 cached_2000_NT_2000_OT ${model}
#    sh evaluate.sh cached_2000_NT_2000_TED_${model}_${seed} 2000_2000 cached_2000_NT_2000_TED ${model}
#    sh evaluate.sh cached_2000_NT_2000_WIKI_${model}_${seed} 2000_2000 cached_2000_NT_2000_WIKI ${model}
#    sh evaluate.sh cached_2000_NT_2000_INV_${model}_${seed} 2000_2000 cached_2000_NT_2000_INV ${model}
#    sh evaluate.sh cached_2000_NT_2000_RND_${model}_${seed} 2000_2000 cached_2000_NT_2000_RND ${model}
#    sh evaluate.sh cached_2000_NT_500_NT_${model}_${seed} 2000_500 cached_2000_NT_500_NT ${model}
#    sh evaluate.sh cached_2000_NT_1000_NT_${model}_${seed} 2000_1000 cached_2000_NT_1000_NT ${model}
#    sh evaluate.sh cached_2000_NT_4000_NT_${model}_${seed} 2000_4000 cached_2000_NT_4000_NT ${model}
#  done
#done

for nn in "dist" "iter_dist" "freq" "iter_freq" "control" "low_freq" "mid_freq" "high_freq" "nn_low_freq" "nn_mid_freq" "nn_high_freq"
do
  # sh evaluate.sh cached_2000_NT_2000_NT_mlm_0_${nn} 2000_2000 cached_2000_NT_2000_NT mlm
  # sh evaluate.sh cached_2000_NT_2000_OT_mlm_0_${nn} 2000_2000 cached_2000_NT_2000_OT mlm
  # sh evaluate.sh cached_2000_NT_2000_TED_mlm_0_${nn} 2000_2000 cached_2000_NT_2000_TED mlm
  # sh evaluate.sh cached_2000_NT_2000_WIKI_mlm_0_${nn} 2000_2000 cached_2000_NT_2000_WIKI mlm
  # sh evaluate.sh cached_2000_NT_2000_RND_mlm_0_${nn} 2000_2000 cached_2000_NT_2000_RND mlm
  sh evaluate.sh cached_2000_NT_2000_INV_mlm_0_${nn} 2000_2000 cached_2000_NT_2000_INV mlm
done