#!/usr/bin/env bash
#SBATCH --job-name=run
#SBATCH --output=/home/grandee/grandee/projects/TACL_v1/logs/run.log
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

# domain, order, tok
for seed in 0 7 11 31 41 51 101
do
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_2000_NT 2000_2000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_2000_OT 2000_2000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_2000_TED 2000_2000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_2000_WIKI 2000_2000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_2000_INV 2000_2000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_2000_RND 2000_2000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_500_NT 2000_500
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_1000_NT 2000_1000
  sh expt_mlm.sh ${seed} 150 1000 cached_2000_NT_4000_NT 2000_4000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_2000_NT 2000_2000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_2000_OT 2000_2000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_2000_TED 2000_2000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_2000_WIKI 2000_2000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_2000_INV 2000_2000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_2000_RND 2000_2000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_500_NT 2000_500
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_1000_NT 2000_1000
  sh expt_clm.sh ${seed} 30 200 cached_2000_NT_4000_NT 2000_4000
done
