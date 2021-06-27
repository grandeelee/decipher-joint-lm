#!/usr/bin/env bash
#SBATCH --job-name=run
#SBATCH --output=/home/grandee/grandee/projects/TACL_v1/logs/run.log
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

for nn in "low_freq" "mid_freq" "high_freq" "nn_low_freq" "nn_mid_freq" "nn_high_freq"
do
  sh improve.sh 0 150 1000 cached_2000_NT_2000_NT 2000_2000 ${nn}
  sh improve.sh 0 150 1000 cached_2000_NT_2000_OT 2000_2000 ${nn}
  sh improve.sh 0 150 1000 cached_2000_NT_2000_TED 2000_2000 ${nn}
  sh improve.sh 0 150 1000 cached_2000_NT_2000_WIKI 2000_2000 ${nn}
done