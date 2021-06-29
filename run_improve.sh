#!/usr/bin/env bash
#SBATCH --job-name=improve
#SBATCH --output=/home/grandee/grandee/projects/TACL_v1/logs/improve.log
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

for nn in "dist" "iter_dist"
do
  sh improve.sh 0 50 500 cached_2000_NT_2000_NT 2000_2000 ${nn} models/cached_2000_NT_2000_NT_mlm_0
  sh improve.sh 0 50 500 cached_2000_NT_2000_OT 2000_2000 ${nn} models/cached_2000_NT_2000_OT_mlm_0
  sh improve.sh 0 50 500 cached_2000_NT_2000_TED 2000_2000 ${nn} models/cached_2000_NT_2000_TED_mlm_0
  sh improve.sh 0 50 500 cached_2000_NT_2000_WIKI 2000_2000 ${nn} models/cached_2000_NT_2000_WIKI_mlm_0
  sh improve.sh 0 50 500 cached_2000_NT_2000_RND 2000_2000 ${nn} models/cached_2000_NT_2000_RND_mlm_0
  sh improve.sh 0 50 500 cached_2000_NT_2000_INV 2000_2000 ${nn} models/cached_2000_NT_2000_INV_mlm_0
done

