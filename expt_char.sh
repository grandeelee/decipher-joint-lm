#!/usr/bin/env bash

#SBATCH --job-name=char
#SBATCH --output=/home/grandee/grandee/projects/TACL_v1/logs/char.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

WORKDIR="/home/grandee/grandee/projects/TACL_v1"
DATASET="cached_102_NT"
CONFIG="bert-tiny"
TOKENIZER="102"
model="mlm"

export seed=0

# mlm
/home/grandee/anaconda3/envs/huggingface/bin/python language_modeling.py \
	--train_data_file ${WORKDIR}/data/${DATASET}_train.txt \
	--output_dir ${WORKDIR}/models/${DATASET}_${model} \
	--model_type bert \
	--${model} \
	--config_name ${WORKDIR}/configs/${CONFIG}.json \
	--tokenizer_name ${WORKDIR}/tokenizer/${TOKENIZER} \
	--do_train \
	--do_eval \
	--logging_dir runs/${DATASET}_${model} \
	--per_gpu_train_batch_size 256 \
	--num_train_epochs 150 \
	--warmup_steps 50 \
	--logging_steps 1500 \
	--save_steps 1500 \
	--overwrite_output_dir \
	--block_size 128 \
	--eval_data_file ${WORKDIR}/data/${DATASET}_valid.txt \
	--per_gpu_eval_batch_size 256 \
	--seed ${seed} \
	--gradient_accumulation_steps 1 \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--learning_rate 2e-3

# clm
model="clm"
/home/grandee/anaconda3/envs/huggingface/bin/python language_modeling.py \
	--train_data_file ${WORKDIR}/data/${DATASET}_train.txt \
	--output_dir ${WORKDIR}/models/${DATASET}_${model} \
	--model_type bert \
	--${model} \
	--config_name ${WORKDIR}/configs/${CONFIG}.json \
	--tokenizer_name ${WORKDIR}/tokenizer/${TOKENIZER} \
	--do_train \
	--do_eval \
	--logging_dir runs/${DATASET}_${model} \
	--per_gpu_train_batch_size 256 \
	--num_train_epochs 40 \
	--warmup_steps 50 \
	--logging_steps 400 \
	--save_steps 400 \
	--overwrite_output_dir \
	--block_size 128 \
	--eval_data_file ${WORKDIR}/data/${DATASET}_valid.txt \
	--per_gpu_eval_batch_size 256 \
	--seed ${seed} \
	--gradient_accumulation_steps 1 \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--learning_rate 2e-3

