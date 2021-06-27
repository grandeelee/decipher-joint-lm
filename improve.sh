#!/usr/bin/env bash

WORKDIR="/home/grandee/grandee/projects/TACL_v1"
DATASET=${4}
CONFIG="bert-tiny"
TOKENIZER=${5}
epoch=${2}
save=${3}
improve=${6}
export seed=${1}

/home/grandee/anaconda3/envs/hf451/bin/python improve.py \
	--train_data_file ${WORKDIR}/data/${DATASET}_train.txt \
	--output_dir ${WORKDIR}/models/${DATASET}_mlm_${seed}_${improve} \
	--model_type bert \
	--mlm \
	--config_name ${WORKDIR}/configs/${CONFIG}.json \
	--tokenizer_name ${WORKDIR}/tokenizer/${TOKENIZER} \
	--do_train \
	--do_eval \
	--logging_dir runs/${DATASET}_mlm_${seed}_${improve} \
	--per_gpu_train_batch_size 512 \
	--num_train_epochs ${epoch} \
	--warmup_steps 50 \
	--logging_steps ${save} \
	--save_steps ${save} \
	--overwrite_output_dir \
	--block_size 128 \
	--eval_data_file ${WORKDIR}/data/${DATASET}_valid.txt \
	--per_gpu_eval_batch_size 512 \
	--seed ${seed} \
	--gradient_accumulation_steps 1 \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--learning_rate 2e-3 \
	--improve_list ${improve}
