#!/usr/bin/env bash

WORKDIR=$(pwd)
DATASET=${4}
CONFIG="bert-tiny"
TOKENIZER=${5}
epoch=${2}
save=${3}
export seed=${1}


/home/grandee/anaconda3/envs/hf451/bin/python language_modeling.py \
	--train_data_file ${WORKDIR}/data/${DATASET}_train.txt \
	--output_dir ${WORKDIR}/models/${DATASET}_clm_${seed} \
	--model_type bert \
	--clm \
	--config_name ${WORKDIR}/configs/${CONFIG}.json \
	--tokenizer_name ${WORKDIR}/tokenizer/${TOKENIZER} \
	--do_train \
	--do_eval \
	--logging_dir runs/${DATASET}_clm_${seed} \
	--per_gpu_train_batch_size 384 \
	--num_train_epochs ${epoch} \
	--warmup_steps 50 \
	--logging_steps ${save} \
	--save_steps ${save} \
	--overwrite_output_dir \
	--block_size 128 \
	--eval_data_file ${WORKDIR}/data/${DATASET}_valid.txt \
	--per_gpu_eval_batch_size 384 \
	--seed ${seed} \
	--gradient_accumulation_steps 1 \
	--weight_decay 0.01 \
	--adam_epsilon 1e-6 \
	--learning_rate 2e-3





