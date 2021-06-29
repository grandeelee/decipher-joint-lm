#!/usr/bin/env bash

WORKDIR=$(pwd)
path=${1}
TOKENIZER=${2}
DATASET=${3}
model=${4}
# get args
/home/grandee/anaconda3/envs/hf451/bin/python evaluate.py \
  --model_name_or_path ${WORKDIR}/models/${path} \
  --tokenizer_name ${WORKDIR}/tokenizer/${TOKENIZER} \
  --eval_retrieval ${WORKDIR}/data/cached_${TOKENIZER}_retrieval.txt \
  --eval_perplexity ${WORKDIR}/data/${DATASET}_valid.txt \
  --eval_layers "0,4,8,12" \
  --${model}


