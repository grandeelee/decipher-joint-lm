#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""
import json
import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

import torch
from torch.utils.data import ConcatDataset
from utils.custom_dataset import LoadTextDataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    BertLMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils.utils import _sorted_checkpoints_from_path, get_unigram_from_tokenized

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    improve_list: str = field(
        default="low_freq",
        metadata={"help": "choose which set of vocab to improve"}
    )
    load_model_from: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    clm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


# align_list1 = []
# align_list2 = []
# cnt = 0
# op = ''


def get_list(arg, model, count=0):
    if arg in ["low_freq" "mid_freq" "high_freq" "nn_low_freq" "nn_mid_freq" "nn_high_freq"]:
        with open('improve_list.json', 'r') as f:
            improvement_lists = json.load(f)
        list1 = improvement_lists[arg]
        list2 = [i + 2000 for i in list1]

    elif arg in ["freq", "iter_freq"]:
        with open("data/cached_2000_NT_train.txt", 'rb') as f:
            data = pickle.load(f)
        freq_list = get_unigram_from_tokenized(data)
        list1 = [i for i, j in freq_list.items() if i not in [0, 1, 2, 3, 4]]
        with torch.no_grad():
            e1 = model.bert.embeddings.word_embeddings.weight[:2000]
            e2 = model.bert.embeddings.word_embeddings.weight[2000:]
            ssm = e1 @ e2.T
            ssm = ssm[list1]
            list2 = torch.argmax(ssm, -1).tolist()
            if arg == "freq":
                return list1[:100], [i + 2000 for i in list2[:100]]
            if arg == "iter_freq":
                start = count // 500 * 50
                return list1[start:start + 50], [i + 2000 for i in list2[start:start + 50]]

    elif arg in ["dist", "iter_dist"]:
        with torch.no_grad():
            e1 = model.bert.embeddings.word_embeddings.weight[:2000]
            e2 = model.bert.embeddings.word_embeddings.weight[2000:]
            e1 /= torch.norm(e1, dim=-1, keepdim=True)
            e2 /= torch.norm(e2, dim=-1, keepdim=True)
            ssm = 1 - e1 @ e2.T
            nns = torch.argmin(ssm, dim=-1)
            dist = ssm[torch.arange(len(e1)), nns]
            dist_list_1 = torch.argsort(dist).tolist()
            dist_list_2 = nns[dist_list_1].tolist()
            lists = [(i, j) for i, j in zip(dist_list_1, dist_list_2) if
                     i not in [0, 1, 2, 3, 4] and j not in [0, 1, 2, 3, 4]]
            list1, list2 = zip(*lists)
            if arg == 'dist':
                return list(list1[:100]), [i + 2000 for i in list2[:100]]
            if arg == 'iter_dist':
                start = count // 500 * 50
                return list(list1[start:start + 50]), [ i + 2000 for i in list2[start:start + 50]]

    return list1, list2


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # global cnt, align_list1, align_list2, op
        word_loss = torch.nn.MSELoss(reduction='sum')
        outputs = model(**inputs)
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # if align_list1:
        if self.args.align_list1:
            new_loss = 0.5 * word_loss(
                model.bert.embeddings.word_embeddings(
                    torch.tensor([self.args.align_list1], dtype=torch.long).to(self.args.device)),
                model.bert.embeddings.word_embeddings(
                    torch.tensor([self.args.align_list2], dtype=torch.long).to(self.args.device)),
            )
            loss = loss + new_loss
            self.args.improve_cnt += 1
            if self.args.improve_cnt >= 500 and self.args.improve_cnt % 500 == 0 and len(self.args.align_list1) == 50:
                self.args.align_list1, self.args.align_list2 = get_list(self.args.improve_op, model, self.args.improve_cnt)
        return (loss, outputs) if return_outputs else loss


def get_dataset(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
        cache_dir: Optional[str] = None,
):
    def _dataset(file_path, ref_path=None):

        return LoadTextDataset(
            file_path=file_path,
        )

    if evaluate:
        return _dataset(args.eval_data_file, args.eval_ref_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file, args.train_ref_file)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # with open("test_args.bin", 'wb') as f:
    #     torch.save((model_args, data_args, training_args), f)
    # # return 0
    # model_args, data_args, training_args = torch.load('test_args.bin')

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.max_position_embeddings = data_args.block_size
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, config=config)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    config.vocab_size = len(tokenizer)

    sorted_checkpoints = _sorted_checkpoints_from_path(model_args.load_model_from)
    if len(sorted_checkpoints) == 0:
        raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
    else:
        model_args.model_name_or_path = os.path.join(sorted_checkpoints[-1], "pytorch_model.bin")

    model = AutoModelWithLMHead.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if len(tokenizer) != 4000:
        raise ValueError("we can only use improve list for same vocab")

    # if not model_args.improve_list == "control":
    #     global align_list1, align_list2, op
    #     op = model_args.improve_list
    #     align_list1, align_list2 = get_list(op, model)
    #     logger.info(align_list1, align_list2.tolist())

    if data_args.block_size <= 0:
        raise ValueError("set --block_size")
        # Our input block size will be the max possible for the model
    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )

    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )

    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        if data_args.mlm and data_args.whole_word_mask:
            data_collator = DataCollatorForWholeWordMask(
                tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
            )

    model.train()
    # Initialize our Trainer
    training_args.improve_op = model_args.improve_list
    align_list1, align_list2 = get_list(training_args.improve_op, model)
    training_args.align_list1 = None if model_args.improve_list == "control" else align_list1
    training_args.align_list2 = None if model_args.improve_list == "control" else align_list2
    training_args.improve_cnt = 0
    logger.info("{}\n{}".format(align_list1, align_list2))

    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        logger.info("Saving model checkpoint to %s", training_args.output_dir)
        trainer.save_model()

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)

        tokenizer.save_pretrained(training_args.output_dir)
        torch.save((model_args, data_args, training_args),
                   os.path.join(training_args.output_dir, "training_args.bin"))
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
