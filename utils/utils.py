import os
import re
import glob
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
from collections import Counter, OrderedDict


def load_embedding_model(model):
    embedding_config = AutoConfig.from_pretrained(model)
    embedding_config.output_hidden_states = True
    embedding_model = AutoModelWithLMHead.from_pretrained(model, config=embedding_config)
    embedding_model.eval()
    # embedding_model.to('cuda')
    embedding_tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False, config=embedding_config)
    return embedding_model, embedding_tokenizer, embedding_config


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def get_unigram_from_tokenized(tokenized_list):
    """
    :param tokenized_list: the input is a list or a list of lists
    :return: a dictionary in descending order of word frequency
    """
    cnt = Counter()
    if isinstance(tokenized_list[0], list):
        # this is a list of list
        for line in tokenized_list:
            cnt.update(line)
    else:
        if isinstance(tokenized_list, list):
            # this is a list
            cnt.update(tokenized_list)
    # sort counts into descending order
    ordered_counts = OrderedDict(sorted(cnt.items(), key=lambda item: (-item[1], item[0])))
    return ordered_counts


def get_vocab_from_tokenized(tokenized_list):
    """
    :param tokenized_list: the input is a list or a list of lists
    :return: word2id, id2word
    """
    ordered_counts = get_unigram_from_tokenized(tokenized_list)
    word2id = {j: i for i, j in enumerate(ordered_counts.keys())}
    id2word = {i: j for i, j in enumerate(ordered_counts.keys())}

    return word2id, id2word


def _sorted_checkpoints_from_path(path, checkpoint_prefix="checkpoint"):
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(path, "{}-*".format(checkpoint_prefix)))

    for checkpoint in glob_checkpoints:
        regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), checkpoint)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), checkpoint))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted
