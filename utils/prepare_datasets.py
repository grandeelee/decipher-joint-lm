import logging
import random
from typing import List, Set, Optional
from copy import deepcopy
import pickle
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from custom_dataset import LoadTextDataset, save_line_by_line_dataset, save_retrieval_dataset

logger = logging.getLogger(__name__)


def shift_example(original: List[int], do_not_shift: Set[int], shift: int):
    tmp = []
    for x in original:
        if x in do_not_shift:
            tmp.append(x)
        else:
            tmp.append(x + shift)
    return tmp


def shifted_input(original: List[List[int]], do_not_shift: Set[int], shift: int) -> None:
    for i in range(len(original)):
        original[i] = [x if x in do_not_shift else x + shift for x in original[i]]


def invert_dataset(original: List[List[int]]) -> None:
    for i in range(len(original)):
        temp = original[i][1:-1]
        original[i][1:-1] = temp[::-1]


def random_dataset(original: List[List[int]]) -> None:
    for i in range(len(original)):
        temp = original[i][1:-1]
        random.shuffle(temp)
        original[i][1:-1] = temp


def single_dataset():
    # ######################################################
    # ######### single datasets ############################
    # ######################################################
    # prepare using 2000 tokenizer for all corpora
    vocab = "2000"
    config = AutoConfig.from_pretrained("../configs/bert-tiny.json")
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer/{}".format(vocab), config=config)
    corpus_list = ["NT", "OT", "TED", "WIKI"]
    for corpus in corpus_list:
        corpus_path = "../corpora/{}/train.txt".format(corpus)
        save_line_by_line_dataset(tokenizer=tokenizer,
                                  file_path=corpus_path,
                                  block_size=128,
                                  save_name="{}_{}".format(vocab, corpus),
                                  cache_dir="../data")
        print(corpus_path)
        corpus_path = "../corpora/{}/valid.txt".format(corpus)
        save_line_by_line_dataset(tokenizer=tokenizer,
                                  file_path=corpus_path,
                                  block_size=128,
                                  save_name="{}_{}".format(vocab, corpus),
                                  cache_dir="../data")
        print(corpus_path)
    # prepare using other tokenizer for NT corpus only
    for vocab in ["102", "500", "1000", "4000"]:
        tokenizer = AutoTokenizer.from_pretrained("../tokenizer/{}".format(vocab), config=config)
        corpus = "NT"
        corpus_path = "../corpora/{}/train.txt".format(corpus)
        save_line_by_line_dataset(tokenizer=tokenizer,
                                  file_path=corpus_path,
                                  block_size=128,
                                  save_name="{}_{}".format(vocab, corpus),
                                  cache_dir="../data")
        print(corpus_path)
        corpus_path = "../corpora/{}/valid.txt".format(corpus)
        save_line_by_line_dataset(tokenizer=tokenizer,
                                  file_path=corpus_path,
                                  block_size=128,
                                  save_name="{}_{}".format(vocab, corpus),
                                  cache_dir="../data")
        print(corpus_path)
    # prepare using 2000 tokenizer for INV corpora
    NT_2000_train = LoadTextDataset("../data/cached_2000_NT_train.txt")
    NT_2000_valid = LoadTextDataset("../data/cached_2000_NT_valid.txt")
    INV_2000_train = NT_2000_train.examples
    INV_2000_valid = NT_2000_valid.examples
    invert_dataset(INV_2000_train)
    invert_dataset(INV_2000_valid)
    save_path = "2000_INV"
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(INV_2000_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(INV_2000_valid, f, protocol=pickle.HIGHEST_PROTOCOL)

    # prepare using 2000 tokenizer for RND corpora
    NT_2000_train = LoadTextDataset("../data/cached_2000_NT_train.txt")
    NT_2000_valid = LoadTextDataset("../data/cached_2000_NT_valid.txt")
    RND_2000_train = NT_2000_train.examples
    RND_2000_valid = NT_2000_valid.examples
    random_dataset(RND_2000_train)
    random_dataset(RND_2000_valid)
    save_path = "2000_RND"
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(RND_2000_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(RND_2000_valid, f, protocol=pickle.HIGHEST_PROTOCOL)


def retrieval_dataset():
    # ##########################################
    # ######### retrieval datasets #############
    # ##########################################
    config = AutoConfig.from_pretrained("../configs/bert-tiny.json")
    corpus_path = "../corpora/1K/retrieval.txt"
    for vocab in ["500", "1000", "2000", "4000"]:
        tokenizer = AutoTokenizer.from_pretrained("../tokenizer/{}".format(vocab), config=config)
        save_retrieval_dataset(tokenizer=tokenizer,
                               file_path=corpus_path,
                               block_size=128,
                               save_name="{}".format(vocab),
                               cache_dir="../data")
        print(corpus_path)

    with open("../data/cached_2000_retrieval.txt", "rb") as f:
        ret_2000 = pickle.load(f)
    with open("../data/cached_500_retrieval.txt", "rb") as f:
        ret_500 = pickle.load(f)
    with open("../data/cached_1000_retrieval.txt", "rb") as f:
        ret_1000 = pickle.load(f)
    with open("../data/cached_4000_retrieval.txt", "rb") as f:
        ret_4000 = pickle.load(f)

    # 2000_500
    save_path = "2000_500"
    with open("../data/cached_2000_retrieval.txt", "rb") as f:
        data = pickle.load(f)
    input_ids = ret_500["input_ids"]
    shifted_input(input_ids, tokenizer.all_special_ids, 2000)
    data["input_ids"] += input_ids
    data["word_ids"] += ret_500["word_ids"]
    with open("../data/cached_{}_retrieval.txt".format(save_path), "wb") as f:
        pickle.dump(data, f)
    print(save_path)

    # 2000_1000
    save_path = "2000_1000"
    with open("../data/cached_2000_retrieval.txt", "rb") as f:
        data = pickle.load(f)
    input_ids = ret_1000["input_ids"]
    shifted_input(input_ids, tokenizer.all_special_ids, 2000)
    data["input_ids"] += input_ids
    data["word_ids"] += ret_1000["word_ids"]
    with open("../data/cached_{}_retrieval.txt".format(save_path), "wb") as f:
        pickle.dump(data, f)
    print(save_path)

    # 2000_2000
    save_path = "2000_2000"
    with open("../data/cached_2000_retrieval.txt", "rb") as f:
        data = pickle.load(f)
    input_ids = ret_2000["input_ids"]
    shifted_input(input_ids, tokenizer.all_special_ids, 2000)
    data["input_ids"] += input_ids
    data["word_ids"] += ret_2000["word_ids"]
    with open("../data/cached_{}_retrieval.txt".format(save_path), "wb") as f:
        pickle.dump(data, f)
    print(save_path)

    # 2000_4000
    save_path = "2000_4000"
    with open("../data/cached_2000_retrieval.txt", "rb") as f:
        data = pickle.load(f)
    input_ids = ret_4000["input_ids"]
    shifted_input(input_ids, tokenizer.all_special_ids, 2000)
    data["input_ids"] += input_ids
    data["word_ids"] += ret_4000["word_ids"]
    with open("../data/cached_{}_retrieval.txt".format(save_path), "wb") as f:
        pickle.dump(data, f)
    print(save_path)


def bilingual_dataset():
    # ##########################################
    # ######### 'bilingual' datasets ###########
    # ##########################################
    config = AutoConfig.from_pretrained("../configs/bert-tiny.json")
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer/2000", config=config)

    NT_2000_train = LoadTextDataset("../data/cached_2000_NT_train.txt")
    NT_2000_valid = LoadTextDataset("../data/cached_2000_NT_valid.txt")
    NT_500_train = LoadTextDataset("../data/cached_500_NT_train.txt")
    NT_500_valid = LoadTextDataset("../data/cached_500_NT_valid.txt")
    NT_1000_train = LoadTextDataset("../data/cached_1000_NT_train.txt")
    NT_1000_valid = LoadTextDataset("../data/cached_1000_NT_valid.txt")
    NT_4000_train = LoadTextDataset("../data/cached_4000_NT_train.txt")
    NT_4000_valid = LoadTextDataset("../data/cached_4000_NT_valid.txt")
    OT_2000_train = LoadTextDataset("../data/cached_2000_OT_train.txt")
    OT_2000_valid = LoadTextDataset("../data/cached_2000_OT_valid.txt")
    TED_2000_train = LoadTextDataset("../data/cached_2000_TED_train.txt")
    TED_2000_valid = LoadTextDataset("../data/cached_2000_TED_valid.txt")
    WIKI_2000_train = LoadTextDataset("../data/cached_2000_WIKI_train.txt")
    WIKI_2000_valid = LoadTextDataset("../data/cached_2000_WIKI_valid.txt")
    INV_2000_train = LoadTextDataset("../data/cached_2000_INV_train.txt")
    INV_2000_valid = LoadTextDataset("../data/cached_2000_INV_valid.txt")
    RND_2000_train = LoadTextDataset("../data/cached_2000_RND_train.txt")
    RND_2000_valid = LoadTextDataset("../data/cached_2000_RND_valid.txt")

    # 2000_NT_2000_NT
    save_path = "2000_NT_2000_NT"
    shifted_train = deepcopy(NT_2000_train.examples)
    shifted_valid = deepcopy(NT_2000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_2000_OT
    save_path = "2000_NT_2000_OT"
    shifted_train = deepcopy(OT_2000_train.examples)
    shifted_valid = deepcopy(OT_2000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_2000_TED
    save_path = "2000_NT_2000_TED"
    shifted_train = deepcopy(TED_2000_train.examples)
    shifted_valid = deepcopy(TED_2000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_2000_WIKI
    save_path = "2000_NT_2000_WIKI"
    shifted_train = deepcopy(WIKI_2000_train.examples)
    shifted_valid = deepcopy(WIKI_2000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_2000_INV
    save_path = "2000_NT_2000_INV"
    shifted_train = deepcopy(INV_2000_train.examples)
    shifted_valid = deepcopy(INV_2000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_2000_RND
    save_path = "2000_NT_2000_RND"
    shifted_train = deepcopy(RND_2000_train.examples)
    shifted_valid = deepcopy(RND_2000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_500_NT
    save_path = "2000_NT_500_NT"
    shifted_train = deepcopy(NT_500_train.examples)
    shifted_valid = deepcopy(NT_500_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_1000_NT
    save_path = "2000_NT_1000_NT"
    shifted_train = deepcopy(NT_1000_train.examples)
    shifted_valid = deepcopy(NT_1000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

    # 2000_NT_4000_NT
    save_path = "2000_NT_4000_NT"
    shifted_train = deepcopy(NT_4000_train.examples)
    shifted_valid = deepcopy(NT_4000_valid.examples)
    shifted_input(shifted_train, tokenizer.all_special_ids, 2000)
    shifted_input(shifted_valid, tokenizer.all_special_ids, 2000)
    train = NT_2000_train.examples + shifted_train
    valid = NT_2000_valid.examples + shifted_valid
    with open("../data/cached_{}_train.txt".format(save_path), "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../data/cached_{}_valid.txt".format(save_path), "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(save_path)

if __name__ == "__main__":
    single_dataset()
    retrieval_dataset()
    bilingual_dataset()
