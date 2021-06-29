import os
import collections
import math
import argparse
import json
import numpy as np
import re
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig, Trainer, \
    DataCollatorForLanguageModeling, BertLMHeadModel
from utils.custom_dataset import LoadTextDataset, LoadRetrievalDataset
from utils.utils import _sorted_checkpoints_from_path

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval_retrieval",
    default="data/cached_2000_2000_retrieval.txt",
    type=str,
    help="",
)
parser.add_argument(
    "--eval_perplexity",
    default="data/cached_2000_NT_2000_NT_valid.txt",
    type=str,
    help="",
)
parser.add_argument(
    "--model_name_or_path",
    default='models/cached_2000_NT_2000_NT_clm_0',
    type=str,
    help="The model checkpoint for weights initialization.",
)
parser.add_argument(
    "--config_name",
    default="configs/bert-tiny.json",
    type=str,
    help="The config for loading tokenizer and model.",
)
parser.add_argument(
    "--tokenizer_name",
    default="tokenizer/2000_2000",
    type=str,
    help="The tokenizer path for loading tokenizer and model.",
)
parser.add_argument(
    "--block_size",
    default=128,
    type=int,
    help="Optional input sequence length after tokenization."
         "The training dataset will be truncated in block of this size for training."
         "Default to the model max input length for single sentence inputs (take into account special tokens).",
)
parser.add_argument(
    "--clm",
    action="store_true",
    help="Optional input sequence length after tokenization."
         "The training dataset will be truncated in block of this size for training."
         "Default to the model max input length for single sentence inputs (take into account special tokens).",
)

parser.add_argument(
    "--mlm",
    action="store_true",
    help="Optional input sequence length after tokenization."
         "The training dataset will be truncated in block of this size for training."
         "Default to the model max input length for single sentence inputs (take into account special tokens).",
)

parser.add_argument("--eval_layers", default="0,4,8,11", type=str, help="")

args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = 0 if not torch.cuda.is_available() else torch.cuda.device_count()
args.eval_layers = [int(x) for x in args.eval_layers.split(",")]

config = AutoConfig.from_pretrained(args.config_name)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, config=config)

config.vocab_size = len(tokenizer)
config.max_position_embeddings = args.block_size

with open(os.path.join(args.tokenizer_name, "vocab.txt"), 'r') as f:
    vocab = f.read().split()
vocab_1 = []
vocab_2 = []
for word in vocab:
    if not word.startswith("::"):
        vocab_1.append(word)
    else:
        vocab_2.append(re.sub(r"::", "", word))

word_mask = []
alignment_gt = []
for word in vocab_1:
    if word.startswith("##"):
        word_mask.append(False)
    else:
        try:
            index = vocab_2.index(word)
            word_mask.append(True)
            alignment_gt.append(index)
        except ValueError:
            word_mask.append(False)
            logger.info("drop a word in bli")


def collate(examples):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)


def get_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X /= np.linalg.norm(X, axis=-1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=-1, keepdims=True)
    cosine_similarity = X @ Y.T
    cosine_distance = 1 - cosine_similarity
    return cosine_distance


def evaluate_retrieval(vectors):
    n = len(vectors)
    if n % 2 != 0:
        raise ValueError("Something's wrong.")
    vectors_e, vectors_f = vectors[:n // 2], vectors[n // 2:]
    vectors_e = np.array([x.cpu().numpy().reshape(-1, 64).mean(axis=0) for x in vectors_e])
    vectors_f = np.array([x.cpu().numpy().reshape(-1, 64).mean(axis=0) for x in vectors_f])
    dist = get_distances(vectors_e, vectors_f)
    if dist.shape[0] != dist.shape[1]:
        print("Number of sentences is different?")
    # get different p@k
    nns = np.argsort(dist, axis=1)[:, :10]
    gt = np.arange(dist.shape[0]).reshape(-1, 1)
    p = {}
    for considern in [1, 5, 10]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        p["P@{}".format(considern)] = hits1 / dist.shape[0]
    nns = np.argsort(dist, axis=0)[:10, :].transpose()
    gt = np.arange(dist.shape[0]).reshape(-1, 1)
    pinv = {}
    for considern in [1, 5, 10]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        pinv["P@{}".format(considern)] = hits1 / dist.shape[0]
    return {"forward": p, "backward": pinv}


def evaluate_alignment(vectors, dataset, do_ave=True):
    # for each sentence pair get vectors and alignments (using NNs)
    v_n = len(vectors)
    if v_n % 2 != 0:
        raise ValueError("Somethings wrong.")
    vectors_e, vectors_f = vectors[:v_n // 2], vectors[v_n // 2:]
    all_predict = collections.defaultdict(list)
    all_trues = []

    for idx, (e, f) in enumerate(zip(vectors_e, vectors_f)):
        e, f = e.cpu().numpy().reshape(-1, 64)[1:-1, :], f.cpu().numpy().reshape(-1, 64)[1:-1, :]
        true_align_e = dataset.word_ids[idx][1:-1]
        true_align_f = dataset.word_ids[(v_n // 2) + idx][1:-1]
        if max(true_align_e) != max(true_align_f):
            # raise ValueError("alignment label different length", idx)
            logger.info("skipping one example")
            continue
        e_index = np.array([true_align_f.index(i) for i in true_align_e])
        f_index = np.array([true_align_e.index(i) for i in true_align_f])

        if min(min(e.shape), min(f.shape)) == 0:
            raise ValueError("Empty sentence.")

        if not do_ave:
            dist = get_distances(e, f)
            forward = dist.argmin(axis=1) == e_index  # len(e)
            backward = dist.argmin(axis=0) == f_index  # len(f)
            intersect = np.append(forward, backward)
            # filter edges where all systems predicted 0
            all_predict["forward_no_avg"].extend(forward)
            all_predict["backward_no_avg"].extend(backward)
            all_predict["intersect_no_avg"].extend(intersect)
        else:
            ave_e = [e[0]]
            count = 1
            for i in range(1, len(e)):
                if true_align_e[i] == true_align_e[i - 1]:
                    ave_e[-1] += e[i]
                    count += 1
                else:
                    ave_e[-1] /= count
                    ave_e.append(e[i])
                    count = 1
            ave_f = [f[0]]
            count = 1
            for i in range(1, len(f)):
                if true_align_f[i] == true_align_f[i - 1]:
                    ave_f[-1] += f[i]
                    count += 1
                else:
                    ave_f[-1] /= count
                    ave_f.append(f[i])
                    count = 1
            dist = get_distances(np.array(ave_e), np.array(ave_f))
            m, n = dist.shape
            forward = np.eye(n)[dist.argmin(axis=1)]  # m x n
            backward = np.eye(m)[dist.argmin(axis=0)].T
            intersect = forward * backward
            if dist.shape[0] != dist.shape[1]:
                raise ValueError("Sentence length is different: {}.".format(i))
            gold = np.eye(dist.shape[0])
            # filter edges where all systems predicted 0
            non_zero = (forward.flatten() + backward.flatten() + intersect.flatten() + gold.flatten()) > 0
            all_predict["forward"].extend(list(forward.flatten()[non_zero]))
            all_predict["backward"].extend(list(backward.flatten()[non_zero]))
            all_predict["intersect"].extend(list(intersect.flatten()[non_zero]))
            all_trues.extend(list(gold.flatten()[non_zero]))

    result = {}
    for k, v in all_predict.items():
        acc = accuracy_score(all_trues, v)
        prec = precision_score(all_trues, v)
        rec = recall_score(all_trues, v)
        f1 = f1_score(all_trues, v)
        result[k] = {"acc": acc,
                     "prec": prec,
                     "rec": rec,
                     "f1": f1}

        return result


def evaluate_translation(model, n, word_mask=None, gt=None):
    emb = model.base_model.embeddings.word_embeddings.weight.data.cpu().detach().numpy()
    emb_e, emb_f = emb[:n], emb[n:]
    dist = get_distances(emb_e, emb_f)
    # get different p@k
    nns = np.argsort(dist, axis=1)[:, :1][word_mask]
    gt = np.array(gt).reshape(-1, 1)
    if len(nns) != len(gt):
        raise ValueError("nns not equals to gt in length")
    p = {}
    for considern in [1]:
        hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
        p["P@{}".format(considern)] = hits1 / dist.shape[0]
    # nns = np.argsort(dist, axis=0)[:10, :].transpose()
    # gt = np.arange(dist.shape[0]).reshape(-1, 1)
    # pinv = {}
    # for considern in [1]:
    #     hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
    #     pinv["P@{}".format(considern)] = hits1 / dist.shape[0]
    return {"forward": p}


def run_model(path):
    logger.info("loading model from {}".format(path))
    config.output_hidden_states = True
    if args.clm:
        config.is_decoder = True
        model = BertLMHeadModel.from_pretrained(
            path,
            config=config
        )
    else:
        model = AutoModelWithLMHead.from_pretrained(
            path,
            config=config
        )
    model.eval()
    model.to(args.device)

    eval_dataset = LoadTextDataset(
        file_path=args.eval_perplexity,
    )

    retrieval_dataset = LoadRetrievalDataset(
        file_path=args.eval_retrieval,
    )

    eval_sampler = SequentialSampler(retrieval_dataset)
    eval_dataloader = DataLoader(
        retrieval_dataset, sampler=eval_sampler, batch_size=1, collate_fn=collate
    )

    all_vectors = collections.defaultdict(list)
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(batch.to(args.device))
            for layer in args.eval_layers:
                try:
                    all_vectors[layer].extend(outputs["hidden_states"][layer])
                except IndexError:
                    print("{},{}".format(layer, len(outputs[1])))

    eval_results = {}
    for layer in args.eval_layers:
        retrieval_results = evaluate_retrieval(all_vectors[layer])
        alignment_results = evaluate_alignment(all_vectors[layer], retrieval_dataset, do_ave=True)
        translation_results = evaluate_translation(model, len(vocab_1), word_mask, alignment_gt)
        eval_results["layer{}".format(layer)] = {
            "retrieval_results": retrieval_results,
            "alignment_results": alignment_results,
            "translation_results": translation_results,
        }
    ############# PPL ################
    config.output_hidden_states = False
    if args.clm:
        config.is_decoder = True
        model = BertLMHeadModel.from_pretrained(
            path,
            config=config
        )
    else:
        model = AutoModelWithLMHead.from_pretrained(
            path,
            config=config
        )

    model.eval()
    model.to(args.device)

    if args.clm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, mlm_probability=0.15
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
    )

    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    eval_results["perplexity"] = perplexity

    directory, filename = os.path.split(path)
    save_path = os.path.join(directory, "results.json")
    logger.info("saving to {}".format(save_path))
    with open(save_path, 'w') as f:
        json.dump(eval_results, f)


def main():
    path_list = _sorted_checkpoints_from_path(args.model_name_or_path)
    for model_path in path_list:
        run_model(os.path.join(model_path, "pytorch_model.bin"))


if __name__ == "__main__":
    main()
