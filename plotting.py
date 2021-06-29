import glob
import os
import re
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
from utils.utils import get_unigram_from_tokenized, _sorted_checkpoints_from_path
import pickle
import torch
from transformers import AutoModelWithLMHead, AutoConfig
import nltk
from collections import defaultdict


def get_distances(X: np.ndarray, Y: np.ndarray, norm: bool = True) -> np.ndarray:
    if norm:
        X /= np.linalg.norm(X, axis=-1, keepdims=True)
        Y /= np.linalg.norm(Y, axis=-1, keepdims=True)
    cosine_similarity = X @ Y.T
    cosine_distance = 1 - cosine_similarity
    return cosine_distance


def evaluate_translation(model, n):
    emb = model.base_model.embeddings.word_embeddings.weight.data.cpu().detach().numpy()
    emb_e, emb_f = emb[:n], emb[n:]
    dist = get_distances(emb_e, emb_f)
    # get different p@k
    nns = np.argsort(dist, axis=1)[:, :1]
    gt = np.arange(len(nns)).reshape(-1, 1)
    if len(nns) != len(gt):
        raise ValueError("nns not equals to gt in length")

    for considern in [1]:
        hits1 = (nns[:, :considern] == gt).sum(axis=1) > 0

    return hits1, dist.diagonal()


def return_translation_indices(model, n):
    emb = model.base_model.embeddings.word_embeddings.weight.data.cpu().detach().numpy()
    emb_e, emb_f = emb[:n], emb[n:]
    dist = get_distances(emb_e, emb_f)
    # get different p@k
    nns = np.argsort(dist, axis=1)[:, :1]

    return nns.reshape(-1)


def get_unigram_from_path(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return get_unigram_from_tokenized(data)


def get_final_checkpoint_result(path):
    checkpoints = _sorted_checkpoints_from_path(path)
    final_checkpoint = checkpoints[-1]
    result_path = os.path.join(final_checkpoint, "results.json")
    with open(result_path, 'r') as f:
        result = json.load(f)
    return result


def get_checkpoint_result(checkpoint_path):
    result_path = os.path.join(checkpoint_path, "results.json")
    with open(result_path, 'r') as f:
        result = json.load(f)
    return result


def get_average_from_result(result, is_clm=False):
    layers = ['layer0', 'layer4', "layer8", 'layer12' if not is_clm else "layer11"]
    retrieval_results, alignment_results, translation_results = [], [], []
    for layer in layers:
        retrieval_results.append(result[layer]["retrieval_results"]["forward"]['P@1'])
        alignment_results.append(result[layer]["alignment_results"]['forward']['f1'])
        translation_results.append(result[layer]["translation_results"]["forward"]['P@1'])
    ave_retrieval = sum(retrieval_results) / len(retrieval_results)
    ave_alignment = sum(alignment_results) / len(alignment_results)
    ave_translation = sum(translation_results) / len(translation_results)
    return ave_retrieval, ave_alignment, ave_translation


def plot_clm_mlm_score():
    clm_paths = ["cached_lm_ERV_NEW_2000_ERV_NEW_126_clm", "cached_lm_ERV_NEW_2000_ERV_OLD_126_clm",
                 "cached_lm_ERV_NEW_2000_TED2013_126_clm", "cached_lm_ERV_NEW_2000_wiki_simple_sents_126_clm",
                 "cached_lm_ERV_NEW_2000_INV_126_clm", "cached_lm_ERV_NEW_2000_RND_126_clm", ]
    # "cached_lm_ERV_NEW_2000ERV_NEW_500_ERV_NEW_126_clm",
    # "cached_lm_ERV_NEW_2000ERV_NEW_1000_ERV_NEW_126_clm",
    # "cached_lm_ERV_NEW_2000ERV_NEW_4000_ERV_NEW_126_clm"]
    mlm_paths = ["cached_lm_ERV_NEW_2000_ERV_NEW_126", "cached_lm_ERV_NEW_2000_ERV_OLD_126",
                 "cached_lm_ERV_NEW_2000_TED2013_126", "cached_lm_ERV_NEW_2000_wiki_simple_sents_126",
                 "cached_lm_ERV_NEW_2000_INV_126", "cached_lm_ERV_NEW_2000_RND_126", ]
    # "cached_lm_ERV_NEW_2000ERV_NEW_500_ERV_NEW_126",
    # "cached_lm_ERV_NEW_2000ERV_NEW_1000_ERV_NEW_126",
    # "cached_lm_ERV_NEW_2000ERV_NEW_4000_ERV_NEW_126"]
    clm_results, clm_retrieval, clm_alignment, clm_translation = [], [], [], []
    for path in clm_paths:
        result_path = os.path.join("models", path)
        result = get_final_checkpoint_result(result_path)
        ave_results = get_average_from_result(result, is_clm=True)
        clm_results.append(sum(ave_results) / len(ave_results))
        clm_retrieval.append(ave_results[0])
        clm_alignment.append(ave_results[1])
        clm_translation.append(ave_results[2])
    mlm_results, mlm_retrieval, mlm_alignment, mlm_translation = [], [], [], []
    for path in mlm_paths:
        result_path = os.path.join("models", path)
        result = get_final_checkpoint_result(result_path)
        ave_results = get_average_from_result(result, is_clm=False)
        mlm_results.append(sum(ave_results) / len(ave_results))
        mlm_retrieval.append(ave_results[0])
        mlm_alignment.append(ave_results[1])
        mlm_translation.append(ave_results[2])
    labels = ['new', 'old', 'ted', 'wiki', 'inv', 'rand']  # , "500", "1K", "4K"

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    mpl.rcParams.update({'font.size': 18})
    # plt.style.use("seaborn-whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    ax1.bar(x - width / 2, clm_results, width, label='CLM', color='cornflowerblue')
    ax1.bar(x + width / 2, mlm_results, width, label='MLM', color='orange')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_ylabel('Multilingual Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(title='Average')
    ax1.grid()

    ax2.bar(x - width / 2, clm_retrieval, width, label='CLM', color='cornflowerblue')
    ax2.bar(x + width / 2, mlm_retrieval, width, label='MLM', color='orange')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax2.set_ylabel('f1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(title='Retrieval (f1)')
    ax2.grid()

    ax3.grid()
    ax3.bar(x - width / 2, clm_alignment, width, label='CLM', color='cornflowerblue')
    ax3.bar(x + width / 2, mlm_alignment, width, label='MLM', color='orange')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax3.set_ylabel('P@1')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend(title='Alignment')

    ax4.bar(x - width / 2, clm_translation, width, label='CLM', color='cornflowerblue')
    ax4.bar(x + width / 2, mlm_translation, width, label='MLM', color='orange')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax4.set_ylabel('P@1')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend(title='Translation')
    ax4.grid()

    fig.tight_layout()
    plt.savefig('plots/clm_vs_mlm_multilingual_score.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure


def plot_clm_mlm_key():
    with open("/home/grandee/projects/joint_align/clm_mlm.result.clm.json") as f:
        results_clm = json.load(f)
    with open("/home/grandee/projects/joint_align/clm_mlm.result.mlm.json") as f:
        results_mlm = json.load(f)
    sorted_mlm = np.array(results_mlm)
    alternate_mlm = np.append(sorted_mlm[np.arange(39, 0, -2)], sorted_mlm[np.arange(0, 40, 2)])
    sorted_clm = np.array(results_clm)
    alternate_clm = np.append(sorted_clm[np.arange(39, 0, -2)], sorted_clm[np.arange(0, 40, 2)])

    mpl.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(7, 4))
    axe = fig.add_subplot()
    axe.set_xlabel("No. of permutations performed on the key")
    axe.set_ylabel("Perplexity")
    axe.set_yscale("log")
    axe.set_xticklabels(['', '40', '30', '20', '10', '0', '10', '20', '30', '40'])

    axe.plot(alternate_clm, 'o--', label='CLM', color='cornflowerblue', alpha=0.8)
    axe.plot(alternate_mlm, 'o--', label="MLM", color='orange', alpha=0.8)

    axe.grid(axis='y', color='0.7')
    axe.legend(title="CLM vs MLM")
    fig.tight_layout()
    plt.savefig("plots/clm_mlm.pdf", bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure


def plot_mlm_js_score():
    mlm_paths = ["cached_lm_ERV_NEW_2000_ERV_NEW_126", "cached_lm_ERV_NEW_2000ERV_NEW_4000_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000_ERV_OLD_126", "cached_lm_ERV_NEW_2000ERV_NEW_1000_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000_TED2013_126", "cached_lm_ERV_NEW_2000_wiki_simple_sents_126",
                 "cached_lm_ERV_NEW_2000ERV_NEW_500_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000_RND_126", "cached_lm_ERV_NEW_2000_INV_126",
                 ]
    mlm_results, mlm_retrieval, mlm_alignment, mlm_translation = [], [], [], []
    for path in mlm_paths:
        result_path = os.path.join("models", path)
        result = get_final_checkpoint_result(result_path)
        ave_results = get_average_from_result(result, is_clm=False)
        mlm_results.append(sum(ave_results) / len(ave_results))
        mlm_retrieval.append(ave_results[0])
        mlm_alignment.append(ave_results[1])
        mlm_translation.append(ave_results[2])
    labels = ['new', "4K", 'old', "1K", 'ted', 'wiki', "500", 'rand', 'inv']  #

    x = np.arange(len(labels))  # the label locations
    mpl.rcParams.update({'font.size': 18})
    # plt.style.use("seaborn-whitegrid")

    js_d = [0, 0.02037, 0.05992, 0.06336, 0.11081, 0.12362, 0.16668, 0, 0]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Datasets")
    host.set_xticks(x)
    host.set_xticklabels(labels)
    host.set_ylabel("Multilingual Score")
    par.set_ylabel("JS Divergence")

    host.plot(mlm_results, 'o--', label='Mscore', color='cornflowerblue', alpha=0.8)
    par.plot(js_d, 'o--', label="JS Divergence", color='orange', alpha=0.8)

    host.grid(axis='y', color='0.7')
    host.legend()

    plt.savefig('plots/mlm_vs_JS_multilingual_score.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure


def plot_mlm_ppl_score():
    mlm_paths = ["cached_lm_ERV_NEW_2000_ERV_NEW_126", "cached_lm_ERV_NEW_2000ERV_NEW_4000_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000_ERV_OLD_126", "cached_lm_ERV_NEW_2000ERV_NEW_1000_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000_TED2013_126", "cached_lm_ERV_NEW_2000_wiki_simple_sents_126",
                 "cached_lm_ERV_NEW_2000ERV_NEW_500_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000_RND_126", "cached_lm_ERV_NEW_2000_INV_126",
                 ]
    mlm_results, mlm_retrieval, mlm_alignment, mlm_translation = [], [], [], []
    for path in mlm_paths:
        result_path = os.path.join("models", path)
        result = get_final_checkpoint_result(result_path)
        ave_results = get_average_from_result(result, is_clm=False)
        mlm_results.append(sum(ave_results) / len(ave_results))
        mlm_retrieval.append(ave_results[0])
        mlm_alignment.append(ave_results[1])
        mlm_translation.append(ave_results[2])
    labels = ['new', "4K", 'old', "1K", 'ted', 'wiki', "500", 'rand', 'inv']  #

    x = np.arange(len(labels))  # the label locations
    mpl.rcParams.update({'font.size': 18})
    # plt.style.use("seaborn-whitegrid")

    js_d = [6.56193, 22.25534, 53.055726, 58.41452, 4251.97871, 7482.858137]
    js_x = [0, 2, 4, 5, 7, 8]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Datasets")
    host.set_xticks(x)
    host.set_xticklabels(labels)
    host.set_ylabel("Multilingual Score")
    par.set_yscale("log")
    par.set_ylabel("Perplexity")

    host.plot(x, mlm_results, 'o--', label='Mscore', color='cornflowerblue', alpha=0.8)
    par.plot(js_x, js_d, 'o--', label="Perplexity", color='orange', alpha=0.8)

    host.grid(axis='y', color='0.7')
    host.legend()

    plt.savefig('plots/mlm_vs_ppl_multilingual_score.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure


def plot_bli_freq_dist(model_path):
    unigram = get_unigram_from_path(
        "/home/grandee/projects/joint_align/corpora/ERV_NEW/cached_lm_ERV_NEW_2000_ERV_NEW_126_train.txt")
    word_order_1 = [k for k, v in unigram.items() if k < 2000 and k not in [0, 1, 2, 3, 4]]
    word_order_2 = [k - 2000 for k, v in unigram.items() if k >= 2000]
    freq_order = [i for i in word_order_1 if i in word_order_2]
    word_count = [unigram.get(i) for i in freq_order]
    word_count_norm = np.array(word_count) / sum(word_count)
    word_count_log = np.log(word_count_norm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained("/home/grandee/projects/joint_align/configs/bert-tiny.json")
    config.vocab_size = 4000
    config.max_position_embeddings = 128

    model = AutoModelWithLMHead.from_pretrained(
        model_path,
        config=config
    )
    model.eval()
    model.to(device)

    bli_result, dist = evaluate_translation(model, 2000)

    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Words")
    # host.set_xticks(freq_order)
    # host.set_xticklabels(labels)
    host.set_ylabel("Frequency")
    host.set_yscale("log")
    par.set_ylabel("Cosine Dist")

    host.plot(word_count_norm, '--', label='freq', color='cornflowerblue', alpha=0.8)
    par.plot(dist[freq_order], '--', label="dist", color='orange', alpha=0.8)

    host.grid(axis='y', color='0.7')
    host.legend()

    plt.savefig('plots/bli_pos_freq_1.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure

    #### arrange according to dist
    dist_order = np.argsort(dist[freq_order])

    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Words")
    # host.set_xticks(freq_order)
    # host.set_xticklabels(labels)
    host.set_ylabel("Frequency")
    host.set_yscale("log")
    par.set_ylabel("Cosine Dist")

    host.plot(word_count_norm[dist_order], '--', label='freq', color='cornflowerblue', alpha=0.8)
    par.plot(dist[freq_order][dist_order], '--', label="dist", color='orange', alpha=0.8)

    host.grid(axis='y', color='0.7')
    host.legend()

    plt.savefig('plots/bli_pos_freq_2.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure

    ############# bli vs freq ###############
    bli_label = bli_result[freq_order]
    bli_true = np.arange(len(freq_order))[bli_label]
    bli_false = np.arange(len(freq_order))[~bli_label]

    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Words")
    # host.set_xticks(freq_order)
    # host.set_xticklabels(labels)
    host.set_ylabel("Frequency")
    host.set_yscale("log")
    par.set_ylabel("Cosine Dist")

    host.plot(word_count_norm, '--', label='freq', color='cornflowerblue', alpha=1)
    par.hist(bli_true, 50, label='hits', density=False, color='orange', alpha=0.8)
    par.hist(bli_false, 50, label='misses', density=False, color='green', alpha=0.8)

    host.grid(axis='y', color='0.7')
    host.legend()

    plt.savefig('plots/bli_pos_freq_3.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure

    ####### correlation dist freq #######
    _, ax = plt.subplots()
    ax.scatter(word_count_norm[bli_true], dist[freq_order][bli_true], c='tab:blue', label='True',
               alpha=0.3, edgecolors='none')
    ax.scatter(word_count_norm[bli_false], dist[freq_order][bli_false], c='tab:orange', label='False',
               alpha=0.3, edgecolors='none')
    ax.set_xscale("log")
    ax.set_ylabel("dist")
    ax.set_xlabel("freq")
    ax.legend()
    ax.grid(True)
    plt.savefig('plots/bli_pos_freq_4.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()
    ####### contour dist freq bli#######
    # x = word_count_log
    # y = dist[freq_order]
    # z = np.array([1 if i else 0 for i in bli_label])
    # fig, ax2 = plt.subplots()
    # ax2.tricontour(x, y, z, levels=2, linewidths=0.5, colors='k')
    # cntr2 = ax2.tricontourf(x, y, z, levels=2, cmap="RdBu_r")
    #
    # fig.colorbar(cntr2, ax=ax2)
    # ax2.plot(x, y, 'ko', ms=3)
    #
    # plt.subplots_adjust(hspace=0.5)
    # plt.savefig('plots/bli_pos_freq_6.pdf', bbox_inches='tight')
    # plt.show()

    #### bli arrange according to dist
    dist_order = np.argsort(dist[freq_order])
    bli_label = bli_result[freq_order][dist_order]
    bli_true = np.arange(len(freq_order))[bli_label]
    bli_false = np.arange(len(freq_order))[~bli_label]

    host = host_subplot(111)
    par = host.twinx()

    host.set_xlabel("Words")
    # host.set_xticks(freq_order)
    # host.set_xticklabels(labels)
    host.set_ylabel("Frequency")
    host.set_yscale("log")
    par.set_ylabel("Cosine Dist")

    host.plot(dist[freq_order][dist_order], '--', label='dist', color='cornflowerblue', alpha=1)
    par.hist(bli_true, 50, label='hits', density=False, color='orange', alpha=0.8)
    par.hist(bli_false, 50, label='misses', density=False, color='green', alpha=0.8)

    host.grid(axis='y', color='0.7')
    host.legend()

    plt.savefig('plots/bli_pos_freq_5.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # Clear figure


def plot_pos_freq(model_path):
    config = AutoConfig.from_pretrained("/home/grandee/projects/joint_align/configs/bert-tiny.json")
    config.vocab_size = 4000
    config.max_position_embeddings = 128

    model = AutoModelWithLMHead.from_pretrained(
        model_path,
        config=config
    )
    model.eval()
    bli_result, _ = evaluate_translation(model, 2000)
    with open("/home/grandee/projects/joint_align/vocab/ERV_NEW_2000/vocab.txt", 'r') as f:
        text = f.read().split()
    correct_vocab = np.array(text)[bli_result]
    total_word_dict = {}
    for i, word in enumerate(text):
        if not word.startswith("##"):
            total_word_dict[i] = word
    correct_word_dict = {}
    for word in correct_vocab:
        if not word.startswith("##"):
            correct_word_dict[text.index(word)] = word
    total_pos_dict = {}
    for i, word in total_word_dict.items():
        total_pos_dict[i] = nltk.pos_tag([word])[0][1]

    aligned_pos_dict = {}
    for word in correct_word_dict.values():
        aligned_pos_dict[text.index(word)] = nltk.pos_tag([word])[0][1]

    index_freq = get_unigram_from_path(
        "/home/grandee/projects/joint_align/corpora/ERV_NEW/cached_lm_ERV_NEW_2000_126_train.txt")

    pos_count_total = defaultdict(int)
    for idx, pos in total_pos_dict.items():
        if not pos in ["(", ")", ",", ".", ":"]:
            pos_count_total[pos] += index_freq.get(idx, 0)
    pos_count_correct = defaultdict(int)
    for idx, pos in aligned_pos_dict.items():
        if not pos in ["(", ")", ",", ".", ":"]:
            pos_count_correct[pos] += index_freq.get(idx, 0)
    pos_count_wrong = defaultdict(int)
    for pos, count in pos_count_correct.items():
        pos_count_wrong[pos] = pos_count_total[pos] - count

    correct_std = [0] * len(pos_count_correct)
    wrong_std = [0] * len(pos_count_correct)
    width = 0.35  # the width of the bars: can also be len(x) sequence

    labels = list(pos_count_correct.keys())
    hits = list(pos_count_correct.values())
    misses = list(pos_count_wrong.values())
    fig, ax = plt.subplots()

    ax.bar(labels,
           hits,
           width, yerr=correct_std, label='Hits')
    ax.bar(labels,
           misses,
           width, yerr=wrong_std, bottom=hits,
           label='Misses')

    ax.set_ylabel('Counts')
    ax.set_title('POS tags')
    # ax.set_yscale('log')
    ax.legend()

    plt.show()
    plt.savefig('plots/pos_freq.pdf', bbox_inches='tight')


def get_POS_acc(model_path):
    config = AutoConfig.from_pretrained("/home/grandee/projects/joint_align/configs/bert-tiny.json")
    config.vocab_size = 4000
    config.max_position_embeddings = 128
    model = AutoModelWithLMHead.from_pretrained(
        model_path,
        config=config
    )
    model.eval()
    bli_result = return_translation_indices(model, 2000)
    with open("/home/grandee/projects/joint_align/vocab/ERV_NEW_2000/vocab.txt", 'r') as f:
        text = f.read().split()
    total_word_dict = {}
    for i, word in enumerate(text):
        if not word.startswith("##"):
            total_word_dict[i] = word

    total_pos_dict = {}
    for i, word in total_word_dict.items():
        total_pos_dict[i] = nltk.pos_tag([word])[0][1]
    result_list = []
    for i, word in total_word_dict.items():
        scr_pos = total_pos_dict[i]
        tgt_pos = total_pos_dict.get(bli_result[i], None)
        result_list.append(scr_pos == tgt_pos)
    return sum(result_list) / len(total_word_dict)


def plot_POS_acc(model_path):
    path = "/home/grandee/projects/joint_align/models/{}".format(model_path)
    ordering_and_checkpoint_path = []
    files = glob.glob(os.path.join(path, "checkpoint-*"))
    for path in files:
        regex_match = re.match(".*checkpoint-([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    results = {}
    pos_result = []
    for file in checkpoints_sorted:
        pos_acc = get_POS_acc(file)
        _, filename = os.path.split(file)
        result_path = os.path.join(file, "results.json")
        result = json.load(open(result_path, 'r'))
        results[filename] = result
        pos_result.append(pos_acc)

    for layer in [0, 4, 8, 12]:
        layer_ret = {key: value["layer{}".format(layer)]["retrieval_results"]["forward"]["P@1"] for key, value in
                     results.items()}
        layer_alg = {key: value["layer{}".format(layer)]["alignment_results"]["forward"]["f1"] for key, value in
                     results.items()}
        layer_trn = {key: value["layer{}".format(layer)]["translation_results"]["forward"]["P@1"] for key, value in
                     results.items()}
        ppl = {key: value["perplexity"] for key, value in results.items()}

        host = host_subplot(111)
        par = host.twinx()

        host.set_xlabel("Timestep")
        host.set_ylabel("P@1 / F1")
        par.set_ylabel("Perplexity")

        host.plot(layer_ret.values(), 'o--', label='retrieval', color='green', alpha=0.8)
        host.plot(layer_alg.values(), 'o--', label='alignment', color='blue', alpha=0.8)
        host.plot(layer_trn.values(), 'o--', label='translation', color='brown', alpha=0.8)
        host.plot(pos_result, 'o--', label='pos translation', color='orange', alpha=0.8)
        par.plot(ppl.values(), 'o--', label="perlexity", color='black', alpha=0.8)

        host.grid(axis='y', color='0.7')
        host.legend(title='Layer {}'.format(layer))
        plt.savefig("plots/{}_layer{}_pos.pdf".format(model_path, layer), bbox_inches='tight')
        plt.show()
        plt.cla()  # Clear axis
        plt.clf()  # Clear figure


def plot_ret_algn_trans_ppl(model_path):
    path = "/home/grandee/projects/joint_align/models/{}".format(model_path)
    ordering_and_checkpoint_path = []
    files = glob.glob(os.path.join(path, "checkpoint-*"))
    for path in files:
        regex_match = re.match(".*checkpoint-([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    results = {}
    for file in checkpoints_sorted:
        _, filename = os.path.split(file)
        result_path = os.path.join(file, "results.json")
        result = json.load(open(result_path, 'r'))
        results[filename] = result

    from mpl_toolkits.axes_grid1 import host_subplot
    import matplotlib.pyplot as plt

    for layer in [0, 4, 8, 11]:
        layer_ret = {key: value["layer{}".format(layer)]["retrieval_results"]["forward"]["P@1"] for key, value in
                     results.items()}
        layer_alg = {key: value["layer{}".format(layer)]["alignment_results"]["forward"]["f1"] for key, value in
                     results.items()}
        layer_trn = {key: value["layer{}".format(layer)]["translation_results"]["forward"]["P@1"] for key, value in
                     results.items()}
        ppl = {key: value["perplexity"] for key, value in results.items()}

        host = host_subplot(111)
        par = host.twinx()

        host.set_xlabel("Timestep")
        host.set_ylabel("P@1 / F1")
        par.set_ylabel("Perplexity")

        host.plot(layer_ret.values(), 'o--', label='retrieval', color='green', alpha=0.8)
        host.plot(layer_alg.values(), 'o--', label='alignment', color='blue', alpha=0.8)
        host.plot(layer_trn.values(), 'o--', label='translation', color='brown', alpha=0.8)
        par.plot(ppl.values(), 'o--', label="perlexity", color='black', alpha=0.8)

        host.grid(axis='y', color='0.7')
        host.legend(title='Layer {}'.format(layer))

        plt.savefig("plots/{}_layer{}.pdf".format(model_path, layer), bbox_inches='tight')
        plt.cla()  # Clear axis
        plt.clf()  # Clear figure


def plot_intra_ppl_mscore_corel():
    mpl.rcParams.update({'font.size': 18})
    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_ylabel("mscore")
    ax.set_xlabel("ppl")
    mlm_paths = ["cached_lm_ERV_NEW_2000_ERV_OLD_126",
                 "cached_lm_ERV_NEW_2000_TED2013_126", "cached_lm_ERV_NEW_2000_wiki_simple_sents_126",
                 "cached_lm_ERV_NEW_2000_INV_126", "cached_lm_ERV_NEW_2000_RND_126",
                 "cached_lm_ERV_NEW_2000ERV_NEW_500_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000ERV_NEW_1000_ERV_NEW_126",
                 "cached_lm_ERV_NEW_2000ERV_NEW_4000_ERV_NEW_126"]
    mlm_labels = ["OLD",
                  "TED", "wiki",
                  "INV", "RND",
                  "500",
                  "1000",
                  "4000"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:gray", "tab:pink", "tab:brown", "tab:purple"]
    for path, label, color in zip(mlm_paths, mlm_labels, colors):
        checkpoints = _sorted_checkpoints_from_path(os.path.join("models", path))
        mlm_results, mlm_retrieval, mlm_alignment, mlm_translation = [], [], [], []
        perplexitys = []
        for checkpoint in checkpoints:
            result = get_checkpoint_result(checkpoint)
            ave_results = get_average_from_result(result, is_clm=False)
            mlm_results.append(sum(ave_results) / len(ave_results))
            mlm_retrieval.append(ave_results[0])
            mlm_alignment.append(ave_results[1])
            mlm_translation.append(ave_results[2])
            perplexitys.append(result['perplexity'])

        ax.scatter(perplexitys, mlm_results, c=color, label=label,
                   alpha=0.7, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.savefig('plots/intra_ppl_mscore_correl.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # , "500", "1K", "4K"


def plot_improvements():
    mlm_paths = ["cached_lm_ERV_NEW_2000_INV_126",
                 "cached_lm_ERV_NEW_2000_INV_126_improve_high_freq",
                 "cached_lm_ERV_NEW_2000_INV_126_improve_mid_freq",
                 "cached_lm_ERV_NEW_2000_INV_126_improve_low_freq",
                 "cached_lm_ERV_NEW_2000_INV_126_improve_nn_high_freq",
                 "cached_lm_ERV_NEW_2000_INV_126_improve_nn_mid_freq",
                 "cached_lm_ERV_NEW_2000_INV_126_improve_nn_low_freq"]
    all_data = []
    for path in mlm_paths:
        # checkpoints = _sorted_checkpoints(os.path.join("models", path))
        # mlm_results, mlm_retrieval, mlm_alignment, mlm_translation = [], [], [], []
        # perplexitys = []
        # for checkpoint in checkpoints:
        #     result = get_checkpoint_result(checkpoint)
        #     ave_results = get_average_from_result(result, is_clm=False)
        #     mlm_results.append(sum(ave_results) / len(ave_results))
        #     mlm_retrieval.append(ave_results[0])
        #     mlm_alignment.append(ave_results[1])
        #     mlm_translation.append(ave_results[2])
        #     perplexitys.append(result['perplexity'])
        result = get_final_checkpoint_result(os.path.join("models", path))
        ave_results = get_average_from_result(result, is_clm=False)
        all_data.append(sum(ave_results) / len(ave_results))
        diff_data = [i - all_data[0] for i in all_data]
    fig, axs = plt.subplots()
    axs.bar([y for y in range(len(diff_data))], diff_data)
    axs.set_title('improvements')
    plt.setp(axs, xticks=[y for y in range(len(diff_data))],
             xticklabels=['INV', 'high', 'mid', 'low', 'nn_high', 'nn_mid', "nn_low"])
    plt.savefig('plots/improvement_INV.pdf', bbox_inches='tight')
    plt.show()
    plt.cla()  # Clear axis
    plt.clf()  # , "500", "1K", "4K"


if __name__ == "__main__":
    # plot_clm_mlm_score()
    # plot_clm_mlm_key()
    # plot_mlm_js_score()
    # plot_mlm_ppl_score()
    # plot_bli_freq_dist("/home/grandee/projects/joint_align/models/cached_lm_ERV_NEW_2000_ERV_NEW_126/checkpoint-6000/pytorch_model.bin")

    print("done")
    plot_improvements()
