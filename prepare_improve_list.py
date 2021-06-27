# propose alignment list for improvements only try on domain and inv, rnd for now
# same vocab list
import pickle
from nlp.vocab import get_unigram_from_tokenized
from collections import OrderedDict
import nltk
import json

def get_unigram_from_path(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return get_unigram_from_tokenized(data)


unigram = get_unigram_from_path(
    "/home/grandee/Projects/TACL_v1/data/cached_2000_NT_valid.txt")

freq_idx = [i for i in unigram.keys() if i < 2000]

# low freq 100
low_freq = freq_idx[-100:]

# mid freq 100
mid_freq = freq_idx[len(freq_idx) // 2 - 50: len(freq_idx) // 2 + 50]

# high freq 100
high_freq = freq_idx[100:200]

# the above low mid high freq within NN category
with open("/home/grandee/Projects/TACL_v1/tokenizer/2000/vocab.txt", 'r') as f:
    text = f.read().split()
total_word_dict = {}
for i, word in enumerate(text):
    if not word.startswith("##"):
        total_word_dict[i] = word

total_pos_dict = {}
for i, word in total_word_dict.items():
    total_pos_dict[i] = nltk.pos_tag([word])[0][1]

nn_freq = {}
for idx, pos in total_pos_dict.items():
    if pos == "NN" and idx in unigram.keys() and idx not in [1, 2]:
        nn_freq[idx] = unigram[idx]

nn_sorted = OrderedDict(sorted(nn_freq.items(), key=lambda item: (-item[1], item[0])))
nn_freq_idx = [i for i in nn_sorted.keys()]

nn_low_freq = nn_freq_idx[-100:]
nn_mid_freq = nn_freq_idx[len(nn_freq_idx) // 2 - 50: len(nn_freq_idx) // 2 + 50]
nn_high_freq = nn_freq_idx[:100]

with open("improve_list.json", 'w') as f:
    json.dump({
        "low_freq": low_freq,
        "mid_freq": mid_freq,
        "high_freq": high_freq,
        "nn_low_freq": nn_low_freq,
        "nn_mid_freq": nn_mid_freq,
        "nn_high_freq": nn_high_freq,
    }, f)
