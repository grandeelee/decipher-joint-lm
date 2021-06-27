import os
import re
import glob
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig


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