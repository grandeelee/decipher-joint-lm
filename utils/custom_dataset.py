import os
import pickle
import time

import torch
from torch.utils.data.dataset import Dataset
import logging
from filelock import FileLock

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class LoadTextDataset(Dataset):

    def __init__(
            self,
            file_path: str,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        start = time.time()
        with open(file_path, "rb") as handle:
            self.examples = pickle.load(handle)
        logger.info(
            f"Loading features from cached file {file_path} [took %.3f s]", time.time() - start
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def save_line_by_line_dataset(tokenizer: PreTrainedTokenizer,
                              file_path: str,
                              block_size: int,
                              save_name: str,
                              cache_dir: str):
    assert os.path.isfile(file_path), f"Input file path {file_path} not found"
    logger.info(f"Creating features from dataset file at {file_path}")

    block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        cache_dir if cache_dir is not None else directory,
        f"cached_{save_name}_{filename}",
    )

    logger.info(f"Creating features from dataset file at {directory}")

    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
    examples = batch_encoding["input_ids"]
    # self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
    start = time.time()
    with open(cached_features_file, "wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
    )


def save_retrieval_dataset(tokenizer: PreTrainedTokenizer,
                           file_path: str,
                           block_size: int,
                           save_name: str,
                           cache_dir: str):
    assert os.path.isfile(file_path), f"Input file path {file_path} not found"

    logger.info(f"Creating features from dataset file at {file_path}")

    block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        cache_dir if cache_dir is not None else directory,
        f"cached_{save_name}_{filename}",
    )
    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)

    examples = {"input_ids": batch_encoding["input_ids"],
                "word_ids": [batch_encoding.word_ids(i) for i in range(len(batch_encoding["input_ids"]))]}
    start = time.time()
    with open(cached_features_file, "wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
    )


class LoadRetrievalDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
            self,
            file_path: str,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        with open(file_path, "rb") as handle:
            data = pickle.load(handle)
        self.examples = data["input_ids"]
        self.word_ids = data["word_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
