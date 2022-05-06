from typing import List, Tuple, Type

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from torchtext.datasets import IMDB


def get_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(special_tokens=["[PAD]", "[UNK]"])
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def get_train_test_sets():
    train_iter = IMDB(split="train", root=".data")
    test_iter = IMDB(split="test", root=".data")

    return train_iter, test_iter


def get_tokenized_dataset(dataset, tokenizer) -> List[Tuple[List[int], int]]:
    encoded = tokenizer.encode_batch([i[1] for i in dataset])
    tokenized = list(
        zip(
            [output.ids for output in encoded],
            [(1 if i[0] == "pos" else 0) for i in dataset],
        )
    )
    return tokenized
