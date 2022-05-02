import numpy as np
from typing import List, Tuple
from torchtext.datasets import IMDB
from incremental_tasks.base import TokenTask
from tokenizers import normalizers
from tokenizers.normalizers import BertNormalizer

normalizer = BertNormalizer()


def tokenize(label: str, line: str) -> Tuple[List[str], str]:
    line = normalizer.normalize_str(line)
    return list(line.lower()), label.lower()


class IMDBTask(TokenTask):
    def __init__(self, eol_symbol: str = "<EOS>"):
        train_iter = IMDB(split="train")
        # test_iter = IMDB(split="test")
        self.tokens = []
        self.msk = []
        for label, line in train_iter:
            data, label = tokenize(label, line)
            self.tokens += data + [label, eol_symbol]
            self.msk.append(len(self.tokens) - 2)
        dictionary = set(self.tokens)

        super().__init__("imdb", 0, [*dictionary])

    def generate_single(self, **kwargs):
        del kwargs
        index = np.random.randint(1, len(self.msk))
        left = self.tokens[self.msk[index - 1] + 2 : self.msk[index] + 2]
        return left, [len(left) - 2]
