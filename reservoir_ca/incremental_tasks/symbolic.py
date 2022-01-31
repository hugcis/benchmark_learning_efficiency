import collections
from typing import Union, Tuple, List

import numpy as np

from .tasks import TokenTask, TaskType, Mask, choose_minimal_set


class SymbolCounting(TokenTask):
    def __init__(self, lengths: Union[int, List[int]],
                 dictionary: List[str] = ["A", "B", "C"],
                 query_symbol: str = "x"):
        super().__init__("sym-ct", lengths, dictionary +
                         [query_symbol] +
                         [str(i) for i in range(10)])
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.base_dic = dictionary

    def generate_tasks(self, max_n_seq: int = 10,
                       **kwargs) -> Tuple[TaskType, Mask]:
        del kwargs
        tasks = []
        mask = []
        st = set()
        n_rand = max(max_n_seq // len(self.lengths), 1)
        for t in self.lengths:
            for _ in range(3 * n_rand):
                current_task_mask = []
                n_queries = np.random.randint(1, len(self.base_dic) + 1)
                left = np.random.choice(self.base_dic, size=t,
                                        replace=True).tolist()
                ct: collections.Counter = collections.Counter(left)

                # For now avoid the case with more than ten times the same
                # symbol as it makes us choose between treating 10 and plus as
                # separate tokens or as two single digit tokens
                if any(c >= 10 for c in ct.values()):
                    continue

                tk = np.random.choice(self.base_dic, size=n_queries,
                                      replace=False)
                for tc in tk:
                    left = left + [self.query_symbol, tc, str(ct[tc])]
                    current_task_mask.append(len(left) - 1)
                task_str = "".join(left)
                if task_str not in st:
                    tasks.append(left[:])
                    mask.append(current_task_mask[:])
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


class HardSymbolCounting(TokenTask):
    def __init__(self, lengths: Union[int, List[int]],
                 dictionary: List[str] = ["A", "B", "C", "D", "E"],
                 separator_symbol: str = "y",
                 query_symbol: str = "x"):
        super().__init__("hard-sym-ct", lengths, dictionary +
                         [query_symbol, separator_symbol] +
                         [str(i) for i in range(10)])
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        self.base_dic = dictionary + [separator_symbol]

    def generate_tasks(self, max_n_seq: int = 10,
                       **kwargs) -> Tuple[TaskType, Mask]:
        del kwargs
        tasks = []
        mask = []
        st = set()
        n_rand = max(max_n_seq // len(self.lengths), 1)
        for t in self.lengths:
            for _ in range(5 * n_rand):
                current_task_mask = []
                left = np.random.choice(
                    self.base_dic + [self.separator_symbol] * 2
                    * len(self.base_dic),
                    size=int(2.5 * t), replace=True).tolist()
                while left and left[0] == self.separator_symbol:
                    left.pop(0)
                while left and left[-1] == self.separator_symbol:
                    left.pop(0)
                if not left:
                    continue
                # No duplicates separator symbols
                left = [v for i, v in enumerate(left)
                        if (i == 0
                            or (v == self.separator_symbol
                                and v != left[i-1])
                            or v != self.separator_symbol)]

                ct = collections.Counter(
                    "".join(left).split(self.separator_symbol))
                n_queries = np.random.randint(1, len(ct.keys()) + 1)

                tk = np.random.choice(list(ct.keys()), size=n_queries,
                                      replace=False)
                left = left + [self.query_symbol]
                for tc in tk:
                    if ct[tc] >= 10:
                        continue
                    left = left + list(tc) + [self.separator_symbol,
                                              str(ct[tc])]
                    current_task_mask.append(len(left) - 1)
                    if np.random.random() > 0.3:
                        negative = list(3 * tc[:])
                        np.random.shuffle(negative)
                        negative = negative[:len(tc) +
                                            np.random.randint(-2, 3)]
                        if negative and not "".join(negative) in ct:
                            left = left + negative + [
                                self.separator_symbol, "0"]
                            current_task_mask.append(len(left) - 1)

                task_str = "".join(left)
                if task_str not in st:
                    tasks.append(left[:])
                    mask.append(current_task_mask)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)
