from typing import Union, Tuple, List

import numpy as np

from .tasks import BinaryTask, TaskType, Mask, choose_minimal_set


class Periodic(BinaryTask):
    """Generate all binary periodic sequences with lengths."""

    def __init__(self, lengths: Union[int, List[int]]):
        super().__init__("periodic", lengths)

    def generate_tasks(
        self, seq_len: int = 100, max_n_seq: int = 10
    ) -> Tuple[TaskType, Mask]:
        tasks = []
        st = set()
        for t in self.lengths:
            for s in range(2**t):
                ft = "{{0:0{}b}}".format(t)
                base = [int(i) for i in ft.format(s)]
                task = base * (seq_len // len(base))
                task = (task + base[: seq_len - len(task)])[:]
                task_list = [str(i) for i in task]
                task_str = "".join(task_list)
                if task_str not in st:
                    tasks.append(task_list)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq)


class IncreasingPeriod(BinaryTask):
    """Generate all binary periodic sequences with increasing periods with
    lengths.
    """

    def __init__(self, lengths: Union[int, List[int]]):
        super().__init__("inc-per", lengths)

    def generate_tasks(
        self, seq_len: int = 100, max_n_seq: int = 10
    ) -> Tuple[TaskType, Mask]:
        tasks = []
        st = set()
        for t in self.lengths:
            for s in range(2**t):
                ft = "{{0:0{}b}}".format(t)
                base = [int(i) for i in ft.format(s)]
                task = base[:]
                ct = 2
                while len(task) < seq_len:
                    task = (
                        task
                        + np.concatenate(
                            [np.array(base)[:, None] for _ in range(ct)], axis=1
                        )
                        .reshape(-1)
                        .tolist()
                    )
                    ct += 1

                task = task[:seq_len]
                task_list = [str(i) for i in task]
                task_str = "".join(task_list)
                if task_str not in st:
                    tasks.append(task_list)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq)


class RandomPeriodic(BinaryTask):
    """Generate random sequences with N-Grams of length in lenghts."""

    def __init__(self, lengths: Union[int, List[int]]):
        super().__init__("rand-per", lengths)

    def generate_tasks(
        self, seq_len: int = 100, max_n_seq: int = 10
    ) -> Tuple[TaskType, Mask]:
        tasks = []
        st = set()
        for t in self.lengths:
            for _ in range(max_n_seq):
                seq = np.random.randint(2**t, size=1 + seq_len // t)
                ft = "{{0:0{}b}}".format(t)
                task: list[int] = sum(
                    [[int(i) for i in ft.format(q)] for q in seq], []
                )[:seq_len]

                task_list = [str(i) for i in task]
                task_str = "".join(task_list)
                if task_str not in st:
                    tasks.append(task_list)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq)
