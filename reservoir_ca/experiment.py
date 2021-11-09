from dataclasses import dataclass, field
from typing import List, Any, Tuple, Optional

import numpy as np
from sklearn.svm import LinearSVC

from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.tasks import Task

@dataclass
class ExpOptions:
    seq_len: int = 100
    max_n_seq: int = 300
    n_rep: int = 10
    seed: int = 0
    redundancy: int = 4
    rules: list[int] = field(default_factory=lambda : list(range(256)))


def to_dim_one_hot(data, out_dim):
    return np.eye(out_dim)[data]


def group_by_lens(seqs: List[List[Any]]) -> List[np.ndarray]:
    lens = set(len(c) for c in seqs)
    return [np.array([c for c in seqs if len(c) == l]) for l in lens]


class Experiment:
    def __init__(self, ca: CAReservoir, task: Task, exp_options: ExpOptions = ExpOptions()):
        self.opts = exp_options
        self.ca = ca
        self.reg = LinearSVC(dual=False)
        self.task = task
        tasks = task.generate_tasks(seq_len=exp_options.seq_len,
                                    max_n_seq=exp_options.max_n_seq)

        self.dic = {d: n for n, d in enumerate(self.task.dictionary)}

        split = np.random.permutation(range(len(tasks)))
        train_seqs = [[self.dic[k] for k in tasks[i]]
                      for i in split[:int(.8 * len(tasks))]]
        self.training_tasks = group_by_lens(train_seqs)
        test_seqs = [[self.dic[k] for k in tasks[i]]
                     for i in split[int(.8 * len(tasks)):]]
        self.testing_tasks = group_by_lens(test_seqs)

    @property
    def output_dim(self) -> int:
        return self.task.output_dimension()

    def process_tasks(self, tasks) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_data = []
        all_tgts = []
        for task_l in tasks:
            single_length_data = []
            state = np.zeros((task_l.shape[0], self.ca.state_size))
            for t in range(task_l.shape[1] - 1):
                inp = to_dim_one_hot(task_l[:, t], self.output_dim)
                output, state = self.ca(state, inp)
                single_length_data.append(output[:, None, :, :])
            single_length_tgts = task_l[:, 1:].reshape(-1)

            all_data.append(np.concatenate(single_length_data, axis=1).reshape(-1, self.ca.output_size))
            all_tgts.append(single_length_tgts)
        return all_data, all_tgts

    def fit(self, return_data=False) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        all_data, all_tgts = self.process_tasks(self.training_tasks)
        # Flatten inp and targets for training of SVM
        self.reg.fit(np.concatenate(all_data, axis=0), np.concatenate(all_tgts, axis=0))

        if return_data:
            inp = [all_data[c].reshape(len(task_l), -1, self.ca.state_size)
                   for c, task_l in enumerate(self.training_tasks)]
            tgts = [all_tgts[c].reshape(len(task_l), -1)
                   for c, task_l in enumerate(self.training_tasks)]
            return inp, tgts
        else:
            return None

    def eval_test(self) -> float:
        all_data, all_tgts = self.process_tasks(self.testing_tasks)

        return self.reg.score(np.concatenate(all_data, axis=0), np.concatenate(all_tgts, axis=0))
