from dataclasses import dataclass, field

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


class Experiment:
    def __init__(self, ca: CAReservoir, task: Task, exp_options: ExpOptions = ExpOptions()):
        self.ca = ca
        self.reg = LinearSVC(dual=False)
        tasks = task.generate_tasks(seq_len=exp_options.seq_len, max_n_seq=exp_options.max_n_seq)

        split = np.random.permutation(range(len(tasks)))
        self.training_tasks = np.array([tasks[i]
                                        for i in split[:int(.8 * len(tasks))]])
        self.testing_tasks = np.array([tasks[i]
                                       for i in split[int(.8 * len(tasks)):]])

    def fit(self, return_data=False):
        all_data = []

        state = np.zeros((self.training_tasks.shape[0], self.ca.state_size))
        for t in range(self.training_tasks.shape[1] - 1):
            inp = np.eye(2)[self.training_tasks[:, t]]
            output, state = self.ca(state, inp)
            all_data.append(output[:, None, :, :])
        all_tgts = self.training_tasks[:, 1:].reshape(-1)
        all_data = np.concatenate(all_data, axis=1).reshape(-1, self.ca.output_size)
        self.reg.fit(all_data, all_tgts)

        if return_data:
            return all_data.reshape(len(self.training_tasks), -1, self.ca.output_size), all_tgts

    def eval_test(self):
        all_data = []

        state = np.zeros((self.testing_tasks.shape[0], self.ca.state_size))
        for t in range(self.testing_tasks.shape[1] - 1):
            inp = np.eye(2)[self.testing_tasks[:, t]]
            output, state = self.ca(state, inp)
            all_data.append(output[:, None, :, :])
        all_tgts = self.testing_tasks[:, 1:].reshape(-1)
        all_data = np.concatenate(all_data, axis=1).reshape(-1, self.ca.output_size)

        return self.reg.score(all_data, all_tgts)
