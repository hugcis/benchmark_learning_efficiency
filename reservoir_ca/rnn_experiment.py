"""This module if for running the experiments with the standard recurrent models
such as RNNs.
"""
from typing import Optional

import numpy as np

from reservoir_ca.experiment import (
    ExpOptions,
    get_train_test_split,
    group_by_lens,
    to_dim_one_hot,
)
from reservoir_ca.tasks import Task, TokenTask, BinarizedTask
from reservoir_ca.standard_recurrent import RNN


class RNNExperiment:
    rnn: Optional[RNN] = None
    task: Task

    def __init__(
        self,
        task: Task,
        exp_options: ExpOptions = ExpOptions(),
        rnn: Optional[RNN] = None,
    ):
        self.opts = exp_options
        if rnn is not None:
            self.rnn = rnn

        self.preproc = None

        if exp_options.binarized_task and isinstance(task, TokenTask):
            self.task = BinarizedTask(task)
        elif exp_options.binarized_task:
            raise ValueError("Task cannot be binarized")
        else:
            self.task = task

        tasks, masks = self.task.generate_tasks(
            seq_len=exp_options.seq_len, max_n_seq=exp_options.max_n_seq
        )

        self.dic = {d: n for n, d in enumerate(self.task.dictionary)}

        train_seqs, train_masks, test_seqs, test_masks = get_train_test_split(
            tasks, masks, self.dic, self.opts.ignore_mask
        )

        # Group sequences by lengths for batch processing
        self.training_tasks, self.training_masks = group_by_lens(
            train_seqs, train_masks
        )
        self.testing_tasks, self.testing_masks = group_by_lens(test_seqs, test_masks)
        self.shuffle = True

    @property
    def output_dim(self) -> int:
        return self.task.output_dimension()

    def check_rnn(self) -> RNN:
        if self.rnn is not None:
            return self.rnn
        else:
            raise ValueError("RNN should be set with set_rnn")

    def set_rnn(self, rnn: RNN):
        self.rnn = rnn

    def fit_with_eval(self) -> list[float]:
        rnn = self.check_rnn()
        if self.training_masks is not None:
            all_tasks = [
                (example, self.training_masks[length_idx][ex_idx])
                for length_idx, task in enumerate(self.training_tasks)
                for ex_idx, example in enumerate(task)
            ]
            np.random.shuffle(all_tasks)
            for (example, msk) in all_tasks:
                encoded = to_dim_one_hot(example, self.output_dim)
                encoded = encoded.reshape(encoded.shape[0], 1, encoded.shape[1])
                rnn.step(encoded, example, msk)
