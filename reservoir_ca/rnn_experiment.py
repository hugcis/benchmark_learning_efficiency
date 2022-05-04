"""This module if for running the experiments with the standard recurrent models
such as RNNs.
"""
import logging
from typing import Optional, List

import numpy as np

from reservoir_ca.experiment import (
    ExpOptions,
    get_train_test_split,
    group_by_lens,
    to_dim_one_hot,
)
from reservoir_ca.standard_recurrent import RNN
from reservoir_ca.tasks import BinarizedTask, Task, TokenTask


class RNNExperiment:
    """An experiment with fully training a recurrent model."""

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
        logging.debug(
            "Number of tasks %s -- %s training -- %s testing",
            len(tasks),
            len(train_seqs),
            len(test_seqs),
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

    def validation_score(self) -> float:
        rnn = self.check_rnn()
        score: List[List[float]] = []
        if self.testing_masks is not None:
            all_tasks = [
                [
                    (example, self.testing_masks[length_idx][ex_idx])
                    for ex_idx, example in enumerate(task)
                ]
                for length_idx, task in enumerate(self.testing_tasks)
            ]
            for single_length_task in all_tasks:
                encoded = np.concatenate(
                    [
                        to_dim_one_hot(i[0], self.output_dim)[:, None, :]
                        for i in single_length_task
                    ],
                    axis=1,
                )
                example = np.concatenate(
                    [np.array(i[0])[:, None] for i in single_length_task],
                    axis=1,
                )
                msk = [i[1] for i in single_length_task]
                score.append(rnn.score(encoded, example, msk))

            # rnn.score return a list of bools for the whole batch
            # We flatten this list of lists into a single bool list and compute
            # the mean to get accuracy
            return np.mean([item for l_score in score for item in l_score])
        else:
            raise ValueError(
                "The mask cannot be ignored for the RNN baseline, "
                "do not use the option --ignore_mask"
            )

    def fit_with_eval(self) -> List[float]:
        rnn = self.check_rnn()
        best_reached = False
        results = []
        if self.training_masks is not None:
            all_tasks = [
                (example, [self.training_masks[length_idx][ex_idx]])
                for length_idx, task in enumerate(self.training_tasks)
                for ex_idx, example in enumerate(task)
            ]
            for _ in range(10):
                logging.debug("New epoch")
                np.random.shuffle(all_tasks)
                for (example, msk) in all_tasks:
                    encoded = to_dim_one_hot(example, self.output_dim)
                    encoded = encoded.reshape(encoded.shape[0], 1, encoded.shape[1])
                    error = rnn.step(encoded, example, msk)
                    if error is not None:
                        results.append(self.validation_score())
                        logging.debug("Validation score %s", results[-1])
                        if results[-1] == 1.0 and not best_reached:
                            logging.info(
                                "Perfect accuracy reached in %d steps", len(results)
                            )
                            best_reached = True
        else:
            raise ValueError(
                "The mask cannot be ignored for the RNN baseline, "
                "do not use the option --ignore_mask"
            )

        return results
