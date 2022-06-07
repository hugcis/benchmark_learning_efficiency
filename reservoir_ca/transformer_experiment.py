"""This module if for running the experiments with the standard transformer
models.
"""
import logging
from typing import Optional, List

import numpy as np
import torch
from torch import nn

from reservoir_ca.experiment import (
    ExpOptions,
    get_train_test_split,
    group_by_lens,
    to_dim_one_hot,
)
from reservoir_ca.supervised_wade_exps.model import TransformerModel
from reservoir_ca.tasks import BinarizedTask, Task, TokenTask

DEVICE = torch.device("cpu")


class Transformer:
    def __init__(
        self,
        n_tokens: int,
        n_inp: int,
        n_head: int,
        n_hidden: int,
        n_output: int,
        batch_size: int = 8,
    ):
        self.transformer = TransformerModel(
            ntoken=n_tokens,
            ninp=n_inp,
            nhead=n_head,
            nhid=n_hidden,
            nlayers=1,
            noutput=n_output,
            add_dropout=False,
        )
        self.transformer.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.transformer.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

        # Some state variables for training
        self.running_err = 0
        self.counter = 0

    def n_parameters(self) -> int:
        return sum(i.reshape(-1).size()[0] for i in self.transformer.parameters())

    def apply(self, inp: np.ndarray, mask: List[List[int]]) -> torch.Tensor:
        out = self.transformer(torch.Tensor(inp).long().to(DEVICE), take_mean=False)
        if mask is not None:
            masked_output = torch.cat(
                [
                    out[t - 1 : t, i : i + 1, :]
                    for i, s_msk in enumerate(mask)
                    for t in s_msk
                ],
                1,
            )
        else:
            masked_output = out[:, :-1]
        masked_output = masked_output.reshape(
            masked_output.shape[0] * masked_output.shape[1], -1
        )
        return masked_output

    def step(
        self, inp: np.ndarray, targets: np.ndarray, mask: List[List[int]]
    ) -> Optional[float]:
        """
        Args:
            inp: array of shape [seq_len, b_size (=1)]
            targets: array of shape [seq_len]
            mask: list of mask for each sequence. the length of the list is equal
                to batch size.
        Returns:
            A tensor with batch_size * seq_len
        """
        self.transformer.train()
        masked_output = self.apply(inp, mask)
        if mask is not None:
            tgt = targets[mask[0]]
        else:
            tgt = targets[1:]

        err = self.loss_fn(masked_output, torch.Tensor(tgt).long())
        err.backward()
        self.running_err += err.item()
        self.counter += 1

        if self.counter % self.batch_size == 0:
            self.optimizer.step()
            for p in self.transformer.parameters():
                p.grad = None
            batch_err = self.running_err / self.batch_size
            logging.debug("Transformer training error %s", batch_err)
            self.running_err = 0
            return batch_err
        else:
            return None

    def score(
        self, inp: np.ndarray, targets: np.ndarray, mask: List[List[int]]
    ) -> List[float]:
        self.transformer.eval()
        with torch.no_grad():
            msk_out = self.apply(inp, mask)
            if mask is not None:
                tgt = np.concatenate([targets[mask[i], i] for i in range(len(mask))])
            else:
                tgt = targets[1:]
            return np.argmax(msk_out.cpu().detach().numpy(), axis=1) == tgt


class TransformerExperiment:
    """An experiment with fully training a recurrent model."""

    transformer: Optional[Transformer] = None
    task: Task

    def __init__(
        self,
        task: Task,
        exp_options: ExpOptions = ExpOptions(),
        transformer: Optional[Transformer] = None,
    ):
        self.opts = exp_options
        if transformer is not None:
            self.transformer = transformer

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

    def check_model(self) -> Transformer:
        if self.transformer is not None:
            return self.transformer
        else:
            raise ValueError("RNN should be set with set_model")

    def set_model(self, transformer: Transformer):
        self.transformer = transformer

    def validation_score(self) -> float:
        transformer = self.check_model()
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
                        # to_dim_one_hot(i[0], self.output_dim)[:, None, :]
                        i[0][:, None]
                        for i in single_length_task
                    ],
                    axis=1,
                )
                example = np.concatenate(
                    [np.array(i[0])[:, None] for i in single_length_task],
                    axis=1,
                )
                msk = [i[1] for i in single_length_task]
                score.append(transformer.score(encoded, example, msk))

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
        model = self.check_model()
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
                    # encoded = to_dim_one_hot(example, self.output_dim)
                    encoded = example.reshape(example.shape[0], 1)
                    error = model.step(encoded, example, msk)
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
