"""The supervised recurent models module."""
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
from torch import nn, optim


class RNN:
    """The trainable RNN baseline class."""

    def __init__(
        self, n_input: int, hidden_size: int, out_size: int, batch_size: int = 16
    ):
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.rnn = nn.RNN(n_input, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
        self.optimizer = optim.SGD(
            chain(self.linear.parameters(), self.rnn.parameters()),
            lr=0.01,
            weight_decay=0.01,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

        # Some state variables for training
        self.running_err = 0
        self.counter = 0

    @property
    def state_size(self):
        return self.hidden_size

    def step(
        self, inp: np.ndarray, targets: np.ndarray, mask: List[int]
    ) -> Optional[float]:
        out, _ = self.rnn(torch.Tensor(inp))

        if mask is not None:
            tgt = targets[mask]
            msk_out = self.linear(out[np.array(mask) - 1])
        else:
            tgt = targets[1:]
            msk_out = self.linear(out[:-1])

        msk_out = msk_out.reshape(msk_out.shape[0] * msk_out.shape[1], -1)
        err = self.loss_fn(msk_out, torch.Tensor(tgt).long())
        err.backward()

        if counter % self.batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_err = self.running_err / self.batch_size
            self.running_err = 0
        return batch_err
