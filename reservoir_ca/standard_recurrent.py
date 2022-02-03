from itertools import chain
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RNN:
    def __init__(
        self, n_input: int, hidden_size: int, out_size: int, batch_size: int = 8
    ):
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.rnn = nn.RNN(n_input, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
        self.optimizer = optim.SGD(
            chain(self.linear.parameters(), self.rnn.parameters()),
            lr=0.0001,
            weight_decay=1.0,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def state_size(self):
        return self.hidden_size

    def step(self, inp: np.ndarray, targets: np.ndarray, mask: List[int]) -> float:
        out, _ = self.rnn(torch.Tensor(inp))
        if mask is not None:
            tgt = targets[mask]
            msk_out = self.linear(out[np.array(mask) - 1])
        else:
            tgt = targets[1:]
            msk_out = self.linear(out[:-1])
        msk_out = msk_out.reshape(msk_out.shape[0] * msk_out.shape[1], -1)
        err = self.loss_fn(msk_out, torch.Tensor(tgt).long())
        print(err.item())
        err.backward()
        self.optimizer.step()
        return err.item()
