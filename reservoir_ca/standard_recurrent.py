"""The supervised recurent models module."""
import logging
from enum import Enum
from itertools import chain
from typing import Optional, Union

import numpy as np
import torch
from torch import nn, optim

DEVICE = torch.device("cpu")


class NetType(Enum):
    RNN = 1
    LSTM = 2


class RNN:
    """The trainable RNN baseline class."""

    def __init__(
        self,
        n_input: int,
        hidden_size: int,
        out_size: int,
        batch_size: int = 8,
        net_type: NetType = NetType.RNN,
    ):
        self.hidden_size = hidden_size
        self.out_size = out_size

        if net_type == NetType.RNN:
            self.rnn: Union[nn.RNN, nn.LSTM] = nn.RNN(n_input, self.hidden_size)
        else:
            self.rnn = nn.LSTM(n_input, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
        self.rnn.to(DEVICE)
        self.linear.to(DEVICE)
        # self.optimizer = optim.SGD(
        # chain(self.linear.parameters(), self.rnn.parameters()),
        # lr=0.01,
        # weight_decay=0.01,
        # )
        self.optimizer = optim.Adam(
            chain(self.linear.parameters(), self.rnn.parameters()),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

        # Some state variables for training
        self.running_err = 0
        self.counter = 0

    @property
    def state_size(self):
        return self.hidden_size

    def apply(self, inp: np.ndarray, mask: list[list[int]]) -> torch.Tensor:
        """Apply the RNN.

        Inp: shape = (L, Batch=1, n_inputs)
        Mask: shape = (N,)

        returns:
          Tensor: shape = (L, N, n_inputs)
        """
        out, _ = self.rnn(torch.Tensor(inp).to(DEVICE))
        if mask is not None:
            masked_output = torch.cat(
                [
                    out[t - 1 : t, i : i + 1, :]
                    for i, s_msk in enumerate(mask)
                    for t in s_msk
                ],
                1,
            )
            msk_out = self.linear(masked_output)
        else:
            msk_out = self.linear(out[:-1])

        msk_out = msk_out.reshape(msk_out.shape[0] * msk_out.shape[1], -1)
        return msk_out

    def score(
        self, inp: np.ndarray, targets: np.ndarray, mask: list[list[int]]
    ) -> list[float]:
        self.rnn.eval()
        with torch.no_grad():
            msk_out = self.apply(inp, mask)
            if mask is not None:
                tgt = np.concatenate([targets[mask[i], i] for i in range(len(mask))])
            else:
                tgt = targets[1:]
            return np.argmax(msk_out.cpu().detach().numpy(), axis=1) == tgt

    def step(
        self, inp: np.ndarray, targets: np.ndarray, mask: list[list[int]]
    ) -> Optional[float]:

        self.rnn.train()
        msk_out = self.apply(inp, mask)
        if mask is not None:
            tgt = targets[mask[0]]
        else:
            tgt = targets[1:]

        err = self.loss_fn(msk_out, torch.Tensor(tgt).long())
        err.backward()
        self.running_err += err.item()
        self.counter += 1

        if self.counter % self.batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_err = self.running_err / self.batch_size
            logging.debug("RNN training error %s", batch_err)
            self.running_err = 0
            return batch_err
        else:
            return None
