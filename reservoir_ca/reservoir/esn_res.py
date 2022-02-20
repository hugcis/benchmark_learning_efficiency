from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from reservoir_ca.reservoir.base import RState, Reservoir


class ESN(Reservoir):
    def __init__(
        self,
        inp_size: int,
        redundancy: int = 4,
        r_height: int = 2,
        proj_factor: int = 40,
    ):
        self.proj_factor = proj_factor
        self.redundancy = redundancy
        self.r_height = r_height
        self.inp_size = inp_size
        self.rnn = nn.RNNCell(self.inp_size, self.state_size)

        wg_shape = self.rnn.weight_hh.data.shape
        wg_mask = torch.rand(wg_shape) < 10 / wg_shape[1]

        self.rnn.weight_hh.data = torch.Tensor((2 * torch.rand(wg_shape) - 1) * wg_mask)
        self.rnn.weight_ih.data = torch.Tensor(
            2 * torch.rand(self.rnn.weight_ih.data.shape) - 1
        )

    @property
    def state_size(self) -> int:
        return self.proj_factor * self.redundancy * self.r_height

    @property
    def output_size(self) -> int:
        return self.state_size

    def __call__(self, state: np.ndarray, inp: np.ndarray) -> Tuple[np.ndarray, RState]:
        mod_state = self.rnn(torch.Tensor(inp), torch.Tensor(state))
        output = mod_state[:, None, :]

        return output.detach().numpy(), mod_state.detach().numpy()
