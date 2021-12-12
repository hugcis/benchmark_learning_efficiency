from enum import Enum
from typing import Optional, Tuple

import numpy as np


class ProjectionType(Enum):
    ONE_TO_ONE = 1
    ONE_TO_MANY = 2
    ONE_TO_PATTERN = 3

    def __str__(self):
        return f"{self.name}"


def rule_array_from_int(ca_rule: int) -> np.ndarray:
    """
    Converts a rule id to an array corresponding to all ordered binary inputs.
    Only works with ECA rule/ids (ie 2 states, neighborhood size 1).
    """
    rule_list = bin(ca_rule)[2:]
    # Zero padding of the rule
    rule_list = [0] * (8 - len(rule_list)) + [int(i) for i in rule_list]
    # Convert to array and reverse the list
    return np.array(rule_list[::-1])


def validate_rule(rule: int, n_size: int, n_states: int):
    neigh_size = 2 * n_size + 1
    n_possible_neigh = n_states ** neigh_size
    n_possible_rules = n_states ** n_possible_neigh
    return rule < n_possible_rules


class CAReservoir:
    def __init__(self, ca_rule: int, inp_size: int, redundancy: int = 4,
                 r_height: int = 2, proj_factor: int = 40,
                 proj_type: ProjectionType = ProjectionType.ONE_TO_ONE,
                 proj_pattern: Optional[int] = None, n_size: int = 1):
        self.redundancy = redundancy
        self.r_height = r_height
        self.proj_factor = proj_factor
        self.ca_rule = ca_rule
        self.n_size = n_size
        # We only support rules with neighborhood sizes 1 or 2
        assert self.n_size == 1 or self.n_size == 2
        if not validate_rule(self.ca_rule, self.n_size, 2):
            raise ValueError(f"Rule {self.ca_rule} incompatible with 2 states and "
                             f"neighborhood size {self.n_size}")

        self.rule_array = rule_array_from_int(self.ca_rule)

        self.inp_size = inp_size
        assert isinstance(proj_type, ProjectionType)
        self.proj_type = proj_type
        self.proj_pattern = proj_pattern
        self.proj_matrix = self.set_proj_matrix()

        self.input_function = np.logical_xor


    def set_proj_matrix(self) -> np.ndarray:
        proj_matrix = np.zeros((self.inp_size, self.state_size))
        if self.proj_type is ProjectionType.ONE_TO_ONE:
            for t in range(self.redundancy):
                idx_x = np.random.permutation(self.inp_size)
                idx_y = np.random.choice(self.proj_factor, size=self.inp_size)
                proj_matrix[:, t * self.proj_factor:(t + 1) * self.proj_factor][idx_x, idx_y] = 1

        elif self.proj_type is ProjectionType.ONE_TO_PATTERN:
            if self.proj_pattern is None:
                raise ValueError("Parameter proj_pattern must be set for this projection type")
            for t in range(self.redundancy):
                idx_x = np.random.permutation(self.inp_size)
                idx_y = np.random.choice(self.proj_factor - self.proj_pattern, size=self.inp_size)
                pat = np.zeros((self.inp_size, self.proj_pattern))
                while np.any(pat.sum(1) == 0):
                    pat = np.random.randint(2, size=(self.inp_size, self.proj_pattern))
                for n, y in enumerate(idx_y):
                    proj_matrix[:, t * self.proj_factor:(t + 1) * self.proj_factor][
                        idx_x[n], y:y + self.proj_pattern
                    ] = pat[idx_x[n]]

        elif self.proj_type is ProjectionType.ONE_TO_MANY:
            if self.proj_pattern is None:
                raise ValueError("Parameter proj_pattern must be set for this projection type")
            for t in range(self.redundancy):
                idx_x = np.random.permutation(self.inp_size)
                idx_y = [np.random.choice(self.proj_factor, size=self.inp_size)
                         for _ in range(self.proj_pattern)]
                for n, y in enumerate(idx_y):
                    proj_matrix[:, t * self.proj_factor:(t + 1) * self.proj_factor][idx_x, y] = 1

        else:
            raise ValueError("Unrecognized projection type {}".format(self.proj_type))

        return proj_matrix

    @property
    def state_size(self) -> int:
        return self.proj_factor * self.redundancy

    @property
    def output_size(self) -> int:
        return self.state_size * self.r_height

    def apply_rule(self, state: np.ndarray):
        return self.rule_array[
            np.roll(state, -1, axis=1) + 2 * state +
            4 * np.roll(state, 1, axis=1)]

    def __call__(self, state: np.ndarray, inp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert state.shape[1] == self.state_size
        assert inp.shape[1] == self.inp_size
        projected_inp = inp @ self.proj_matrix

        # The input is encoded into the current state via the input function
        mod_state = self.input_function(projected_inp, state)
        output = np.zeros((inp.shape[0], self.r_height, self.state_size))
        # We apply r_height steps of the CA
        for i in range(self.r_height):
            mod_state = self.apply_rule(mod_state)
            output[:, i, :] = mod_state
        return output, mod_state
