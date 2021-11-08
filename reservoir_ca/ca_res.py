import numpy as np


class CAReservoir:
    def __init__(self, ca_rule: int, inp_size: int, redundancy: int = 4,
                 r_height: int = 2, proj_factor: int = 40):
        self.redundancy = redundancy
        self.r_height = r_height
        self.proj_factor = proj_factor
        self.ca_rule = ca_rule

        rule_list = bin(ca_rule)[2:]
        # Zero padding of the rule
        rule_list = [0] * (8 - len(rule_list)) + [int(i) for i in rule_list]
        # Convert to array and reverse the list
        self.rule_array = np.array(rule_list[::-1])

        self.inp_size = inp_size
        self.proj_matrix = np.zeros((self.inp_size, self.state_size))
        for t in range(self.redundancy):
            idx_x = np.random.permutation(inp_size)
            idx_y = np.random.choice(self.proj_factor, size=inp_size)
            self.proj_matrix[:, t * proj_factor:(t + 1) * proj_factor][idx_x, idx_y] = 1

    @property
    def state_size(self):
        return self.proj_factor * self.redundancy

    @property
    def output_size(self):
        return self.state_size * self.r_height

    def apply_rule(self, state):
        return self.rule_array[
            np.roll(state, -1, axis=1) + 2 * state +
            4 * np.roll(state, 1, axis=1)]

    def __call__(self, state, inp):
        assert state.shape[1] == self.state_size
        assert inp.shape[1] == self.inp_size
        projected_inp = inp @ self.proj_matrix
        xored_state = np.logical_xor(projected_inp, state)
        output = np.zeros((inp.shape[0], self.r_height, self.state_size))
        for i in range(self.r_height):
            xored_state = self.apply_rule(xored_state)
            output[:, i, :] = xored_state
        return output, xored_state
