import numpy as np
from typing import Union, Tuple
from collections import namedtuple
from reservoir_ca.reservoir.ca_res import CAInputFeedback
from reservoir_ca.reservoir import RState
from reservoir_ca.evo.es import SimpleGA
from incremental_tasks.symbolic import SymbolCounting
from joblib import Parallel, delayed

Dna = namedtuple("DNA", ["rule_array", "proj_in", "proj_err", "out"])
tsk = SymbolCounting([35], dictionary=["A", "B", "C", "D", "E"])
inp_size = tsk.output_dimension()
ca_rule = np.random.randint(0, 2**32)

ca = CAInputFeedback(ca_rule, inp_size, r_height=1, redundancy=10)
dna = Dna(
    ca.rule_array,
    ca.proj_matrix,
    ca.err_proj_matrix,
    (np.random.random(size=(ca.state_size, inp_size)) - 0.5)
    * (2 / np.sqrt(ca.state_size)),
)


def candidate_to_dna(candidate) -> Dna:
    binary, continuous = candidate
    r_array = binary[: dna.rule_array.size].reshape(dna.rule_array.shape)
    proj_in = binary[
        dna.rule_array.size : dna.rule_array.size + dna.proj_in.size
    ].reshape(dna.proj_in.shape)
    proj_err = binary[
        dna.rule_array.size
        + dna.proj_in.size : dna.rule_array.size
        + dna.proj_in.size
        + dna.proj_err.size
    ].reshape(dna.proj_err.shape)

    out = continuous.reshape(dna.out.shape)
    return Dna(r_array, proj_in, proj_err, out)


def ca_from_dna(dna, inp_size):
    ca = CAInputFeedback(0, inp_size, r_height=1, redundancy=10)

    ca.rule_array = dna.rule_array
    ca.proj_matrix = dna.proj_in
    ca.err_proj_matrix = dna.proj_err
    return ca


def softmax(x):
    t = np.exp(x)
    return t / t.sum()


def evaluate_dna(
    dna: Dna, return_state: bool = False
) -> Union[float, Tuple[int, int, np.ndarray, list[Tuple[str]]]]:
    inp_size = tsk.output_dimension()
    sentences, masks = tsk.generate_tasks(1200)
    token_mapping = dict(zip(tsk.dictionary, range(len(tsk.dictionary))))
    train_sentences = sentences
    train_sentences_mapped = [[token_mapping[i] for i in c] for c in train_sentences]
    # test_sentences = sentences[cutoff:]
    # test_sentences_mapped = [[token_mapping[i] for i in c] for c in test_sentences]
    train_masks = masks
    # test_masks = masks[cutoff:]

    reward = 0
    total = 0
    ca = ca_from_dna(dna, inp_size)
    state = RState(np.zeros((1, ca.state_size), dtype=int))
    err = np.eye(3)[np.array([0])]

    out = dna.out
    all_states = []
    results = []
    for k, s in enumerate(train_sentences_mapped):
        one_hot = np.eye(inp_size)[s]
        for n, i in enumerate(one_hot):
            _, state = ca(state, i[None, :], err)
            if return_state:
                all_states.append(state.copy())
            if train_masks is not None and n + 1 in train_masks[k]:
                total += 1
                result = softmax(state @ out)
                reward += np.log(result[0, s[n + 1]]) / result.shape[0]

                if result.argmax() == s[n + 1]:
                    err = np.eye(3)[np.array([1])]
                    if return_state:
                        results.append((result[:], s[n + 1]))
                else:
                    err = np.eye(3)[np.array([2])]
            else:
                err = np.eye(3)[np.array([0])]
    if return_state:
        return reward, total, np.array(all_states), results
    else:
        return reward / total


num_params = dna.out.reshape(-1).size
num_bin_params = dna.rule_array.size + dna.proj_in.size + dna.proj_err.size
ga = SimpleGA(num_params, num_bin_params, popsize=256)


def process_candidate(candidate):
    return evaluate_dna(candidate_to_dna(candidate))


for i in range(100):
    print(f"generation {i}")
    candidates = ga.ask()
    rews = []
    results = Parallel(n_jobs=8, verbose=1)(
        delayed(process_candidate)(candidate) for candidate in candidates
    )
    ga.tell(results)
    print(ga.best_reward)
