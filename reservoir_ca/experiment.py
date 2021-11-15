import hashlib
import json
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from reservoir_ca.tasks import BinarizedTask, Task, TokenTask
from reservoir_ca.ca_res import CAReservoir, ProjectionType


class RegType(Enum):
    LINEARSVM = 1
    RBFSVM = 2
    MLP = 3
    RANDOMFOREST = 4

    @staticmethod
    def from_str(label: str) -> "RegType":
        if label in ("linearsvm", "linear"):
            return RegType.LINEARSVM
        elif label in ("rbfsvm", "rbf"):
            return RegType.RBFSVM
        elif label in ("mlp", "neural_network"):
            return RegType.MLP
        elif label in ("randomforest"):
            return RegType.RANDOMFOREST
        else:
            raise NotImplementedError


@dataclass
class ExpOptions:
    seq_len: int = 100
    max_n_seq: int = 300
    n_rep: int = 10
    seed: int = 0
    redundancy: int = 4
    r_height: int = 2
    proj_factor: int = 40
    rules: list[int] = field(default_factory=lambda : list(range(256)))
    reg_type: RegType = RegType.LINEARSVM
    ignore_mask: bool = True
    binarized_task: bool = False
    proj_type: ProjectionType = ProjectionType.ONE_TO_ONE
    proj_pattern: int = 4

    def to_json(self, filter_out: Optional[List[str]] = ["rules"]):
        dict_rep = dataclasses.asdict(self)
        if filter_out is not None:
            for s in filter_out:
                dict_rep.pop(s)
        for s in dict_rep:
            if isinstance(dict_rep[s], Enum):
                dict_rep[s] = dict_rep[s].name
        return json.dumps(dict_rep)

    @classmethod
    def from_json(cls, json_str: str):
        opts = cls()
        for name, val in json.loads(json_str).items():
            if isinstance(getattr(opts, name), Enum):
                setattr(opts, name, type(getattr(opts, name))[val])
            else:
                setattr(opts, name, val)
        return opts

    def hashed_repr(self) -> str:
        hasher = hashlib.md5()
        hasher.update(self.to_json(filter_out=["rules"]).encode())
        return hasher.hexdigest()[:8]

def to_dim_one_hot(data, out_dim):
    return np.eye(out_dim)[data]


def group_by_lens(
        seqs: List[List[Any]],
        masks: Optional[List[List[int]]] = None
) -> Tuple[List[np.ndarray], Optional[List[List[int]]]]:
    lens = set(len(c) for c in seqs)
    grouped_seqs = []
    grouped_masks = []
    for l in lens:
        grouped_seqs.append(np.array([c for c in seqs if len(c) == l]))
        if masks is not None:
            grouped_masks.append([masks[n] for n, c in enumerate(seqs) if len(c) == l])
    return grouped_seqs, grouped_masks if masks is not None else None


class Experiment:
    def __init__(self, ca: CAReservoir, task: Task, exp_options: ExpOptions = ExpOptions()):
        self.opts = exp_options
        self.ca = ca
        self.scaler = StandardScaler()
        if exp_options.reg_type == RegType.LINEARSVM:
            self.reg = LinearSVC(dual=False, C=1.)
        elif exp_options.reg_type == RegType.RBFSVM:
            self.reg = SVC(kernel="rbf", C=1.)
        elif exp_options.reg_type == RegType.MLP:
            self.reg = MLPClassifier(hidden_layer_sizes=(100, 200, 100))
        elif exp_options.reg_type == RegType.RANDOMFOREST:
            self.reg = RandomForestClassifier(n_estimators=100)

        if exp_options.binarized_task and isinstance(task, TokenTask):
            self.task = BinarizedTask(task)
        elif exp_options.binarized_task:
            raise ValueError("Task cannot be binarized")
        else:
            self.task = task
        tasks, masks = task.generate_tasks(seq_len=exp_options.seq_len,
                                           max_n_seq=exp_options.max_n_seq)

        self.dic = {d: n for n, d in enumerate(self.task.dictionary)}

        # Split into a training and testing set
        split = np.random.permutation(range(len(tasks)))
        train_seqs = [[self.dic[k] for k in tasks[i]]
                      for i in split[:int(.8 * len(tasks))]]
        test_seqs = [[self.dic[k] for k in tasks[i]]
                     for i in split[int(.8 * len(tasks)):]]

        # For masked tasks compute masks
        if masks is not None and not self.opts.ignore_mask:
            train_masks = [masks[i] for i in split[:int(.8 * len(tasks))]]
            test_masks = [masks[i] for i in split[int(.8 * len(tasks)):]]
        else:
            train_masks, test_masks = None, None

        # Group sequences by lengths for batch processing
        self.training_tasks, self.training_masks = group_by_lens(train_seqs, train_masks)
        self.testing_tasks, self.testing_masks = group_by_lens(test_seqs, test_masks)

    @property
    def output_dim(self) -> int:
        return self.task.output_dimension()

    def process_tasks(self, tasks: List[np.ndarray],
                      masks: Optional[List[List[int]]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_data = []
        all_tgts = []
        for l_idx, task_l in enumerate(tasks):
            single_length_data = []
            single_length_tgts = []
            state = np.zeros((task_l.shape[0], self.ca.state_size))
            for t in range(task_l.shape[1] - 1):
                inp = to_dim_one_hot(task_l[:, t], self.output_dim)
                output, state = self.ca(state, inp)
                if masks is not None:
                    for q in range(output.shape[0]):
                        if t + 1 in masks[l_idx][q]:
                            single_length_data.append(output[q:q+1, None, :, :])
                            single_length_tgts.append(task_l[q:q+1, t + 1])
                else:
                    single_length_data.append(output[:, None, :, :])
            if masks is None:
                single_length_tgts = task_l[:, 1:].reshape(-1)
            else:
                single_length_data = np.concatenate(single_length_data, axis=0)
                single_length_tgts = np.concatenate(single_length_tgts, axis=0)
            # print(single_length_data, single_length_tgts)

            all_data.append(np.concatenate(single_length_data, axis=1).reshape(-1, self.ca.output_size))
            all_tgts.append(single_length_tgts)
        return all_data, all_tgts

    def fit(self, return_data=False) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Fit the experiments' regressor on the training and testing data of its task. If
        you set `return_data=True`, the model won't be fitted and the function will return
        the training dataset processed by the CA reservoir as well as the targets.
        """
        all_data, all_tgts = self.process_tasks(self.training_tasks, self.training_masks)

        if return_data:
            inp = [all_data[c].reshape(len(task_l), -1, self.ca.state_size)
                   for c, task_l in enumerate(self.training_tasks)]
            tgts = [all_tgts[c].reshape(len(task_l), -1)
                   for c, task_l in enumerate(self.training_tasks)]
            return inp, tgts
        else:
            # Flatten inp and targets for training of SVM
            self.reg.fit(self.scaler.fit_transform(np.concatenate(all_data, axis=0)),
                         np.concatenate(all_tgts, axis=0))
            return None

    def eval_test(self) -> float:
        all_data, all_tgts = self.process_tasks(self.testing_tasks, self.testing_masks)

        return self.reg.score(self.scaler.transform(np.concatenate(all_data, axis=0)),
                              np.concatenate(all_tgts, axis=0))
