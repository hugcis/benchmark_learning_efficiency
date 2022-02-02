import hashlib
import json
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

import numpy as np

from reservoir_ca.tasks import BinarizedTask, Task, TokenTask, Mask
from reservoir_ca.esn_res import ESN
from reservoir_ca.ca_res import CAReservoir, CARuleType, ProjectionType
from reservoir_ca.decoders import (
    LinearSVC,
    SVC,
    SGDCls,
    StandardScaler,
    MLPClassifier,
    RandomForestClassifier,
    ConvClassifier,
    LogisticRegression,
)


class RegType(Enum):
    """The type of regression used in the reservoir computing outer layer."""

    LINEARSVM = 1
    RBFSVM = 2
    MLP = 3
    RANDOMFOREST = 4
    CONV_MLP = 5
    LOGISTICREG = 6
    SGDCLS = 7

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
        elif label in ("conv", "conv_mlp"):
            return RegType.CONV_MLP
        elif label in ("logistic", "logistic_reg"):
            return RegType.LOGISTICREG
        elif label in ("sgd", "sgd_svm"):
            return RegType.SGDCLS
        else:
            raise NotImplementedError

    def __str__(self):
        return "%s" % self.name


@dataclass
class ExpOptions:
    """An option class that holds all of the experiment parameters."""

    seq_len: int = 100
    max_n_seq: int = 300
    n_rep: int = 10
    seed: int = 0
    redundancy: int = 4
    r_height: int = 2
    proj_factor: int = 40
    reg_type: RegType = RegType.LINEARSVM
    ignore_mask: bool = False
    binarized_task: bool = False
    proj_type: ProjectionType = ProjectionType.ONE_TO_ONE
    proj_pattern: int = 4
    ca_rule_type: CARuleType = CARuleType.STANDARD

    def to_json(self, filter_out: Optional[List[str]] = ["seed"]):
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
        hasher.update(self.to_json().encode())
        return hasher.hexdigest()[:8]


def to_dim_one_hot(data, out_dim):
    return np.eye(out_dim)[data]


GroupedMasks = List[List[List[int]]]
NumTaskType = List[List[int]]


def group_by_lens(
    seqs: NumTaskType, masks: Mask = None
) -> Tuple[List[np.ndarray], Optional[GroupedMasks]]:
    lens = set(len(c) for c in seqs)
    grouped_seqs = []
    grouped_masks = []
    for l in lens:
        grouped_seqs.append(np.array([c for c in seqs if len(c) == l]))
        if masks is not None:
            grouped_masks.append([masks[n] for n, c in enumerate(seqs) if len(c) == l])
    return grouped_seqs, grouped_masks if masks is not None else None


class Preprocessor(ABC):
    @abstractmethod
    def fit_transform(self, X):
        del X

    @abstractmethod
    def transform(self, X):
        del X


class ConvPreprocessor(Preprocessor):
    def __init__(self, r_height: int, state_size: int):
        self.r_height = r_height
        self.state_size = state_size

    def reshape_vec(self, X: List[np.ndarray]):
        return [v.reshape(-1, self.r_height, self.state_size) for v in X]

    def fit(self, X):
        return np.concatenate(self.reshape_vec(X), axis=0)

    def transform(self, X):
        return np.concatenate(self.reshape_vec(X), axis=0)

    def fit_transform(self, X):
        return np.concatenate(self.reshape_vec(X), axis=0)


class ScalePreprocessor(Preprocessor):
    def __init__(self, output_size):
        self.output_size = output_size
        self.scaler = StandardScaler()

    def fit(self, X):
        return self.scaler.fit(np.concatenate(X, axis=1).reshape(-1, self.output_size))

    def transform(self, X):
        return self.scaler.transform(
            np.concatenate(X, axis=1).reshape(-1, self.output_size)
        )

    def fit_transform(self, X):
        return self.scaler.fit_transform(
            np.concatenate(X, axis=1).reshape(-1, self.output_size)
        )


Reservoir = Union[CAReservoir, ESN]


class Experiment:
    ca: Optional[Reservoir] = None
    preproc: Optional[Preprocessor] = None
    task: Task

    def __init__(
        self,
        task: Task,
        exp_options: ExpOptions = ExpOptions(),
        ca: Optional[Reservoir] = None,
    ):
        self.opts = exp_options
        if ca is not None:
            self.set_ca(ca)
        if exp_options.reg_type == RegType.LINEARSVM:
            self.reg = LinearSVC(dual=False, C=1.0, max_iter=100)
        elif exp_options.reg_type == RegType.RBFSVM:
            self.reg = SVC(kernel="rbf", C=1.0)
        elif exp_options.reg_type == RegType.MLP:
            self.reg = MLPClassifier(hidden_layer_sizes=(100, 200, 100))
        elif exp_options.reg_type == RegType.RANDOMFOREST:
            self.reg = RandomForestClassifier(n_estimators=100)
        elif exp_options.reg_type == RegType.CONV_MLP:
            self.reg = ConvClassifier((32, 16))
        elif exp_options.reg_type == RegType.LOGISTICREG:
            self.reg = LogisticRegression(C=1.0, solver="liblinear")
        elif exp_options.reg_type == RegType.SGDCLS:
            self.reg = SGDCls()

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

        # Split into a training and testing set
        split = np.random.permutation(range(len(tasks)))
        train_seqs = [
            [self.dic[k] for k in tasks[i]] for i in split[: int(0.8 * len(tasks))]
        ]
        test_seqs = [
            [self.dic[k] for k in tasks[i]] for i in split[int(0.8 * len(tasks)) :]
        ]

        # For masked tasks compute masks
        if masks is not None and not self.opts.ignore_mask:
            train_masks = [masks[i] for i in split[: int(0.8 * len(tasks))]]
            test_masks = [masks[i] for i in split[int(0.8 * len(tasks)) :]]
        else:
            train_masks, test_masks = None, None

        # Group sequences by lengths for batch processing
        self.training_tasks, self.training_masks = group_by_lens(
            train_seqs, train_masks
        )
        self.testing_tasks, self.testing_masks = group_by_lens(test_seqs, test_masks)

    @property
    def output_dim(self) -> int:
        return self.task.output_dimension()

    def set_ca(self, ca: Reservoir):
        if self.ca is not None:
            current_size = self.ca.state_size
        else:
            current_size = None
        self.ca = ca
        assert self.ca is not None
        if current_size is None or self.ca.state_size != current_size:
            if self.opts.reg_type == RegType.CONV_MLP:
                self.preproc = ConvPreprocessor(self.opts.r_height, self.ca.state_size)
            else:
                self.preproc = ScalePreprocessor(self.ca.output_size)

    def check_ca(self) -> Tuple[Reservoir, Preprocessor]:
        if self.ca is None or self.preproc is None:
            raise ValueError("Must initialize ca with method `set_ca`")
        else:
            return self.ca, self.preproc

    def process_tasks(
        self, tasks: List[np.ndarray], masks: Optional[GroupedMasks] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ca, _ = self.check_ca()
        all_data = []
        all_tgts = []
        for l_idx, task_l in enumerate(tasks):
            single_length_data = []
            single_length_tgts = []
            state = np.zeros((task_l.shape[0], ca.state_size), dtype=int)
            for t in range(task_l.shape[1] - 1):
                inp = to_dim_one_hot(task_l[:, t], self.output_dim)
                output, state = ca(state, inp)
                if masks is not None:
                    for q in range(output.shape[0]):
                        if t + 1 in masks[l_idx][q]:
                            single_length_data.append(output[q : q + 1, None, :, :])
                            single_length_tgts.append(task_l[q : q + 1, t + 1])
                else:
                    single_length_data.append(output[:, None, :, :])
            if masks is None:
                single_length_tgts_arr = task_l[:, 1:].reshape(-1)
            else:
                single_length_data = np.concatenate(single_length_data, axis=0)
                single_length_tgts_arr = np.concatenate(single_length_tgts, axis=0)

            all_data.append(np.concatenate(single_length_data, axis=1))
            all_tgts.append(single_length_tgts_arr)
        return all_data, all_tgts

    def fit(
        self, return_data=False
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """Fit the experiments' regressor on the training and testing data of its task.

        If you set `return_data=True`, the model won't be fitted and the
        function will return the training dataset processed by the CA reservoir
        as well as the targets.
        """
        ca, preproc = self.check_ca()
        all_data, all_tgts = self.process_tasks(
            self.training_tasks, self.training_masks
        )

        if return_data:
            inp = [
                all_data[c].reshape(len(task_l), -1, ca.state_size)
                for c, task_l in enumerate(self.training_tasks)
            ]
            tgts = [
                all_tgts[c].reshape(len(task_l), -1)
                for c, task_l in enumerate(self.training_tasks)
            ]
            return inp, tgts
        else:
            # Flatten inp and targets for training of SVM
            self.reg.fit(
                preproc.fit_transform(all_data), np.concatenate(all_tgts, axis=0)
            )
            return None

    def predict_test(self) -> np.ndarray:
        _, preproc = self.check_ca()
        all_data, all_tgts = self.process_tasks(self.testing_tasks, self.testing_masks)

        return self.reg.predict(preproc.transform(all_data))

    def eval_test(self) -> float:
        _, preproc = self.check_ca()
        all_data, all_tgts = self.process_tasks(self.testing_tasks, self.testing_masks)

        return self.reg.score(
            preproc.transform(all_data), np.concatenate(all_tgts, axis=0)
        )

    def fit_with_eval(self):
        if isinstance(self.reg, SGDCls):
            _, preproc = self.check_ca()
            all_data, all_tgts = self.process_tasks(
                self.training_tasks, self.training_masks
            )
            all_data_test, all_tgts_test = self.process_tasks(
                self.testing_tasks, self.testing_masks
            )
            # At this point self.reg can only be of SGD type
            self.reg.fit(
                preproc.fit_transform(all_data),
                np.concatenate(all_tgts, axis=0),
                X_t=preproc.transform(all_data_test),
                y_t=np.concatenate(all_tgts_test, axis=0),
            )
        else:
            raise ValueError("Regressor type should be SGD")

        return self.reg.test_values
