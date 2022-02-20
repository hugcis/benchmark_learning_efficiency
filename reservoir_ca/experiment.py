import logging
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict

import numpy as np

from reservoir_ca.ca_res import CAReservoir, CARuleType, ProjectionType
from reservoir_ca.decoders import (
    SVC,
    ConvClassifier,
    LinearSVC,
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
    SGDCls,
    CLSType,
)
from reservoir_ca.esn_res import ESN
from reservoir_ca.preprocessors import ConvPreprocessor, Preprocessor, ScalePreprocessor
from reservoir_ca.tasks import BinarizedTask, Mask, Task, TokenTask, TaskType


class RegType(Enum):
    """The type of regression used in the reservoir computing outer layer."""

    LINEARSVM = 1
    RBFSVM = 2
    MLP = 3
    RANDOMFOREST = 4
    CONV_MLP = 5
    LOGISTICREG = 6
    SGDCLS = 7
    ADAMCLS = 8

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
        elif label in ("adam", "adam_cls"):
            return RegType.ADAMCLS
        else:
            raise NotImplementedError

    def __str__(self):
        return f"{self.name}"


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
    pretrain_for: int = 0
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


def to_dim_one_hot(data: np.ndarray, out_dim: int) -> np.ndarray:
    return np.eye(out_dim)[data]


GroupedMasks = List[List[List[int]]]
NumTaskType = List[List[int]]


def group_by_lens(
    seqs: NumTaskType, masks: Mask = None
) -> Tuple[List[np.ndarray], Optional[GroupedMasks]]:
    lens = set(len(c) for c in seqs)
    grouped_seqs = []
    grouped_masks = []
    for length in lens:
        grouped_seqs.append(np.array([c for c in seqs if len(c) == length]))
        if masks is not None:
            grouped_masks.append(
                [masks[n] for n, c in enumerate(seqs) if len(c) == length]
            )
    return grouped_seqs, grouped_masks if masks is not None else None


Reservoir = Union[CAReservoir, ESN]


def convert_task_to_num(tasks: TaskType, dic: Dict[str, int]) -> NumTaskType:
    return [[dic[k] for k in task] for task in tasks]


def get_train_test_split(
    tasks: TaskType, masks: Mask, dic: Dict[str, int], ignore_mask: bool
) -> Tuple[NumTaskType, Mask, NumTaskType, Mask]:
    # Split into a training and testing set
    split = np.random.permutation(range(len(tasks)))
    num_tasks = convert_task_to_num(tasks, dic)
    train_seqs = [num_tasks[i] for i in split[: int(0.8 * len(tasks))]]
    test_seqs = [num_tasks[i] for i in split[int(0.8 * len(tasks)) :]]

    # For masked tasks compute masks
    if masks is not None and not ignore_mask:
        train_masks = [masks[i] for i in split[: int(0.8 * len(tasks))]]
        test_masks = [masks[i] for i in split[int(0.8 * len(tasks)) :]]
    else:
        train_masks, test_masks = None, None

    return train_seqs, train_masks, test_seqs, test_masks


class Experiment:
    """An experiment with a reservoir and a task."""

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
            self.set_reservoir(ca)
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
            self.reg = SGDCls(cls_type=CLSType.SGD)
        elif exp_options.reg_type == RegType.ADAMCLS:
            self.reg = SGDCls(cls_type=CLSType.ADAM)
        else:
            raise ValueError(f"Unknown regression type {exp_options.reg_type}")

        self.preproc = None

        if exp_options.binarized_task and isinstance(task, TokenTask):
            self.task = BinarizedTask(task)
        elif exp_options.binarized_task:
            raise ValueError("Task cannot be binarized")
        else:
            self.task = task

        # Generate the tasks
        tasks, masks = self.task.generate_tasks(
            seq_len=exp_options.seq_len, max_n_seq=exp_options.max_n_seq
        )

        self.dic = {d: n for n, d in enumerate(self.task.dictionary)}

        # Split tasks into training/testing
        train_seqs, train_masks, test_seqs, test_masks = get_train_test_split(
            tasks, masks, self.dic, self.opts.ignore_mask
        )

        # Group sequences by lengths for batch processing
        self.training_tasks, self.training_masks = group_by_lens(
            train_seqs, train_masks
        )
        self.testing_tasks, self.testing_masks = group_by_lens(test_seqs, test_masks)
        self.shuffle = True

        # Pretraining options
        if exp_options.pretrain_for > 0:
            self.pretrain_tasks, _ = group_by_lens(
                convert_task_to_num(
                    self.task.generate_tasks(
                        seq_len=exp_options.seq_len, max_n_seq=exp_options.pretrain_for
                    ),
                    self.dic,
                )
            )
        else:
            self.pretrain_tasks = None

    @property
    def output_dim(self) -> int:
        return self.task.output_dimension()

    def set_reservoir(self, ca: Reservoir):
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
            raise ValueError("Must initialize reservoir with method `set_reservoir`")
        else:
            return self.ca, self.preproc

    def process_tasks(
        self, tasks: List[np.ndarray], masks: Optional[GroupedMasks] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        ca, _ = self.check_ca()
        all_data = []
        all_tgts = []
        # Initialize the CA state
        state = np.zeros((1, ca.state_size), dtype=int)

        # "Pretraining"
        if self.pretrain_tasks is not None:
            for l_idx, task_l in enumerate(self.pretrain_tasks):
                for b in range(task_l.shape[0]):
                    for t in range(task_l.shape[1] - 1):
                        inp = to_dim_one_hot(task_l[b : b + 1, t], self.output_dim)
                        _, state = ca(state, inp)

        # We loop across all the length groupings found
        for l_idx, task_l in enumerate(tasks):
            logging.debug(
                "Processing %d tasks of length %d", task_l.shape[0], task_l.shape[1]
            )
            for b in range(task_l.shape[0]):
                single_length_data = []
                single_length_tgts = []
                # task_l is of dimension `batch x (length of sequence)`
                for t in range(task_l.shape[1] - 1):
                    inp = to_dim_one_hot(task_l[b : b + 1, t], self.output_dim)

                    output, state = ca(state, inp)  # The CA outputs a vector of
                    # size (batch x r_height x state_size)

                    if masks is not None:
                        # Apply masking separatly for each item of the batch
                        for q in range(output.shape[0]):
                            if t + 1 in masks[l_idx][q]:
                                single_length_data.append(output[q : q + 1, None, :, :])
                                single_length_tgts.append(task_l[q : q + 1, t + 1])
                    else:
                        # Just add the whole output if not masking
                        single_length_data.append(output[:, None, :, :])

                if masks is None:
                    single_length_data_arr = np.concatenate(single_length_data, axis=1)
                    single_length_tgts_arr = task_l[b, 1:]
                else:
                    single_length_data_arr = np.concatenate(single_length_data, axis=0)
                    single_length_tgts_arr = np.concatenate(single_length_tgts, axis=0)

                all_data.append(single_length_data_arr)
                all_tgts.append(single_length_tgts_arr)

        # all_data is a list of vectors of shape (n_example x 1 x r_height x
        # state_size) if masked and (n_example x length_grp x r_height x
        # state_size) if not masked
        logging.debug(
            "Processed all %d examples", sum(i.shape[0] * i.shape[1] for i in all_data)
        )
        return all_data, all_tgts

    def shape_for_preproc(self, all_data: list[np.ndarray]) -> np.ndarray:
        dt_list = [c.reshape(-1, c.shape[-2], c.shape[-1]) for c in all_data]
        return np.concatenate(dt_list, axis=0)

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
                preproc.fit_transform(self.shape_for_preproc(all_data)),
                np.concatenate([c.reshape(-1) for c in all_tgts], axis=0),
            )
            return None

    def predict_test(
        self, return_target: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Output the test set predictions for that experiment."""
        _, preproc = self.check_ca()
        all_data, all_tgts = self.process_tasks(self.testing_tasks, self.testing_masks)
        prediction = self.reg.predict(
            preproc.transform(self.shape_for_preproc(all_data))
        )
        if return_target:
            return (
                prediction,
                np.concatenate([c.reshape(-1) for c in all_tgts], axis=0),
            )
        else:
            return prediction

    def eval_test(self) -> float:
        """Output the test set scores for that experiment, as computed by the
        regressor's score function.

        """
        _, preproc = self.check_ca()
        all_data, all_tgts = self.process_tasks(self.testing_tasks, self.testing_masks)

        return self.reg.score(
            preproc.transform(np.concatenate(all_data, axis=0)),
            np.concatenate([c.reshape(-1) for c in all_tgts], axis=0),
        )

    def fit_with_eval(self) -> list[float]:
        """Fit the regressor with evaluation checkpoints and return eval values."""
        logging.debug("Fitting a model with periodic evaluations")
        if isinstance(self.reg, SGDCls):
            _, preproc = self.check_ca()
            all_data, all_tgts = self.process_tasks(
                self.training_tasks, self.training_masks
            )

            if self.shuffle:
                all_data_arr = self.shape_for_preproc(all_data)
                shuffle_index = np.random.permutation(all_data_arr.shape[0])
                all_data_arr = all_data_arr[shuffle_index, :]
                all_tgts = np.concatenate([c.reshape(-1) for c in all_tgts], axis=0)[
                    shuffle_index
                ]
            else:
                all_data_arr = self.shape_for_preproc(all_data)
                all_tgts = np.concatenate([c.reshape(-1) for c in all_tgts], axis=0)

            all_data_test, all_tgts_test = self.process_tasks(
                self.testing_tasks, self.testing_masks
            )
            # At this point self.reg can only be of SGD type
            self.reg.fit(
                preproc.fit_transform(all_data_arr),
                all_tgts,
                X_t=preproc.transform(self.shape_for_preproc(all_data_test)),
                y_t=np.concatenate([c.reshape(-1) for c in all_tgts_test], axis=0),
            )
        else:
            raise ValueError("Regressor type should be SGD to evaluate during fit")

        return self.reg.test_values
