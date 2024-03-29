import pickle as pkl
import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import typing
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from reservoir_ca.decoders import (
    SVC,
    CLSType,
    ConvClassifier,
    LinearSVC,
    LogisticRegression,
    MLPClassifier,
    RandomForestClassifier,
    SGDCls,
)
from reservoir_ca.preprocessors import ConvPreprocessor, Preprocessor, ScalePreprocessor
from reservoir_ca.reservoir import CARuleType, ProjectionType, Reservoir, RState
from reservoir_ca.reservoir.ca_res import CAReservoir
from reservoir_ca.tasks import BinarizedTask, Mask, Task, TaskType, TokenTask


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
    feedback_state: bool = False

    def to_json(self, filter_out: Optional[Iterable[str]] = None):
        dict_rep = dataclasses.asdict(self)
        if filter_out is None:
            filter_out = ["seed"]
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


def convert_single_task_to_num(task: List[str], dic: Dict[str, int]) -> List[int]:
    return [dic[k] for k in task]


def convert_task_to_num(tasks: TaskType, dic: Dict[str, int]) -> NumTaskType:
    return [convert_single_task_to_num(task, dic) for task in tasks]


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


def compute_single_length(
    idx: int,
    task_l: np.ndarray,
    output_dim: int,
    ca: Reservoir,
    state: RState,
    masks: Optional[GroupedMasks],
    batched: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, RState]:
    single_length_data = []
    single_length_tgts = []
    # task_l is of dimension `batch x (length of sequence)`
    for t in range(task_l.shape[1] - 1):
        if batched is not None:
            inp_base = task_l[batched : batched + 1, t]
        else:
            inp_base = task_l[:, t]
        inp = to_dim_one_hot(inp_base, output_dim)

        output, state = ca(state, inp)  # The CA outputs a vector of
        # size (batch x r_height x state_size)

        if masks is not None:
            # Apply masking separatly for each item of the batch
            for q in range(output.shape[0]):
                if t + 1 in masks[idx][q]:
                    single_length_data.append(output[q : q + 1, None, :, :])
                    single_length_tgts.append(task_l[q : q + 1, t + 1])
        else:
            # Just add the whole output if not masking
            single_length_data.append(output[:, None, :, :])

    if masks is None:
        single_length_data_arr = np.concatenate(single_length_data, axis=1)
        single_length_tgts_arr = task_l[batched, 1:]
    else:
        single_length_data_arr = np.concatenate(single_length_data, axis=0)
        single_length_tgts_arr = np.concatenate(single_length_tgts, axis=0)
    return single_length_data_arr, single_length_tgts_arr, state


class Experiment:
    """An experiment with a reservoir and a task."""

    ca: Optional[Reservoir] = None
    preproc: Optional[Preprocessor] = None
    task: Task
    pretrained_state: Optional[RState]

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
        self.pretrain_for = exp_options.pretrain_for
        self.pretrained_state = None

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
        state = RState(np.zeros((1, ca.state_size), dtype=int))

        # "Pretraining"
        if self.pretrain_for > 0 and self.pretrained_state is None:
            logging.debug("Doing pretraining for %d steps", self.pretrain_for)
            # We use the generator version since it is designed to be used with
            # large numbers that couldn't fit into memory
            for task, _ in self.task.generate_tasks_generator(
                max_n_seq=self.pretrain_for
            ):
                inp = to_dim_one_hot(
                    np.array(convert_single_task_to_num(task, self.dic)),
                    self.output_dim,
                )
                for step in inp:
                    _, state = ca(state, step[None, :])
            self.pretrained_state = RState(state[:])
        elif self.pretrained_state is not None:
            state = RState(self.pretrained_state[:])

        # We loop across all the length groupings found
        for l_idx, task_l in enumerate(tasks):
            logging.debug(
                "Processing %d tasks of length %d", task_l.shape[0], task_l.shape[1]
            )
            # Pretraining implies feeback_state = True
            if self.opts.feedback_state or self.pretrain_for > 0:
                # We need to step through the batch in sequence
                for b in range(task_l.shape[0]):
                    (
                        single_length_data_arr,
                        single_length_tgts_arr,
                        state,
                    ) = compute_single_length(
                        l_idx, task_l, self.output_dim, ca, state, masks, batched=b
                    )
                    all_data.append(single_length_data_arr)
                    all_tgts.append(single_length_tgts_arr)
            else:
                # Initialize the CA state every step
                state = RState(np.zeros((task_l.shape[0], ca.state_size), dtype=int))
                (
                    single_length_data_arr,
                    single_length_tgts_arr,
                    state,
                ) = compute_single_length(
                    l_idx, task_l, self.output_dim, ca, state, masks
                )
                all_data.append(single_length_data_arr)
                all_tgts.append(single_length_tgts_arr)

        # all_data is a list of vectors of shape (n_example x 1 x r_height x
        # state_size) if masked and (n_example x length_grp x r_height x
        # state_size) if not masked
        logging.debug(
            "Processed all %d examples", sum(i.shape[0] * i.shape[1] for i in all_data)
        )
        return all_data, all_tgts

    def shape_for_preproc(self, all_data: Iterable[np.ndarray]) -> np.ndarray:
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
        prediction = typing.cast(
            np.ndarray,
            self.reg.predict(preproc.transform(self.shape_for_preproc(all_data))),
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
        score = self.reg.score(
            preproc.transform(np.concatenate(all_data, axis=0)),
            np.concatenate([c.reshape(-1) for c in all_tgts], axis=0),
        )
        return typing.cast(float, score)

    def fit_with_eval(self) -> List[float]:
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

    def save_params(self, name: Path):
        if not isinstance(self.reg, SGDCls):
            raise ValueError("Parameter saving is only possible with the SGD regressor")
        reg_params = self.reg.params()
        params_obj: Dict[str, np.ndarray] = {
            "out.weight": reg_params[0],
            "out.bias": reg_params[1],
        }
        if isinstance(self.ca, CAReservoir):
            params_obj.update(self.ca.params())

        with open(name, "wb") as f:
            pkl.dump(params_obj, f)
