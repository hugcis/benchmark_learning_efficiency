"""
Helper function for running the experiments. Since all experiments will be
run in parallel we use file locking to ensure that each write to the aggregated
result file is atomic.
"""
import argparse
import logging
import os
import pathlib
import pickle as pkl
import random
import sys
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from reservoir_ca.ca_res import CAInput, CAReservoir, CARuleType, rule_array_from_int
from reservoir_ca.esn_res import ESN
from reservoir_ca.experiment import (
    Experiment,
    ExpOptions,
    ProjectionType,
    RegType,
    Reservoir,
)
from reservoir_ca.tasks import Task
from tqdm import tqdm

# This is a very hack-it-yourself way to implement file locking in Python. I
# would prefer a proper library
try:
    # Posix based file locking (Linux, Ubuntu, MacOS, etc.)
    #   Only allows locking on writable files, might cause
    #   strange results for reading.
    import fcntl

    def lock_file(f):
        if f.writable():
            fcntl.lockf(f, fcntl.LOCK_EX)

    def unlock_file(f):
        if f.writable():
            fcntl.lockf(f, fcntl.LOCK_UN)

except ModuleNotFoundError:
    # Windows file locking
    import msvcrts

    def file_size(f):
        return os.path.getsize(os.path.realpath(f.name))

    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_RLCK, file_size(f))

    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_size(f))


# Class for ensuring that all file operations are atomic, treat
# initialization like a standard call to 'open' that happens to be atomic.
class AtomicOpen:
    """
    Open the file with arguments provided by user. Then acquire
    a lock on that file object (WARNING: Advisory locking).

    This file opener *must* be used in a "with" block.
    """

    def __init__(self, path, desc: str, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.file = open(path, desc, *args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs):
        return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        del exc_value, traceback
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        os.fsync(self.file.fileno())
        # Release the lock on the file.
        unlock_file(self.file)
        self.file.close()
        # Handle exceptions that may have come up during execution, by
        # default any exceptions are raised to the user.
        if exc_type is not None:
            return False
        else:
            return True


ENUM_CHOICES = {
    "reg_type": [
        "linearsvm",
        "rbfsvm",
        "linear",
        "rbf",
        "mlp",
        "randomforest",
        "conv",
        "conv_mlp",
        "logistic",
        "sgd",
        "sgd_svm",
    ],
    "proj_type": ["one_to_one", "one_to_many", "one_to_pattern"],
    "ca_rule_type": ["standard", "winput", "winputonce"],
}


def make_parser() -> argparse.ArgumentParser:
    """
    Creates the command line interface. Many of its options are indirectly
    generated from the ExpOptions dataclass.

    Returns:
        an ArgumentParser instance to parse CL arguments.
    """
    parser = argparse.ArgumentParser()
    base_options = ExpOptions()
    opts_dict = asdict(base_options)
    for opt in opts_dict:
        if opt == "rules" or opt == "seed":
            continue
        elif isinstance(opts_dict[opt], Enum):
            choices = ENUM_CHOICES[opt]
            parser.add_argument(
                f"--{opt}", default=choices[0], choices=choices, type=str
            )
        else:
            parser.add_argument(
                f"--{opt}",
                default=vars(base_options).get(opt, None),
                type=type(opts_dict[opt]),
            )

    parser.add_argument("--return-name", default=False, action="store_true")
    parser.add_argument("--increment-data", default=False, action="store_true")
    parser.add_argument("--results-file-name", type=str, default=None)
    parser.add_argument("--rules", nargs="+", default=list(range(256)))
    parser.add_argument("--seed", type=int, default=84923)
    parser.add_argument("--no-write", action="store_true", default=False)
    parser.add_argument("--exp_dirname", default=None, type=str)
    parser.add_argument("--esn_baseline", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser


def atomic_write_to_path(path: pathlib.Path, data: Dict[Any, list[Any]]):
    if path.exists():
        with AtomicOpen(path, "rb+") as f:
            prev = pkl.loads(f.read())
            for r in data:
                prev[r] = prev.get(r, []) + data[r]
            f.seek(0)
            f.write(pkl.dumps(prev))
            f.truncate()
    else:
        with AtomicOpen(path, "wb") as f:
            pkl.dump(data, f)


ResultType = Tuple[Dict[int, list[float]], Optional[Dict[str, Dict[int, list[Any]]]]]


class Result:
    """A class to manage and update results for the experiments.

    The experiments are run separately for different CA rules and stored in a
    dictionary `res`. This dictionary has one list of scores for each rule.

    Each parallel job has a separate `Result` instance which points to the
    results file and hold a temporary copy of the current scores, updated with
    `res.update(score)`.

    When finished, the scores can be flushed in a thread-safe way to the results
    file with `res.save()`.

    """

    res: Dict[int, list[float]]
    res_extra: Dict[str, Dict[int, list[Any]]]

    def __init__(
        self,
        path: pathlib.Path,
        opts_path: pathlib.Path,
        counts_path: pathlib.Path,
        path_extra: Optional[pathlib.Path] = None,
        no_write: bool = False,
    ):
        self.path = path
        self.opts_path = opts_path
        self.counts_path = counts_path
        self.path_extra = path_extra
        self.res = {}
        self.res_extra = {}
        self.no_write = no_write

    def update(self, rule: int, result: float):
        """
        Update the temporary dictionary with a new datapoint `result` for rule `rule`.
        To make these new points persist one must save them with the `save` method.
        """
        if rule not in self.res:
            self.res[rule] = []
        self.res[rule].append(result)

    def update_extra(self, prefix: str, rule: int, result: Any):
        if self.path_extra is None:
            raise TypeError("path_extra cannot be None if updating extra")
        if prefix not in self.res_extra:
            self.res_extra[prefix] = {}
        if rule not in self.res_extra[prefix]:
            self.res_extra[prefix][rule] = []
        self.res_extra[prefix][rule].append(result)

    def save(self) -> ResultType:
        """Flush the current results to disk."""
        added_values: Dict[int, int] = {}
        if not self.no_write:
            for r in self.res:
                added_values[r] = added_values.get(r, 0) + len(self.res[r])
            if not self.counts_path.exists():
                pkl.dump({}, open(self.counts_path, "wb"))

            with AtomicOpen(self.counts_path, "rb+") as f_counts:
                counts_content = f_counts.read()
                if counts_content:
                    ct = pkl.loads(counts_content)
                else:
                    ct = {}
                for r in added_values:
                    ct[r] = ct.get(r, 0) + added_values[r]
                f_counts.seek(0)
                f_counts.write(pkl.dumps(ct))
                f_counts.truncate()

                atomic_write_to_path(self.path, self.res)

        # Flush base results
        ret_res = self.res
        self.res = {}
        # Extra results:
        res_extra = self.save_extra()

        return ret_res, res_extra

    def save_extra(self) -> Optional[Dict[str, Dict[int, Any]]]:
        if self.path_extra is not None and self.res_extra:
            for prefix in self.res_extra:
                if not self.no_write:
                    name = self.path_extra.stem + f"_{prefix}" + self.path_extra.suffix
                    pt = self.path_extra.parent / name
                    atomic_write_to_path(pt, self.res_extra[prefix])
            ret_res_extra = self.res_extra
            self.res_extra = {}

            return ret_res_extra
        else:
            return None

    def read(self) -> Dict[int, int]:
        """Reads and returns the current state of the results dictionary. This
        will not change or flush the current results to disk.

        """
        if self.counts_path.exists():
            with AtomicOpen(self.counts_path, "rb") as f:
                prev = pkl.loads(f.read())
            return prev
        else:
            return {}


InitExp = Tuple[Result, ExpOptions, List[int], Type[Reservoir]]


def init_logging(debug):
    log_level = os.environ.get("LOGLEVEL", "INFO")
    if debug:
        log_level = "DEBUG"
    logging.basicConfig(level=log_level)


def set_opts_from_args(opts: ExpOptions, args: argparse.Namespace) -> ExpOptions:
    for p in vars(opts):
        if p == "reg_type":
            opts.reg_type = RegType.from_str(args.reg_type)
        elif p == "proj_type":
            opts.proj_type = ProjectionType[args.proj_type.upper()]
        elif p == "ca_rule_type":
            opts.ca_rule_type = CARuleType[args.ca_rule_type.upper()]
        elif p in vars(args):
            setattr(opts, p, vars(args)[p])

    return opts


def init_exp(
    name: str,
    opts_extra: Dict[str, Any],
    rules: Optional[list[int]] = None,
    exp_dirname: str = "experiment_results",
) -> InitExp:
    """
    Initialize an experiment. This will read the command line arguments as well as the
    optional extra options and return a Result object, the experiment options as well as
    the list of rules to be processed.
    """

    parser = make_parser()
    args = parser.parse_args()
    init_logging(args.debug)

    opts = ExpOptions()

    if rules is None:
        rules = [int(i) for i in args.rules]
    opts = set_opts_from_args(opts, args)

    # opts_extra overwrites the command line arguments
    for k, v in opts_extra.items():
        if not hasattr(opts, k):
            raise ValueError(f"Wrong option {k} passed to opts.")
        setattr(opts, k, v)

    # Base file name for experiments can be overridden in the CLI options
    if args.results_file_name is not None and "#" in args.results_file_name:
        name = args.results_file_name
    if args.return_name:
        print(name.replace("#", f"_{opts.hashed_repr()}").replace(".pkl", ""))
        sys.exit(0)

    print(opts)
    json_opts = name.replace("#", f"_{opts.hashed_repr()}").replace(".pkl", ".json")
    interm_rep = name.split("#")[0]
    base = pathlib.Path().resolve()
    #
    # Command line experiment dirname overwrites the function argument
    if args.exp_dirname is not None:
        exp_dirname = args.exp_dirname
    out_dir = base / exp_dirname / interm_rep
    out_dir.mkdir(parents=True, exist_ok=True)

    opts_path = out_dir / pathlib.Path(json_opts)
    if not opts_path.exists():
        with open(opts_path, "w") as f:
            f.write(opts.to_json())

    res_fname = name.replace("#", f"_{opts.hashed_repr()}")
    path = out_dir / pathlib.Path(res_fname)

    count_fname = name.replace("#", f"_count_{opts.hashed_repr()}")
    counts_path = out_dir / pathlib.Path(count_fname)

    extra_fname = name.replace("#", f"_extra_{opts.hashed_repr()}")
    extra_path = out_dir / pathlib.Path(extra_fname)

    res = Result(path, opts_path, counts_path, extra_path, no_write=args.no_write)

    # We skip if some of the rules we are processing already have the desired number of
    # experimental results.
    if args.increment_data:
        dic = res.read()
        new_rules = [r for r in rules if dic.get(r, 0) < opts.n_rep]
        skip = list(set(rules).difference(new_rules))
        if skip:
            print("Skipping rules {} for incremental".format(skip))
        rules = new_rules

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)

    ca_class, rules = get_ca_class(args, opts, rules)

    return res, opts, rules, ca_class


def get_ca_class(
    args: argparse.Namespace, opts: ExpOptions, rules: list[int]
) -> Tuple[Reservoir, list[int]]:
    if args.esn_baseline:
        ca_class = ESN
        rules = [-1]
    elif opts.ca_rule_type == CARuleType.STANDARD:
        ca_class = CAReservoir
    elif opts.ca_rule_type in [CARuleType.WINPUT, CARuleType.WINPUTONCE]:
        ca_class = CAInput
    else:
        raise ValueError(f"Incorrect CA rule type {opts.ca_rule_type}")
    return ca_class, rules


def make_ca_reservoir(
    ca_class: Type[CAReservoir], t: int, exp: Experiment, opts: ExpOptions
) -> Reservoir:
    ca = ca_class(
        t,
        exp.task.output_dimension(),
        redundancy=opts.redundancy,
        r_height=opts.r_height,
        proj_factor=opts.proj_factor,
        proj_type=opts.proj_type,
        proj_pattern=opts.proj_pattern,
    )
    if opts.ca_rule_type == CARuleType.WINPUTONCE and isinstance(ca, CAInput):
        ca.use_input_once = True
    return ca


def make_esn_reservoir(
    esn_class: Type[ESN], exp: Experiment, opts: ExpOptions
) -> Reservoir:
    esn = esn_class(
        exp.task.output_dimension(),
        redundancy=opts.redundancy,
        r_height=opts.r_height,
        proj_factor=opts.proj_factor,
    )
    return esn


def run_task(
    task_cls: Type[Task],
    cls_args: List[Any],
    opts_extra: Dict[str, Any] = {},
    fname: Optional[str] = None,
    rules: Optional[List[int]] = None,
) -> ResultType:
    """
    This function runs an experiment specified by its task, options and
    optional name for the output.
    """
    task = task_cls(*cls_args)
    if fname is None:
        fname = task.name + "#.pkl"

    res, opts, rules, ca_class = init_exp(fname, opts_extra, rules=rules)
    if rules:
        for _ in tqdm(range(opts.n_rep), miniters=10):
            exp = Experiment(task, opts)
            for t in rules:
                if ca_class in [CAReservoir, CAInput]:
                    ca = make_ca_reservoir(ca_class, t, exp, opts)
                elif ca_class == ESN:
                    ca = make_esn_reservoir(ca_class, exp, opts)
                else:
                    raise TypeError(f"Unknown ca_class {ca_class}")
                exp.set_ca(ca)
                if opts.reg_type == RegType.SGDCLS:
                    partial_test_results = exp.fit_with_eval()
                    res.update(t, partial_test_results[-1])
                    res.update_extra("tta", t, partial_test_results)
                else:
                    exp.fit()
                    res.update(t, exp.eval_test())
        return res.save()
    return {}, None


class RuleOptimizer:
    def __init__(
        self,
        task_cls: Type[Task],
        cls_args: List[Any],
        opts_extra: Dict[str, Any] = {},
        initial_rule: Optional[int] = None,
    ) -> None:
        self.task_cls = task_cls
        self.cls_args = cls_args
        self.opts_extra = opts_extra

        if initial_rule is None:
            self.initial_rule = rule_array_from_int(
                np.random.randint(256), 1, check_input=True
            )
        else:
            self.initial_rule = rule_array_from_int(initial_rule, 1, check_input=True)

    def mutate_rule(self) -> list[list[int]]:

        return []
