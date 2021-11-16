"""
Helper function for running the experiments. Since all experiments will be
run in parallel we use file locking to ensure that each write to the aggregated
result file is atomic.
"""
import sys
from dataclasses import asdict
import random
import pickle as pkl
import pathlib
import argparse
from typing import Dict, Tuple, Type, List, Any, Optional

import numpy as np
from tqdm import tqdm

from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.tasks import Task
from reservoir_ca.experiment import ExpOptions, Experiment, ProjectionType, RegType

try:
    # Posix based file locking (Linux, Ubuntu, MacOS, etc.)
    #   Only allows locking on writable files, might cause
    #   strange results for reading.
    import fcntl, os
    def lock_file(f):
        if f.writable(): fcntl.lockf(f, fcntl.LOCK_EX)
    def unlock_file(f):
        if f.writable(): fcntl.lockf(f, fcntl.LOCK_UN)
except ModuleNotFoundError:
    # Windows file locking
    import msvcrt, os
    def file_size(f):
        return os.path.getsize( os.path.realpath(f.name) )
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_RLCK, file_size(f))
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_size(f))


# Class for ensuring that all file operations are atomic, treat
# initialization like a standard call to 'open' that happens to be atomic.
# This file opener *must* be used in a "with" block.
class AtomicOpen:
    # Open the file with arguments provided by user. Then acquire
    # a lock on that file object (WARNING: Advisory locking).
    def __init__(self, path, desc: str, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.file = open(path, desc, *args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs): return self.file

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
        if (exc_type != None): return False
        else:                  return True


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
        if opt == "rules" or opt == "seed" or opt == "reg_type" or opt == "proj_type":
            continue
        parser.add_argument(f"--{opt}", default=vars(base_options).get(opt, None),
                            type=type(opts_dict[opt]))

    parser.add_argument("--return-name", default=False, action="store_true")
    parser.add_argument("--results-file-name", type=str, default=None)
    parser.add_argument("--rules", nargs="+", default=list(range(256)))
    parser.add_argument("--seed", type=int, default=84923)
    parser.add_argument("--reg-type", type=str, default="linearsvm",
                        choices=["linearsvm", "rbfsvm", "linear", "rbf", "mlp",
                                 "randomforest"])
    parser.add_argument("--proj_type", type=str, default="one_to_one",
                        choices=["one_to_one", "one_to_many", "one_to_pattern"])
    return parser


class Result:
    """
    A class to manage and update results for the experiments. The experiments are
    run separately for different CA rules and stored in a dictionary `res`. This
    dictionary has one list of scores for each rule.

    Each parallel job has a separate `Result` instance which points to the results
    file and hold a temporary copy of the current scores, updated with `res.update(score)`.

    When finished, the scores can be flushed in a thread-safe way to the results file with
    `res.save()`.
    """
    res: Dict[int, list[float]]

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.res = {}

    def update(self, rule: int, result: float):
        if rule not in self.res:
            self.res[rule] = []
        self.res[rule].append(result)

    def save(self):
        if self.path.exists():
            with AtomicOpen(self.path, "rb+") as f:
                prev = pkl.loads(f.read())
                for r in self.res:
                    prev[r] = prev.get(r, []) + self.res[r]
                f.seek(0)
                f.write(pkl.dumps(prev))
                f.truncate()
        else:
            with AtomicOpen(self.path, "wb") as f:
                pkl.dump(self.res, f)
        # Flush results
        self.res = {}


def init_exp(name: str, opts_extra: Dict[str, Any]) -> Tuple[Result, ExpOptions, List[int]]:
    parser = make_parser()
    args = parser.parse_args()
    opts = ExpOptions()
    rules = [int(i) for i in args.rules]
    for p in vars(opts):
        if p != "proj_type" and p != "reg_type" and p in vars(args):
            setattr(opts, p, vars(args)[p])
        elif p == "reg_type":
            opts.reg_type = RegType.from_str(args.reg_type)
        elif p == "proj_type":
            opts.proj_type = ProjectionType[args.proj_type.upper()]

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
    base = pathlib.Path().resolve()

    opts_path = base / "experiment_results" / pathlib.Path(json_opts)
    with open(opts_path, "w") as f:
        f.write(opts.to_json())

    name = name.replace("#", f"_{opts.hashed_repr()}")
    path = base / "experiment_results" / pathlib.Path(name)
    res = Result(path)

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)

    return res, opts, rules


def run_task(task_cls: Type[Task], cls_args: List[Any],
             opts_extra: Dict[str, Any] = {},
             fname: Optional[str] = None):
    task = task_cls(*cls_args)
    if fname is None:
        fname = task.name + "#.pkl"

    res, opts, rules = init_exp(fname, opts_extra)
    for _ in tqdm(range(opts.n_rep), miniters=10):
        ca = CAReservoir(0, task.output_dimension(),
                         r_height=opts.r_height,
                         proj_factor=opts.proj_factor)
        exp = Experiment(ca, task, opts)
        for t in rules:
            ca = CAReservoir(t, task.output_dimension(),
                             redundancy=opts.redundancy,
                             r_height=opts.r_height,
                             proj_factor=opts.proj_factor,
                             proj_type=opts.proj_type,
                             proj_pattern=opts.proj_pattern)
            exp.ca = ca
            exp.fit()
            res.update(t, exp.eval_test())
    res.save()
