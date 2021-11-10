from dataclasses import asdict
import random
import pickle as pkl
import pathlib
import argparse
from typing import Dict, Tuple

import numpy as np

from reservoir_ca.experiment import ExpOptions, NAME_TO_REG_TYPE, RegType

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
    parser = argparse.ArgumentParser()
    base_options = ExpOptions()
    opts_dict = asdict(base_options)
    for opt in opts_dict:
        if opt == "rules" or opt == "seed" or opt == "reg_type":
            continue
        parser.add_argument(f"--{opt}", default=vars(base_options).get(opt, None),
                            type=type(opts_dict[opt]))

    parser.add_argument("--results-file-name", type=str, default=None)
    parser.add_argument("--rules", nargs="+", default=list(range(256)))
    parser.add_argument("--seed", type=int, default=84923)
    parser.add_argument("--reg-type", type=str, default="linearsvm",
                        choices=["linearsvm", "rbfsvm", "linear", "rbf"])
    return parser


class Result:
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


def init_exp(name: str) -> Tuple[Result, ExpOptions]:
    parser = make_parser()
    args = parser.parse_args()
    opts = ExpOptions(rules=[int(i) for i in args.rules])
    for p in vars(opts):
        if p != "reg_type" and p != "rules" and p in vars(args):
            setattr(opts, p, vars(args)[p])
        elif p == "reg_type":
            opts.reg_type = RegType.from_str(args.reg_type)

    # Base file name for experiments can be overridden in the CLI options
    if args.results_file_name is not None and "#" in args.results_file_name:
        name = args.results_file_name
    name = name.replace("#", f"_{opts.redundancy}")
    base = pathlib.Path().resolve()
    path = base / "experiment_results" / pathlib.Path(name)
    res = Result(path)

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)

    return res, opts
