"""The result saver class and functions"""
import logging
import os
import pickle as pkl
import pathlib
from typing import Dict, Optional, Tuple, Any

ResultType = Tuple[Dict[int, list[float]], Optional[Dict[str, Dict[int, list[Any]]]]]

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
            for r, res_value in self.res.items():
                added_values[r] = added_values.get(r, 0) + len(res_value)
            if not self.counts_path.exists():
                with open(self.counts_path, "wb") as count_file:
                    pkl.dump({}, count_file)

            with AtomicOpen(self.counts_path, "rb+") as f_counts:
                counts_content = f_counts.read()
                if counts_content:
                    ct = pkl.loads(counts_content)
                else:
                    ct = {}
                for r, added_val in added_values.items():
                    ct[r] = ct.get(r, 0) + added_val
                f_counts.seek(0)
                f_counts.write(pkl.dumps(ct))
                f_counts.truncate()

                atomic_write_to_path(self.path, self.res)

        # Flush base results
        ret_res = self.res
        logging.info("Saving result for rules %s", ", ".join(str(i) for i in ret_res))
        self.res = {}
        # Extra results:
        res_extra = self.save_extra()

        return ret_res, res_extra

    def save_extra(self) -> Optional[Dict[str, Dict[int, Any]]]:
        if self.path_extra is not None and self.res_extra:
            for prefix, subdict in self.res_extra.items():
                if not self.no_write:
                    name = self.path_extra.stem + f"_{prefix}" + self.path_extra.suffix
                    pt = self.path_extra.parent / name
                    atomic_write_to_path(pt, subdict)
                    logging.info("Saving extra results to %s", pt)
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
