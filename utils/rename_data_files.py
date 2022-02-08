"""
A small tool to rename all data files with the correct hash when the
Experiment object gets changed and the hashes changes too.

"""
import pathlib
import sys

from utils.exp_utils import list_all_exps

if __name__ == "__main__":
    exp_dir = pathlib.Path(sys.argv[1])
    assert exp_dir.exists()
    opts_dict, opts_hashes_dict = list_all_exps(exp_dir)
    for op_key, op in opts_dict.items():
        stem, hsh_k = op_key.split("_")
        base = exp_dir / stem
        if hsh_k != op.hashed_repr():
            opt_path = base / f"{stem}_{hsh_k}.json"
            data_path = base / f"{stem}_{hsh_k}.pkl"
            ct_path = base / f"{stem}_count_{hsh_k}.pkl"
            extra_path = base / f"{stem}_extra_{hsh_k}_tta.pkl"

            if not (opt_path.exists() and data_path.exists() and ct_path.exists()):
                print(
                    "Err missing file",
                    hsh_k,
                    op.hashed_repr(),
                    opt_path.exists(),
                    data_path.exists(),
                    ct_path.exists(),
                )
                continue
            new_opt_path = base / f"{stem}_{op.hashed_repr()}.json"
            new_data_path = base / f"{stem}_{op.hashed_repr()}.pkl"
            new_ct_path = base / f"{stem}_count_{op.hashed_repr()}.pkl"
            new_extra_path = base / f"{stem}_extra_{op.hashed_repr()}_tta.pkl"
            if new_opt_path.exists() or new_data_path.exists() or new_ct_path.exists():
                print("Target path already exists")
                continue
            opt_path.rename(new_opt_path)
            data_path.rename(new_data_path)
            ct_path.rename(new_ct_path)
            if extra_path.exists() and not new_extra_path.exists():
                extra_path.rename(new_extra_path)
