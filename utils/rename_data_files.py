import pathlib
import sys

from utils.exp_utils import list_all_exps

if __name__ == "__main__":
    exp_dir = pathlib.Path(sys.argv[1])
    assert exp_dir.exists()
    opts_dict, opts_hashes_dict = list_all_exps(exp_dir)
    for op_key, op in opts_dict.items():
        stem, hsh_k = op_key.split("_")
        if hsh_k != op.hashed_repr():
            opt_path = pathlib.Path(exp_dir / stem / f"{stem}_{hsh_k}.json")
            data_path = pathlib.Path(exp_dir / stem / f"{stem}_{hsh_k}.pkl")
            ct_path = pathlib.Path(exp_dir / stem / f"{stem}_count_{hsh_k}.pkl")
            if not (opt_path.exists() and data_path.exists() and ct_path.exists()):
                print("Err missing file", hsh_k, op.hashed_repr(), opt_path.exists(),
                      data_path.exists(), ct_path.exists())
                continue
            new_opt_path = pathlib.Path(exp_dir / stem / f"{stem}_{op.hashed_repr()}.json")
            new_data_path = pathlib.Path(exp_dir / stem / f"{stem}_{op.hashed_repr()}.pkl")
            new_ct_path = pathlib.Path(exp_dir / stem / f"{stem}_count_{op.hashed_repr()}.pkl")
            if new_opt_path.exists() or new_data_path.exists() or new_ct_path.exists():
                print("Target path already exists")
                continue
            opt_path.rename(new_opt_path)
            data_path.rename(new_data_path)
            ct_path.rename(new_ct_path)
