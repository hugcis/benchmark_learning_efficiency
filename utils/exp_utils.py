from reservoir_ca.experiment import ExpOptions

def list_all_exps(exp_dir):
    opts_dict = {}
    opts_hashes_dict = {}
    for i in exp_dir.glob("**/*.json"):
        if (i.parent / (i.stem + ".pkl")).exists():
            opts = ExpOptions.from_json(open(i).read())
            opts_dict[i.stem] = opts
            opts_hashes_dict[i.stem] = opts.hashed_repr()
    return opts_dict, opts_hashes_dict
