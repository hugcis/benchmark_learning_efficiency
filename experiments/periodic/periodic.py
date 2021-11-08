import random
import numpy as np
from tqdm import tqdm

from reservoir_ca.tasks import Periodic
from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.experiment import Experiment
from experiments.helpers import init_exp

if __name__ == "__main__":
    res, opts = init_exp("periodic_exp.pkl")

    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    for _ in tqdm(range(opts.n_rep)):
        per = Periodic(5)
        ca = CAReservoir(0, 2)
        exp = Experiment(ca, per, opts)
        for t in opts.rules:
            ca = CAReservoir(t, 2, redundancy=opts.redundancy)
            exp.ca = ca
            exp.fit()
            res.update(t, exp.eval_test())
        res.save()
