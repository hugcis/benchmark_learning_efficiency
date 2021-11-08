from tqdm import tqdm

from reservoir_ca.tasks import IncreasingPeriod
from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.experiment import Experiment
from experiments.helpers import init_exp

if __name__ == "__main__":
    res, args, opts = init_exp("increasing_period_exp.pkl")

    for _ in range(opts.n_rep):
        per = IncreasingPeriod(5)
        ca = CAReservoir(0, 2)
        opts.seq_len = 200
        exp = Experiment(ca, per, opts)
        for t in tqdm(opts.rules):
            ca = CAReservoir(t, 2)
            exp.ca = ca
            exp.fit()
            res.update(t, exp.eval_test())
        res.save()
