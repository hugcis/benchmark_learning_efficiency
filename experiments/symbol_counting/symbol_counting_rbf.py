from tqdm import tqdm

from reservoir_ca.tasks import SymbolCounting
from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.experiment import Experiment, RegType
from experiments.helpers import init_exp

if __name__ == "__main__":
    res, opts = init_exp("symbol_counting_rbf_exp#.pkl")
    opts.reg_type = RegType.RBFSVM

    for _ in tqdm(range(opts.n_rep)):
        task = SymbolCounting(8)
        ca = CAReservoir(0, task.output_dimension())
        exp = Experiment(ca, task, opts)
        for t in opts.rules:
            ca = CAReservoir(t, task.output_dimension(),
                             redundancy=opts.redundancy)
            exp.ca = ca
            exp.fit()
            res.update(t, exp.eval_test())

    res.save()