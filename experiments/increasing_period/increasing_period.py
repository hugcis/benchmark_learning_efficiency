import pathlib
import pickle as pkl

from tqdm import tqdm

from reservoir_ca.tasks import IncreasingPeriod
from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.experiment import ExpOptions, Experiment
from experiments.helpers import parser


if __name__ == "__main__":
    args = parser.parse_args()

    base = pathlib.Path().resolve()
    path = base / "experiment_results" / pathlib.Path("inc_period_exp.pkl")
    res: list[list[float]]
    if path.exists():
        res = pkl.load(open(path, "rb"))
    else:
        res = []

    for _ in range(10):
        per = IncreasingPeriod(5)
        ca = CAReservoir(0, 2)

        exp_opts = ExpOptions(seq_len=200)
        exp = Experiment(ca, per, exp_opts)
        evals = []
        for t in tqdm(args.rules):
            ca = CAReservoir(t, 2)
            exp.ca = ca
            exp.fit()
            evals.append(exp.eval_test())
        res.append(evals)
        pkl.dump(res, open(path, "wb"))
