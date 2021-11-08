import pathlib

from tqdm import tqdm

from reservoir_ca.tasks import Periodic
from reservoir_ca.ca_res import CAReservoir
from reservoir_ca.experiment import Experiment
from experiments.helpers import parser, Result

if __name__ == "__main__":
    args = parser.parse_args()
    base = pathlib.Path().resolve()
    path = base / "experiment_results" / pathlib.Path("periodic_exp.pkl")
    res = Result(path)

    for _ in range(10):
        per = Periodic(5)
        ca = CAReservoir(0, 2)
        exp = Experiment(ca, per)
        for t in tqdm(args.rules):
            ca = CAReservoir(t, 2)
            exp.ca = ca
            exp.fit()
            res.update(t, exp.eval_test())
        res.save()
