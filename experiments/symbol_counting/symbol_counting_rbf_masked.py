from reservoir_ca.experiment import RegType
from reservoir_ca.tasks import SymbolCounting
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task("symbol_counting_rbf_masked_exp#.pkl", SymbolCounting, [8],
             opts_extra={"reg_type": RegType.RBFSVM,
                         "ignore_mask": False})