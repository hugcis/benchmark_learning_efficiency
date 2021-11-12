from reservoir_ca.experiment import RegType
from reservoir_ca.tasks import HardSymbolCounting
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task("hard_sym_count_rbf2_exp#.pkl", HardSymbolCounting, [25],
             opts_extra={"reg_type": RegType.RBFSVM})
