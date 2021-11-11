from reservoir_ca.experiment import RegType
from reservoir_ca.tasks import HardSymbolCounting, HybridTask, SymbolCounting
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task("symbol_ct_hard_hyb_rfc_exp#.pkl", HybridTask,
             [{"sym": SymbolCounting, "hard_sym": HardSymbolCounting},
              {"sym": [8], "hard_sym": [10]}],
             opts_extra={"reg_type": RegType.RANDOMFOREST})
