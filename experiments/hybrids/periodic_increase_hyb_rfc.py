from reservoir_ca.experiment import RegType
from reservoir_ca.tasks import HybridTask, IncreasingPeriod, Periodic
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task("periodic_increase_hyb_rfc_exp#.pkl", HybridTask,
             [{"per": Periodic, "inc": IncreasingPeriod},
              {"per": [5], "inc": [5]}],
             opts_extra={"reg_type": RegType.RANDOMFOREST})
