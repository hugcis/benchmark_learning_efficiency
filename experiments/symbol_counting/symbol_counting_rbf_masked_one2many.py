from reservoir_ca.ca_res import ProjectionType
from reservoir_ca.experiment import RegType
from reservoir_ca.tasks import SymbolCounting
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(SymbolCounting, [10], opts_extra={"reg_type": RegType.RBFSVM,
                                               "ignore_mask": False,
                                               "proj_type": ProjectionType.ONE_TO_MANY})
