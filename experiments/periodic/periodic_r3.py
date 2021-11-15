from reservoir_ca.tasks import Periodic
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(Periodic, [5], opts_extra={"r_height": 3})
