from reservoir_ca.ca_res import ProjectionType
from reservoir_ca.tasks import IncreasingPeriod
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(IncreasingPeriod, [5], opts_extra={"proj_type": ProjectionType.ONE_TO_PATTERN})
