from reservoir_ca.tasks import HardSymbolCounting
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(HardSymbolCounting, [45])
