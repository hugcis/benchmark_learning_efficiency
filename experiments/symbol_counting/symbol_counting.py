from reservoir_ca.tasks import SymbolCounting
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(SymbolCounting, [10])
