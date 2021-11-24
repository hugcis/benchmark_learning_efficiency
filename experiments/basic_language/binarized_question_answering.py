from reservoir_ca.tasks import ElementaryLanguage
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(ElementaryLanguage, [], {"binarized_task": True})
