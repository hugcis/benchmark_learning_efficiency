from reservoir_ca.tasks import ElementaryLanguage
from reservoir_ca.experiment import RegType
from experiments.helpers import run_task

if __name__ == "__main__":
    run_task(ElementaryLanguage, [], opts_extra={"reg_type": RegType.RBFSVM,
                                                 "ignore_mask": False,
                                                 "r_height": 3})
