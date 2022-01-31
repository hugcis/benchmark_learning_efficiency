import numpy as np

from reservoir_ca.tasks import Task, TaskMask

class SequentialTasks:
    def __init__(self, task_list: list[Task], n_running_acc: int = 100) -> None:
        self.task_list = task_list
        self.task_number = 0
        self.last_accuracies = []
        self.n_running_acc = n_running_acc

    @property
    def running_accuracy(self):
        return np.mean(self.last_accuracies + (self.n_running_acc - len(self.last_accuracies)) * [0])

    def output_dimension(self):
        return max(task.output_dimension() for task in self.task_list)

    def clear_running_accuracies(self):
        self.last_accuracies = []

    def get_example(self) -> TaskMask:
        if self.running_accuracy > .9:
            self.task_number += 1
            self.clear_running_accuracies()
            print(f"Switching to next task {self.task_number}")

        return self.get_example_from_list(self.task_number)

    def was_correct(self, c: bool):
        self.last_accuracies.append(float(c))
        if len(self.last_accuracies) > self.n_running_acc:
            self.last_accuracies.pop(0)

    def get_example_from_list(self, task_number: int) -> TaskMask:
        if task_number > len(self.task_list):
            raise ValueError(f"Task number {task_number} is greater than the number of tasks")

        return self.task_list[task_number].generate_tasks(max_n_seq=1)
