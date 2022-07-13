# Benchmarking learning efficiency in deep reservoir computing

This is the code to reproduce results from the paper *Cisneros, H., Mikolov, T.,
& Sivic, J. (2022). Benchmarking Learning Efficiency in Deep Reservoir
Computing. 1st Conference on Lifelong Learning Agents, Montreal, Canada*.

## Run experiments

**WARNING**: Re-running all experiments might take a significant amount of time.
Experiments in the paper were done on a cluster using GPUs and a lot of
parallelism. The docker solution is particularly sub-optimal and will take a
long time to run experiments.

### Running with poetry

The easiest way to run the experiments is to use
[poetry](https://python-poetry.org/). First, clone the repo 
 
```sh
git clone https://github.com/hugcis/benchmark_learning_efficiency.git
```

Then, run `poetry install` to create a virtual environment and install all
the dependencies.

Then run: 
```sh
./run_experiments.sh
```

Finally

### Running in Docker

If you don't have or don't want to install poetry, you can build and install
everything within a docker container. Just run inside the repo:

``` sh
docker build .
```


