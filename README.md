# Benchmarking learning efficiency in deep reservoir computing

This is the code to reproduce results from the paper 

*Benchmarking Learning Efficiency in Deep Reservoir Computing. Cisneros, H.,
Mikolov, T., & Sivic, J. (2022). 1st Conference on Lifelong Learning Agents,
Montreal, Canada*.

## Re-run experiments

**WARNING**: Re-running all experiments might take a significant amount of time.
Experiments in the paper were done on a cluster using GPUs and a lot of
parallelism. The docker solution is particularly sub-optimal and will take a
long time to run experiments.

An alternative to running all the experiments is to download the data directly: 
``` sh
wget https://data.ciirc.cvut.cz/public/projects/2022BenchmarkingLearningEfficiency/experiment_2022-07-13T15:32:50.tar

tar -xvf "experiment_2022-07-13T15:32:50.tar"
```

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

### Running in Docker

If you don't have or don't want to install poetry, you can build and install
everything within a docker container. Just run the following from inside the
repo:

``` sh
docker build -t pypoetry_bledrc .
docker run -it --entrypoint=/bin/bash pypoetry_bledrc -i
```

This will open a bash tty within the docker container where you can run 
```sh
./run_experiments.sh
```

## Generate figures and tables

Once the data is generated or downloaded (make sure that you have the
experiment_gpu and experiment_sgd folders), you can run jupyter notebooks in
order to re-generate the figures and tables from the paper.

Just run
``` sh
poetry run jupyter notebook
```
and open the two jupyter notebooks in the folder `notebooks`.
