#!/usr/bin/env bash

# Run IMDB experiments
# Save logs and results in timestamped pickle and text files.
#
mkdir -p experiment_gpu
for i in $(seq 50); do
    poetry run python reservoir_ca/supervised_wade_exps/main.py --model GRU \
        --output-file experiment_gpu/${i}$(date +%FT%T)_GRU.pkl > experiment_gpu/${i}$(date +%FT%T)_GRU.log
    poetry run python reservoir_ca/supervised_wade_exps/main.py --model Linear \
        --output-file experiment_gpu/${i}$(date +%FT%T)_Linear-init2.pkl > experiment_gpu/${i}$(date +%FT%T)_Linear-init2.log
    poetry run python reservoir_ca/supervised_wade_exps/main.py --model LSTM \
        --output-file experiment_gpu/${i}$(date +%FT%T)_LSTM.pkl > experiment_gpu/${i}$(date +%FT%T)_LSTM.log
    poetry run python reservoir_ca/supervised_wade_exps/main.py --model RNN \
        --output-file experiment_gpu/${i}$(date +%FT%T)_RNN.pkl > experiment_gpu/${i}$(date +%FT%T)_RNN.log
    poetry run python reservoir_ca/supervised_wade_exps/main.py --model Transformer \
        --output-file experiment_gpu/${i}$(date +%FT%T)_Transformer.pkl > experiment_gpu/${i}$(date +%FT%T)_Transformer.log
done


# Run benchmark experiments
# CA experiments
mkdir -p experiment_sgd
for i in $(seq 256); do
    poetry run python ${script_path} --rules $i \
        --n_rep 100 --redundancy 15 --reg_type "sgd" --seed 12305 \
        --proj_type "one_to_one" --r_height 2 --proj_factor 60 --increment-data \
        --exp_dirname experiment_sgd --max_n_seq 1200
done

# Echo state network
poetry run python ${script_path} --rules 0 \
    --n_rep 100 --redundancy 15 --reg_type sgd  --seed 12305 \
    --r_height 2 --proj_factor 60 --increment-data \
    --exp_dirname experiment_sgd --max_n_seq 1200 --esn_baseline

# RNN
poetry run python ${script_path} --rules 0 \
    --n_rep 100 --redundancy 15 --reg_type sgd  --seed 12305 \
    --r_height 2 --proj_factor 60 --increment-data \
    --exp_dirname experiment_sgd --max_n_seq 1200 --rnn_baseline

# LSTM
poetry run python ${script_path} --rules 0 \
    --n_rep 100 --redundancy 15 --reg_type sgd  --seed 12305 \
    --r_height 2 --proj_factor 60 --increment-data \
    --exp_dirname experiment_sgd --max_n_seq 1200 --lstm_baseline

# Transformer
poetry run python ${script_path} --rules 0 \
    --n_rep 100 --redundancy 15 --reg_type sgd  --seed 12305 \
    --r_height 2 --proj_factor 60 --increment-data \
    --exp_dirname experiment_sgd --max_n_seq 1200 --transformer_baseline
