#!/bin/bash
set -e

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type no --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_no_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type er --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_er_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type reservoir --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_reservoir_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type agem --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_agem_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type ogd --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_ogd_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type der --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_der_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type ewc --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_ewc_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type CN --buffer_type lwf --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name CN_lwf_resmlp --seed 42