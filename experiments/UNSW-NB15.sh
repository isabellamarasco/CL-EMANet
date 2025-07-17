#!/bin/bash
set -e

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99
