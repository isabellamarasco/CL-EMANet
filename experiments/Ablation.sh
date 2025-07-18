#!/bin/bash
set -e

python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.8
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.8
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.8

python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.9
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.9
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.9

python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.95
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.95
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.95

python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type no --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.999
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type random --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.999
python trainContinuousFlow.py --data_name CIC-IDS --continuous_flow_type daily --normalization_type EMANet --buffer_type agem --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.999
