#!/bin/bash
set -e

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type no --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_no_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type no --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_no_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type no --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_no_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type no --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_no_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type er  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_er_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type er  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_er_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type er --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_er_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type er --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_er_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type reservoir  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_reservoir_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type reservoir  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_reservoir_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type reservoir  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_reservoir_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type reservoir  --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_reservoir_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type agem  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_agem_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type agem  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_agem_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type agem  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_agem_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type agem  --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_agem_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type ogd  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_ogd_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type ogd  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_ogd_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type ogd  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_ogd_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type ogd  --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_ogd_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type der  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_der_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type der  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_der_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type der  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_der_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type der  --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_der_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type ewc  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_ewc_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type ewc  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_ewc_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type ewc  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_ewc_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type ewc  --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_ewc_resmlp --seed 42

python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type no --buffer_type lwf  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name no_lwf_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type global --buffer_type lwf  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name global_lwf_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type local --buffer_type lwf  --model_name resmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name local_lwf_resmlp --seed 42
python trainContinuousFlow.py --data_name UNSW-NB15 --continuous_flow_type flow --normalization_type EMANet --buffer_type lwf  --model_name emaresmlp --batch_size 20000 --n_epochs 20 --learning_rate 5e-4 --eta 0.99 --run_name ema_lwf_resmlp --seed 42
