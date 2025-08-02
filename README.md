# On Normalization Issues in Continual Learning for Forgetting-Resilient IDS

This repository provides the official implementation of **EMANet**, a novel normalization strategy for continual learning in network intrusion detection systems (IDS). EMANet dynamically adapts to evolving data distributions by using an Exponential Moving Average (EMA) of input statistics during training.

This implementation is associated with our [AAAI2025 paper submission].

## 📋 Overview

Traditional normalization methods either rely on future data (global normalization) or suffer from instability and forgetting (local normalization). **EMANet** overcomes these limitations by:

- Using a learnable min-max normalization layer updated via EMA.
- Supporting plug-and-play use with popular replay-based continual learning strategies (Random, A-GEM).
- Handling real-world cybersecurity benchmarks such as **CIC-IDS 2017** and **UNSW-NB15**.

## 📂 Project Structure

```
EMANet/
│
├── trainContinuousFlow.py   # Main training script
├── tabulator.py             # Contains the script to generate the tables and plots used in the paper
├── preprocessing.py         # Contains the script to preprocess the datasets.
├── logs/                    # Output folder for loggers
├── results/                 # Output folder for the experiments
├── data/                    # (Must be created) folder to save the data
├── materials/               # Model, buffer, normalization, trainer modules
├── materials/               # Some utilities functions for the main code
├── experiments/
│   ├── Ablation_CICIDS.sh   # To reproduce ablation study on CIC-IDS from the paper
│   ├── Ablation_UNSW-NB15.sh# To reproduce ablation study on UNSW-NB15 from the paper
│   ├── CICIDS.sh            # To reproduce experiment on CIC-IDS from the paper
│   └── preprocessing.sh     # To reproduce preprocessing on UNSW-NB15 and CIC-IDS from the paper
│   └── UNSW-NB15.sh         # To reproduce experiment on UNSW-NB15 from the paper
├── logs/                    # Training logs will be stored here
├── requirements.txt         # The libraries required and the version used to run the experiments
└── README.md
```

## 🚀 Getting Started

### 1. Install Requirements

We recommend using Python 3.10+.

```bash
pip install -r requirements.txt
```

### 2. Data download and preparation
TODO: Place the datasets (CIC-IDS 2017, UNSW-NB15) in the `data/` folder.

Download the datasets from the official sources:
- **CIC-IDS 2017**: go to https://www.unb.ca/cic/datasets/ids-2017.html and click `Download this dataset`:
  - Insert the requested information (name, email, etc.).
  - Download the dataset `CIC-IDS-2017/CSVs/MachineLearningCSV.zip`
  - Unzip the downloaded folder and rename it to `CIC-IDS-2017`
  - Place the `CIC-IDS-2017` folder in the `data/` folder.

- **UNSW-NB15**: go to https://research.unsw.edu.au/projects/unsw-nb15-dataset and click `HERE`:
  - Download the `CSV Files` folder
  - Unzip the downloaded folder and rename it in `UNSW-NB15`
  - Maintain the following files: `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv`, and `NUNSW-NB15_features.csv`.
  - Place the folfer `UNSW-NB15` in the `data/` folder.

#### Option 1: Use Pre-defined Scripts
Run all preprocessing scripts to prepare the datasets:

```bash
bash experiments/preprocessing.sh
```

### Option 2: Custom Run with Arguments
You can run the preprocessing script manually for each dataset and configuration:

```bash
python preprocessing.py --data_name CICIDS --mode preprocess_only
python preprocessing.py --data_name CICIDS --mode normalize_only
python preprocessing.py --data_name CICIDS --mode all

python preprocessing.py --data_name UNSW-NB --mode preprocess_only
python preprocessing.py --data_name UNSW-NB --mode normalize_only
python preprocessing.py --data_name UNSW-NB --mode all
```

### ⚙️ Arguments

Below is a list of main command-line arguments supported by `preprocessing.py`:

| Argument | Description |
|--------------|-------------|
|  `--data_name`  | Dataset to use: `CIC-IDS`, `UNSW-NB15` or `all` | 
| `--mode` | Type of preprocessing: `preprocess_only` (not global normalization), `normalize_only` (with global normalization) or `all`| 

## 🧪 Running Experiments from the paper

### Option 1: Use Pre-defined Scripts

Run one of the prepared experiment scripts to replicate the experiments from the paper, once the data has been downloaded and processed as described above:

```bash
bash experiments/Ablation_CICIDS.sh
bash experiments/Ablation_UNSW-NB15.sh
bash experiments/CICIDS.sh
bash experiments/UNSW-NB15.sh
```

### Option 2: Custom Run with Arguments

You can run the training script manually with your desired configuration:

```bash
python trainContinuousFlow.py \
  --data_name UNSW-NB15 \
  --continuous_flow_type flow \
  --normalization_type EMANet \
  --buffer_type agem \
  --buffer_size 500000 \
  --batch_size 20000 \
  --n_epochs 20
```

## ⚙️ Arguments

Below is a list of main command-line arguments supported by `trainContinuousFlow.py`:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_name` | Dataset to use: `CIC-IDS` or `UNSW-NB15` | `UNSW-NB15` |
| `--continuous_flow_type` | Type of data stream: `daily` or `flow` | `flow` |
| `--normalization_type` | Normalization method: `no`, `global`, `local`, `EMANet`, `up_to` | `global` |
| `--buffer_type` | Replay strategy: `no`, `random`, `agem` | `random` |
| `--buffer_size` | Total buffer size | `500000` |
| `--n_layers` | Number of hidden layers in the classifier | `4` |
| `--dropout_rate` | Dropout probability | `0.5` |
| `--learning_rate` | Learning rate | `5e-4` |
| `--eta` | EMA decay rate for EMANet | `0.99` |
| `--chunk_size` | (Flow mode) Size of each experience chunk | `300000` |
| `--stride` | (Flow mode) Step between chunks | `100000` |

## 📈 Citation

*Citation will be added upon acceptance. Stay tuned!*

## 📧 Contact

For questions, please contact [Davide Evangelista](mailto:davide.evangelista5@unibo.it) or [Isabella Marasco](mailto:isabella.marasco4@unibo.it).