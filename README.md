# On Normalization Issues in Continual Learning for Forgetting‑Resilient IDS

This repository provides the official implementation of **EMANet**, a normalization strategy for continual learning in network intrusion detection systems (IDS). EMANet dynamically adapts to evolving data distributions using an Exponential Moving Average (EMA) of input statistics during training and plugs into a variety of continual‑learning strategies.

This codebase now includes:
- Multiple CL methods: **ER**, **Reservoir ER**, **DER/DER++**, **A‑GEM**, **OGD**, **EWC**, **LwF**, and **No‑buffer** baseline.
- **Per‑epoch evaluation hooks** to visualize catastrophic forgetting.
- Built‑in **plotting** and **cross‑run comparison** utilities directly from `train.py`.

> This implementation relates to our ??? submission (citation to be added).


## 📋 Overview

Traditional normalization either relies on future data (global normalization) or becomes unstable (local normalization). **EMANet** overcomes these issues by:
- Using a **learnable min‑max normalization layer** updated via EMA.
- Enabling **plug‑and‑play** with replay and gradient‑projection methods.
- Supporting real‑world IDS benchmarks: **CIC‑IDS 2017** and **UNSW‑NB15**.


## 📂 Project Structure

```
EMANet/
│
├── train.py                 # Main script: training, logging, plotting, comparison
├── preprocessing.py         # Dataset preprocessing and (optional) global normalization
├── tabulator.py             # Paper tables/plots (optional; your script)
├── materials/
│   ├── buffer.py            # Buffer & CL strategies (ER, Reservoir, DER, A-GEM) + utilities
│   ├── trainers.py          # Trainer w/ OGD, EWC, LwF, DER/DER++ logic + per-epoch callbacks
│   ├── normalizers.py       # EMANet, global, local, no-normalization (plug-in API)
│   └── models.py            # Baseline classifier(s)
├── utilities/
│   ├── miscellaneous.py     # Data loader, model/normalizer factories, evaluation helpers
│   └── metrics.py           # Accuracy/AUROC/F1 etc.
├── experiments/
│   ├── preprocessing.sh     # Preprocess CIC-IDS & UNSW-NB15
│   ├── CICIDS.sh            # Example end-to-end run
│   ├── UNSW-NB15.sh         # Example end-to-end run
│   ├── Ablation_CICIDS.sh   # Paper ablation
│   └── Ablation_UNSW-NB15.sh# Paper ablation
├── data/                    # Place datasets here (created by you)
├── logs/                    # Training logs
├── results/                 # Experiment outputs (.pt bundles + CSVs)
├── plots/                   # Saved figures (aggregate & per-experience, comparisons)
├── requirements.txt
└── README.md
```


## 🚀 Getting Started

### 1) Install Requirements

We recommend Python 3.10+.

```bash
pip install -r requirements.txt
```

### 2) Data Download and Preparation

Place the datasets in the `data/` folder.

- **CIC‑IDS 2017**  
  1) Download `CIC-IDS-2017/CSVs/MachineLearningCSV.zip` from the official page.  
  2) Unzip and rename to `CIC-IDS-2017`.  
  3) Move `data/CIC-IDS-2017/` into this repo.

- **UNSW‑NB15**  
  1) Download the **CSV Files** bundle from the official page.  
  2) Unzip and rename to `UNSW-NB15`.  
  3) Keep: `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv`, and `UNSW-NB15_features.csv`.  
  4) Move `data/UNSW-NB15/` into this repo.

#### Option A — Predefined Script
```bash
bash experiments/preprocessing.sh
```

#### Option B — Manual
```bash
python preprocessing.py --data_name CIC-IDS  --mode preprocess_only
python preprocessing.py --data_name CIC-IDS  --mode normalize_only
python preprocessing.py --data_name CIC-IDS  --mode all

python preprocessing.py --data_name UNSW-NB15 --mode preprocess_only
python preprocessing.py --data_name UNSW-NB15 --mode normalize_only
python preprocessing.py --data_name UNSW-NB15 --mode all
```

**`preprocessing.py` main flags**

| Argument       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `--data_name`  | Dataset: `CIC-IDS`, `UNSW-NB15`, or `all`                                   |
| `--mode`       | `preprocess_only`, `normalize_only`, or `all`                               |


## 🧪 Running Experiments

### Option A — Predefined scripts
```bash
bash experiments/CICIDS.sh
bash experiments/UNSW-NB15.sh
bash experiments/Ablation_CICIDS.sh
bash experiments/Ablation_UNSW-NB15.sh
```

### Option B — Custom runs

**Training + plotting this run**
```bash
python train.py \
  --data_name UNSW-NB15 \
  --continuous_flow_type flow \
  --normalization_type EMANet \
  --buffer_type der \
  --buffer_size 500000 \
  --buffer_batch_size 20000 \
  --batch_size 20000 \
  --n_epochs 20 \
  --run_name DER_eta0p99 \
  --plot --plot_per_experience \
  --save_plot_dir ./plots
```

**Compare multiple saved runs (no training)**
```bash
python train.py --mode compare \
  --compare_paths results/2025*_norm-*_buffer-*.pt \
  --save_plot_dir ./plots \
  --smooth_window 1
```

### What `train.py` saves
- `results/<stamp>_norm-<...>_buffer-<...>_eta-<...>_acc_auroc.pt`  
  Contains per‑experience metrics (Acc, AUROC), per‑epoch curves:
  - `epoch_avg_acc_seen`: avg accuracy across all **seen** experiences at each epoch (captures forgetting);
  - `epoch_acc_current`: accuracy on the **current** experience at each epoch.
- `results/<stamp>_..._per_epoch.csv` — tidy CSV: `experience_idx, epoch_idx, avg_acc_seen, acc_current`.
- Plots (if `--plot`): aggregate mean curves across experiences and optionally per‑experience figures.


## ⚙️ Key Arguments

**`train.py` main flags**

| Argument | Description | Default |
|---|---|---|
| `--mode` | `train` or `compare` | `train` |
| `--data_name` | Dataset: `CIC-IDS` or `UNSW-NB15` | `UNSW-NB15` |
| `--continuous_flow_type` | Data stream: `daily` \| `flow` | `flow` |
| `--normalization_type` | `no`, `global`, `local`, `EMANet` | `global` |
| `--buffer_type` | CL strategy: `no`, `er`, `reservoir`, `der`, `agem`, `ogd`, `ewc`, `lwf` | `er` |
| `--buffer_size` | Total memory size (for replay methods) | `500000` |
| `--buffer_batch_size` | Replay mini‑batch size per step | `20000` |
| `--n_layers` | Hidden layers in classifier | `4` |
| `--hidden_dim` | Hidden dimension per layer | `128` |
| `--dropout_rate` | Dropout probability | `0.5` |
| `--batch_size` | SGD batch size | `20000` |
| `--n_epochs` | Epochs per experience | `20` |
| `--learning_rate` | Learning rate | `5e-4` |
| `--eta` | EMA decay rate for EMANet | `0.99` |
| `--chunk_size` | (Flow) elements per experience | `500000` |
| `--stride` | (Flow) stride between experiences | `500000` |
| `--plot` | Save plots for this run | `False` |
| `--plot_per_experience` | Save per‑experience plots | `False` |
| `--save_plot_dir` | Output directory for plots | `./plots` |
| `--run_name` | Label for titles/files | `""` |
| `--compare_paths` | (compare) list of result `.pt` files | `None` |
| `--smooth_window` | Moving‑average window (plots) | `1` |


## 🧠 Implemented CL Strategies (High‑level)

- **No buffer**: plain training per experience.
- **ER** (Experience Replay): random memory + uniform sampling at each step.
- **Reservoir ER**: reservoir sampling policy for memory replacement.
- **DER / DER++**: replay with stored **logits** (KD loss) + optional CE on replay labels.
- **A‑GEM**: gradient projection using a reference gradient from memory.
- **OGD**: orthogonal projection of current gradient onto the complement of stored bases (no memory).
- **EWC**: diagonal Fisher regularization across experiences (no memory).
- **LwF**: distillation from a frozen teacher (previous model snapshot) on current data (no memory).

> All strategies work with **EMANet** or other normalization choices. Replay batches are normalized through the same normalizer during training.


## 📈 Visualization & Comparison

- **Per‑epoch curves** saved for each experience allow you to visualize catastrophic forgetting:
  - `AvgAcc(seen)`: mean accuracy over **all seen** test sets after each epoch.
  - `Acc(current)`: accuracy on the **current** test set after each epoch.
- `train.py --mode compare` plots **mean curves across experiences** for multiple runs in a single figure, enabling quick method comparison.


## 🔧 Reproducibility Tips

- Use `--seed <int>` to fix randomness (PyTorch + NumPy seeded).
- Keep `requirements.txt` pinned. If using GPUs/HPC, record CUDA/cuDNN and driver versions.
- Save `--run_name` per run to keep plots and CSVs organized.


## 📄 Citation

*Citation will be added upon acceptance. Stay tuned!*

