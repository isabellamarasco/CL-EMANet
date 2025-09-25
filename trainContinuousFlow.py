import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import csv

from materials import buffer
from materials import trainers
from utilities import miscellaneous


########################################
########### PARSE ARGUMENTS ############
########################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run continual learning model or compare results."
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "compare"],
        default="train",
        help="Run training pipeline (train) or only compare saved results (compare).",
    )

    # Dataset and experiment setup (used in train mode)
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["CIC-IDS", "UNSW-NB15"],
        default="UNSW-NB15",
        help="A string identifying the dataset to use.",
    )
    parser.add_argument(
        "--continuous_flow_type",
        type=str,
        choices=["daily", "flow"],
        default="flow",
        help="Type of continuous flow, 'daily' if the data supports automatic division into experiences, 'flow' otherwise.",
    )
    parser.add_argument(
        "--normalization_type",
        type=str,
        choices=["no", "global", "local", "EMANet"],
        default="global",
        help="Type of normalization algorithm to use",
    )
    parser.add_argument(
        "--buffer_type",
        type=str,
        choices=["no", "er", "reservoir", "agem", "ogd", "der", "ewc", "lwf"],
        default="er",
        help="Continual learning strategy",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "fc",
            "emafc",
            "resmlp",
            "emaresmlp",
            "glumlp",
            "emaglumlp",
            "ftt",
            "emaftt",
        ],
        default=None,
        help="Model architecture to use. If None, defaults to FCNet/EMAFCNet depending on normalization_type.",
    )

    # Buffer
    parser.add_argument(
        "--buffer_size", type=int, default=500_000, help="Maximum size of the buffer"
    )
    parser.add_argument(
        "--buffer_batch_size",
        type=int,
        default=20_000,
        help="Number of datapoints extracted from the buffer at each training iteration",
    )

    # Model
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of layers defining the baseline neural network",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Number of neurons in each hidden layer defining the baseline neural network",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout value of the baseline neural network",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20_000,
        help="Batch size used for training the model",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of epochs used for training the model",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate used for training the model",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.99,
        help="Value of the momentum of EMA module when EMANet is chosen as normalization type",
    )

    # Optional flow-related
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500_000,
        help="Number of elements per experience when synthetical chunks is used",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=500_000,
        help="Stride used to create syntetic chunks",
    )

    # Seed
    parser.add_argument(
        "--seed", type=str, default="None", help="Random seed (default: None)"
    )

    # Plotting & output
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set (in train mode), produce plots for this run (aggregate and optionally per-experience).",
    )
    parser.add_argument(
        "--plot_per_experience",
        action="store_true",
        help="If set (in train mode with --plot), also save per-experience epoch curves.",
    )
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default="./plots",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run label to embed in CSV/plot titles.",
    )

    # Compare mode inputs
    parser.add_argument(
        "--compare_paths",
        nargs="+",
        default=None,
        help="List of .pt result files to compare (used in --mode compare).",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving-average window for smoothing curves in comparison/plots (1 means no smoothing).",
    )

    return parser.parse_args()


########################################
############# PLOTTING UTILS ###########
########################################
def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    if win > len(arr):
        win = len(arr)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    return (cumsum[win:] - cumsum[:-win]) / float(win)


def _pad_to_same_length(curves: List[np.ndarray]) -> List[np.ndarray]:
    L = min(len(c) for c in curves)
    return [c[:L] for c in curves]


def plot_aggregate_curves(
    epoch_curves: List[List[float]],
    title: str,
    ylabel: str,
    outfile: str,
    smooth: int = 1,
):
    """
    epoch_curves: list over experiences; each is a list over epochs.
    We compute the mean curve across experiences and plot (optionally smoothed).
    """
    if len(epoch_curves) == 0 or len(epoch_curves[0]) == 0:
        return
    curves = [np.array(c, dtype=np.float32) for c in epoch_curves]
    curves = _pad_to_same_length(curves)
    mat = np.stack(curves, axis=0)  # [E, T]
    mean_curve = mat.mean(axis=0)
    if smooth > 1:
        mean_curve = _moving_average(mean_curve, smooth)
        x = np.arange(len(mean_curve))
    else:
        x = np.arange(len(mean_curve))

    plt.figure()
    plt.plot(x, mean_curve)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_per_experience_curves(
    epoch_curves: List[List[float]],
    title_prefix: str,
    ylabel: str,
    outdir: str,
    smooth: int = 1,
):
    os.makedirs(outdir, exist_ok=True)
    for i, curve in enumerate(epoch_curves):
        c = np.array(curve, dtype=np.float32)
        if smooth > 1:
            c = _moving_average(c, smooth)
            x = np.arange(len(c))
        else:
            x = np.arange(len(c))
        plt.figure()
        plt.plot(x, c)
        plt.title(f"{title_prefix} (timestep #{i})")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(
            os.path.join(outdir, f"{title_prefix.replace(' ', '_').lower()}_t{i}.png"),
            bbox_inches="tight",
        )
        plt.close()


def plot_comparison(files: List[str], outdir: str, smooth: int = 1):
    """
    Load multiple .pt result files and compare:
    - aggregate mean over experiences of epoch_avg_acc_seen
    - aggregate mean over experiences of epoch_acc_current
    """
    os.makedirs(outdir, exist_ok=True)

    legends = []
    seen_curves = []
    current_curves = []

    for f in files:
        try:
            d = torch.load(f, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
            continue

        label = os.path.basename(f)
        # Try to parse buffer type from file name (if present)
        if "buffer-" in label:
            try:
                part = label.split("buffer-")[1]
                label = part.split("_")[0]
            except Exception:
                pass

        # Build aggregate (mean across experiences)
        es: List[List[float]] = d.get("epoch_avg_acc_seen", [])
        ec: List[List[float]] = d.get("epoch_acc_current", [])

        if len(es) == 0 or len(ec) == 0:
            print(f"[WARN] No per-epoch curves in {f}, skipping.")
            continue

        es_np = [np.array(e, dtype=np.float32) for e in es if len(e) > 0]
        ec_np = [np.array(e, dtype=np.float32) for e in ec if len(e) > 0]
        if len(es_np) == 0 or len(ec_np) == 0:
            continue

        es_np = _pad_to_same_length(es_np)
        ec_np = _pad_to_same_length(ec_np)
        mean_seen = np.stack(es_np, axis=0).mean(axis=0)
        mean_current = np.stack(ec_np, axis=0).mean(axis=0)

        if smooth > 1:
            mean_seen = _moving_average(mean_seen, smooth)
            mean_current = _moving_average(mean_current, smooth)

        seen_curves.append(mean_seen)
        current_curves.append(mean_current)
        legends.append(label)

    # Plot comparison
    if len(seen_curves) > 0:
        L = min(len(c) for c in seen_curves)
        plt.figure()
        for curve, lab in zip(seen_curves, legends):
            plt.plot(np.arange(L), curve[:L], label=lab)
        plt.title("Comparison — AvgAcc over Seen Tasks (mean across experiences)")
        plt.xlabel("Epoch")
        plt.ylabel("AvgAcc(seen)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(outdir, "compare_seen.png"), bbox_inches="tight")
        plt.close()

    if len(current_curves) > 0:
        L = min(len(c) for c in current_curves)
        plt.figure()
        for curve, lab in zip(current_curves, legends):
            plt.plot(np.arange(L), curve[:L], label=lab)
        plt.title("Comparison — Acc on Current Task (mean across experiences)")
        plt.xlabel("Epoch")
        plt.ylabel("Acc(current)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(outdir, "compare_current.png"), bbox_inches="tight")
        plt.close()


########################################
############### TRAIN ##################
########################################
def main_train(cfg):
    cfg.device = miscellaneous.get_device()

    # Optional seeding
    if cfg.seed != "None":
        seed = int(cfg.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Data setup (dataset-dependent constants)
    if cfg.data_name == "CIC-IDS":
        cfg.n_data = 2_827_876
        cfg.input_dim = 68
    elif cfg.data_name == "UNSW-NB15":
        cfg.n_data = 2_540_047
        cfg.input_dim = 51

    ########################################
    ########### SETTING STUFF UP ###########
    ########################################
    print(f"Using device: {cfg.device}")

    # Ensure output folders exist
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs(cfg.save_plot_dir, exist_ok=True)

    ####### Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    timestamps = datetime.now().strftime("%Y%m%d%H%M%S")
    file_handler = logging.FileHandler(
        f"./logs/{timestamps}_type-{cfg.continuous_flow_type}_norm-{cfg.normalization_type}.txt"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Create console handler (prints to stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Attach handlers (avoid duplicates if re-run in same process)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Print run info
    logging.info(f"-- Dataset: {cfg.data_name}")
    logging.info(f"-- Continuous flow type: {cfg.continuous_flow_type}")
    logging.info(f"-- Normalization type: {cfg.normalization_type}")
    logging.info(f"-- Buffer type: {cfg.buffer_type}")
    logging.info(f"-- eta: {cfg.eta}")
    if cfg.run_name:
        logging.info(f"-- Run name: {cfg.run_name}")

    ########################################
    ###### LOAD DATA AND DEFINE MODEL ######
    ########################################
    data = miscellaneous.get_CF_data(cfg)

    # Set day list
    timesteps = data.timesteps
    N = len(timesteps)

    # Define model and normalizer
    normalizer = (
        miscellaneous.get_normalizer(cfg)
        if hasattr(miscellaneous, "get_normalizer")
        else None
    )
    if normalizer is None:
        from materials import normalizers as _norms

        normalizer = _norms.SimpleNormalization(cfg.normalization_type)

    model = miscellaneous.get_model(cfg).to(cfg.device)

    # Initialize strategy/buffer
    bt = cfg.buffer_type.lower()
    B = None
    if bt == "er":
        B = buffer.ERBuffer(cfg.buffer_size, data.d, cfg.buffer_batch_size)
        logging.info("-- Using ERBuffer")
    elif bt == "reservoir":
        B = buffer.ReservoirBuffer(cfg.buffer_size, data.d, cfg.buffer_batch_size)
        logging.info("-- Using ReservoirBuffer")
    elif bt == "der":
        B = buffer.DERBuffer(
            cfg.buffer_size, data.d, cfg.buffer_batch_size, logit_dim=1
        )
        logging.info("-- Using DERBuffer")
    elif bt == "agem":
        B = buffer.AGEMBuffer(cfg.buffer_size, data.d, cfg.buffer_batch_size, model)
        logging.info("-- Using A-GEM Buffer")
    elif bt in ["no", "ogd", "ewc", "lwf"]:
        B = None
        logging.info("-- Not using memory buffer")
    else:
        B = None
        logging.info("-- Unknown buffer type, defaulting to no buffer")

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Trainer
    trainer = trainers.Trainer(
        model=model,
        buffer=B,
        normalizer=normalizer,
        buffer_type=cfg.buffer_type,  # explicitly pass so ogd/ewc/lwf work with buffer=None
    )

    ########################################
    ############# TRAINING #################
    ########################################
    minmax_list = []
    test_sets = []
    experience_accuracies = []
    experience_aurocs = []

    # Per-epoch (for forgetting plots)
    all_epoch_avg_acc_seen = []  # list of lists, length N, each list has n_epochs items
    all_epoch_acc_current = []  # list of lists, length N, each list has n_epochs items

    for i in range(N):
        # Get data at time t
        x, y = data.input_output_split(i)

        # Train/test split
        (x_train, y_train), (x_test, y_test) = data.train_test_split(
            x, y, train_split=0.9, shuffle=True
        )
        test_sets.append((x_test, y_test))

        # Compute and update normalizer min/max on current train
        Mx, mx = normalizer.get_minmax(x_train)
        normalizer.update(Mx, mx)
        minmax_list.append(torch.stack((Mx[0], mx[0]), dim=1))

        # Normalize current train split
        x_train_norm = normalizer(x_train)

        # Send to device / convert dtypes
        x_train_norm = x_train_norm.float().to(cfg.device)
        y_train_dev = y_train.float().to(cfg.device)

        # DataLoader for current experience
        train_loader = data.create_dataloader(
            x_train_norm, y_train_dev, batch_size=cfg.batch_size, shuffle=True
        )

        # Logging basics for the experience
        logging.info("\n-------------------------")
        logging.info(
            f"Timestep = {timesteps[i]}, Classes: [0: {int(torch.sum(1 - y_train_dev))}, 1: {int(torch.sum(y_train_dev))}]"
        )
        logging.info(
            f"Major Classifier: {miscellaneous.major_classifier_accuracy(data, i):0.4f}"
        )
        logging.info(
            f"Random Classifier: {miscellaneous.random_classifier_accuracy(data, i):0.4f}"
        )
        print()

        # Prepare per-epoch trackers for this experience
        epoch_avg_acc_seen = []  # average accuracy across all test sets seen so far
        epoch_acc_current = []  # accuracy on current experience only

        # Define end-of-epoch evaluation callback
        def _on_epoch_end(epoch_idx: int):
            metrics_t, avg_acc = miscellaneous.test_on_previous_experiences(
                model,
                normalizer,
                test_sets,
                timesteps,
            )
            # avg over seen so far:
            epoch_avg_acc_seen.append(avg_acc)

            # current experience accuracy only
            cur_ts = timesteps[len(test_sets) - 1]  # current timestep
            acc_current = metrics_t["Acc"][cur_ts]
            epoch_acc_current.append(acc_current)

            logging.info(
                f"[Eval e{epoch_idx+1}] AvgAcc(seen): {avg_acc:0.4f} | Acc(current {cur_ts}): {acc_current:0.4f}"
            )

        # Train epochs
        model.train()
        trainer.train(
            train_loader,
            optimizer,
            loss_fn,
            n_epochs=cfg.n_epochs,
            on_epoch_end=_on_epoch_end,
        )

        # Add current train to buffer
        if B is not None:
            if bt == "der":
                with torch.no_grad():
                    x_train_norm_full = normalizer(x_train).float().to(cfg.device)
                    logits = model(x_train_norm_full).detach().cpu()
                B.update(x_train.cpu(), y_train.cpu(), logits)
            else:
                B.update(x_train, y_train)

        ########################################
        ############# TESTING ##################
        ########################################
        metrics_t, avg_acc = miscellaneous.test_on_previous_experiences(
            model,
            normalizer,
            test_sets,
            timesteps,
        )
        logging.info(f"Avg Accuracy: {avg_acc:0.4f}")
        logging.info(f"Accuracy: {metrics_t['Acc']}")
        logging.info(f"FPR: {metrics_t['FPR']}")
        logging.info(f"F1: {metrics_t['F1']}")
        logging.info(f"Precision: {metrics_t['Precision']}")
        logging.info(f"Recall: {metrics_t['Recall']}")
        logging.info(f"AUROC: {metrics_t['AUROC']}")
        logging.info(f"Confusion: (TP, TN, FP, FN) = {metrics_t['Confusion']}")

        # Save test accuracy and AUROC for all past experiences, in order
        current_accs = []
        current_aurocs = []
        for t in timesteps[: len(test_sets)]:  # only tested so far
            current_accs.append(metrics_t["Acc"][t])
            current_aurocs.append(metrics_t["AUROC"][t])

        experience_accuracies.append(current_accs)
        experience_aurocs.append(current_aurocs)

        # Store per-epoch curves for this experience
        all_epoch_avg_acc_seen.append(epoch_avg_acc_seen)
        all_epoch_acc_current.append(epoch_acc_current)

    ########################################
    ############## SAVING ##################
    ########################################
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_base = f"./results/{stamp}_norm-{cfg.normalization_type}_buffer-{cfg.buffer_type}_eta-{cfg.eta}"
    pt_path = f"{save_base}_acc_auroc.pt"
    torch.save(
        {
            "accuracies_per_experience": experience_accuracies,
            "aurocs_per_experience": experience_aurocs,
            "timesteps": timesteps,
            # Per-epoch curves (for forgetting plots)
            "epoch_avg_acc_seen": all_epoch_avg_acc_seen,
            "epoch_acc_current": all_epoch_acc_current,
            "n_epochs": cfg.n_epochs,
            "buffer_type": cfg.buffer_type,
            "run_name": cfg.run_name,
        },
        pt_path,
    )
    logging.info(f"Saved per-experience metrics to {pt_path}")

    # Also save a tidy CSV of per-epoch curves
    csv_path = f"{save_base}_per_epoch.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experience_idx", "epoch_idx", "avg_acc_seen", "acc_current"])
        for exp_idx, (seen_curve, cur_curve) in enumerate(
            zip(all_epoch_avg_acc_seen, all_epoch_acc_current)
        ):
            T = min(len(seen_curve), len(cur_curve))
            for e in range(T):
                writer.writerow([exp_idx, e, seen_curve[e], cur_curve[e]])
    logging.info(f"Saved per-epoch curves to {csv_path}")

    ########################################
    ############### PLOTTING ###############
    ########################################
    if cfg.plot:
        tag = cfg.run_name if cfg.run_name else os.path.basename(pt_path)
        # Aggregate plots (mean across experiences)
        plot_aggregate_curves(
            all_epoch_avg_acc_seen,
            title=f"{tag} — AvgAcc over Seen (mean across experiences)",
            ylabel="AvgAcc(seen)",
            outfile=os.path.join(cfg.save_plot_dir, "aggregate_seen.png"),
            smooth=cfg.smooth_window,
        )
        plot_aggregate_curves(
            all_epoch_acc_current,
            title=f"{tag} — Acc on Current (mean across experiences)",
            ylabel="Acc(current)",
            outfile=os.path.join(cfg.save_plot_dir, "aggregate_current.png"),
            smooth=cfg.smooth_window,
        )

        # Per-experience (optional)
        if cfg.plot_per_experience:
            per_dir = os.path.join(cfg.save_plot_dir, "per_experience")
            plot_per_experience_curves(
                all_epoch_avg_acc_seen,
                title_prefix="AvgAcc over Seen",
                ylabel="AvgAcc(seen)",
                outdir=per_dir,
                smooth=cfg.smooth_window,
            )
            plot_per_experience_curves(
                all_epoch_acc_current,
                title_prefix="Acc on Current",
                ylabel="Acc(current)",
                outdir=per_dir,
                smooth=cfg.smooth_window,
            )


########################################
############## COMPARE #################
########################################
def main_compare(cfg):
    if not cfg.compare_paths:
        print("[ERROR] --mode compare requires --compare_paths file1.pt file2.pt ...")
        return
    os.makedirs(cfg.save_plot_dir, exist_ok=True)
    plot_comparison(cfg.compare_paths, cfg.save_plot_dir, smooth=cfg.smooth_window)
    print(f"[OK] Saved comparison plots to {cfg.save_plot_dir}")


if __name__ == "__main__":
    cfg = parse_args()
    if not hasattr(cfg, "save_plot_dir"):
        cfg.save_plot_dir = "./plots"
    if cfg.mode == "train":
        main_train(cfg)
    else:
        main_compare(cfg)
