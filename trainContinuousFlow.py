import argparse
import logging
import os
from datetime import datetime

import torch
from torch import nn
import numpy as np
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

    # Dataset and experiment setup
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
        choices=["no", "global", "local", "EMANet", "CN"],
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
        help="Momentum of EMA module when EMANet is chosen as normalization type",
    )

    # CN-specific
    parser.add_argument(
        "--cn_momentum",
        type=float,
        default=0.1,
        help="BatchNorm momentum for CN-like input normalization.",
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

    # (Restored so your script doesn’t crash where they are referenced)
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default="./plots",
        help="Directory to save generated plots/figures (if any).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run label stored in the .pt bundle.",
    )

    return parser.parse_args()


########################################
############### TRAIN ##################
########################################
def main(cfg):
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

        if cfg.normalization_type == "CN":
            # Use CN-like input normalization
            normalizer = _norms.ContinualNormInput(
                cfg.input_dim, momentum=cfg.cn_momentum
            )
        else:
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

        # Normalize current train split
        x_train = x_train.float()
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
        # Make sure CN's BN updates during training evaluations in the loop (normalizer follows model mode if you toggle it)
        normalizer.train(True)
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
                # Avoid updating CN running stats while extracting logits (eval mode)
                was_training = normalizer.training
                normalizer.eval()
                with torch.no_grad():
                    x_train_norm_full = normalizer(x_train).float().to(cfg.device)
                    logits = model(x_train_norm_full).detach().cpu()
                if was_training:
                    normalizer.train(True)
                B.update(x_train.cpu(), y_train.cpu(), logits)
            else:
                B.update(x_train, y_train)

        ########################################
        ############# TESTING ##################
        ########################################
        # Use eval mode for model and CN during evaluation
        was_training_m = model.training
        was_training_n = normalizer.training
        model.eval()
        normalizer.eval()

        metrics_t, avg_acc = miscellaneous.test_on_previous_experiences(
            model,
            normalizer,
            test_sets,
            timesteps,
        )

        # restore modes
        if was_training_m:
            model.train(True)
        if was_training_n:
            normalizer.train(True)

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


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
