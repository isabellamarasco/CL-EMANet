import argparse
import logging
import os
from datetime import datetime

import torch
from torch import nn

from materials import buffer, normalizers, trainers
from utilities import miscellaneous


########################################
########### PARSE ARGUMENTS ############
########################################
def parse_args():
    parser = argparse.ArgumentParser(description="Run continual learning model.")

    # Dataset and experiment setup
    parser.add_argument(
        "--data_name",
        type=str,
        choices=["CIC-IDS", "UNSW-NB15"],
        default="CIC-IDS",
        help="A string identifying the dataset to use.",
    )
    parser.add_argument(
        "--continuous_flow_type",
        type=str,
        choices=["daily", "flow"],
        default="daily",
        help="Type of continuous flow, 'daily' if the data supports authomatical division into experiences, 'flow' otherwise.",
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
        choices=["no", "random", "agem"],
        default="random",
        help="Type of buffer to use",
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
    return parser.parse_args()


cfg = parse_args()
cfg.device = miscellaneous.get_device()

# Optional seeding
if cfg.seed != "None":
    seed = int(cfg.seed)
    logging.info(f"Seeding execution with seed: {seed}")
    torch.manual_seed(seed)
else:
    logging.info("No seed provided. Execution will be non-deterministic.")

# Data setup
if cfg.data_name == "CIC-IDS":
    cfg.n_data = 2_827_876
    cfg.input_dim = 68
elif cfg.data_name == "UNSW-NB15":
    cfg.n_data = 2_540_047
    cfg.input_dim = 51

########################################
########### SETTING STUFF UP ###########
########################################
# Load config
print(f"Using device: {cfg.device}")

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

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Print info to log
logging.info(f"-- Dataset: {cfg.data_name}")
logging.info(f"-- Continuous flow type: {cfg.continuous_flow_type}")
logging.info(f"-- Normalization type: {cfg.normalization_type}")
logging.info(f"-- eta: {cfg.eta}")

########################################
###### LOAD DATA AND DEFINE MODEL ######
########################################
data = miscellaneous.get_CF_data(cfg)

# Set day list
timesteps = data.timesteps
N = len(timesteps)

# Define model and normalizer
normalizer = normalizers.SimpleNormalization(cfg.normalization_type)
model = miscellaneous.get_model(cfg).to(cfg.device)

# Initialize (empty) buffer
if cfg.buffer_type.lower() == "random":
    logging.info("-- Using RandomBuffer")
    B = buffer.RandomBuffer(
        buffer_size=cfg.buffer_size,
        buffer_features=data.d,
        buffer_batch_size=cfg.buffer_batch_size,
    )
elif cfg.buffer_type.lower() == "agem":
    logging.info("-- Using A-GEM buffer")
    B = buffer.AGEMBuffer(
        buffer_size=cfg.buffer_size,
        buffer_features=data.d,
        buffer_batch_size=cfg.buffer_batch_size,
        model=model,
    )
else:
    logging.info("-- Not using buffer")
    B = None

# Set loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

# Initializer trainer
trainer = trainers.Trainer(
    model=model,
    buffer=B,
    normalizer=normalizer,
)

########################################
############# TRAINING #################
########################################
# Step 1: loop through time
minmax_list = []
test_sets = []

# To store accuracy and AUROC at each timestep
experience_accuracies = []
experience_aurocs = []
for i in range(N):
    # Get data at time t
    x, y = data.input_output_split(i)

    # Get train-test split
    (x_train, y_train), (x_test, y_test) = data.train_test_split(
        x,
        y,
        train_split=0.9,
        shuffle=True,
    )
    test_sets.append((x_test, y_test))

    # Get data maximum / minimum
    Mx, mx = normalizer.get_minmax(x_train)
    normalizer.update(Mx, mx)

    minmax_list.append(torch.stack((Mx[0], mx[0]), dim=1))

    # Normalize data
    x_train_norm = normalizer(x_train)

    # Send data to device and convert to float
    x_train_norm = x_train_norm.float().to(cfg.device)
    y_train = y_train.float().to(cfg.device)

    # Generate dataloader at time t
    train_loader = data.create_dataloader(
        x_train_norm, y_train, batch_size=cfg.batch_size, shuffle=True
    )

    # Step 2: loop through epochs
    logging.info("\n-------------------------")
    logging.info(
        f"Timestep = {timesteps[i]}, Classes: [0: {int(torch.sum(1 - y_train))}, 1: {int(torch.sum(y_train))}]"
    )

    # Print out accuracy of major and random classifiers
    logging.info(
        f"Major Classifier: {miscellaneous.major_classifier_accuracy(data, i):0.4f}"
    )
    logging.info(
        f"Random Classifier: {miscellaneous.random_classifier_accuracy(data, i):0.4f}"
    )
    print()
    # Switch model to train mode
    model.train()
    trainer.train(train_loader, optimizer, loss_fn, n_epochs=cfg.n_epochs)

    # Add data to buffer
    if B is not None:
        B.update(x_train, y_train, attacks_only=True)

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
    for t in timesteps[: len(test_sets)]:  # only test_sets seen so far
        current_accs.append(metrics_t["Acc"][t])
        current_aurocs.append(metrics_t["AUROC"][t])

    # Save as lists of lists (for each experience, all previous/tested ones)
    experience_accuracies.append(current_accs)
    experience_aurocs.append(current_aurocs)

# Saving resulting metrics into prescribed tensor
save_path = f"./results/{timestamps}_norm-{cfg.normalization_type}_buffer-{cfg.buffer_type}_eta-{cfg.eta}_acc_auroc.pt"
os.makedirs("./results/", exist_ok=True)
torch.save(
    {
        "accuracies_per_experience": experience_accuracies,
        "aurocs_per_experience": experience_aurocs,
        "timesteps": timesteps,
    },
    save_path,
)
logging.info(f"Saved per-experience metrics to {save_path}")
