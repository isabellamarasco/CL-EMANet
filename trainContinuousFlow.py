import logging
from datetime import datetime

import torch
from torch import nn

import config
from materials import buffer, normalizers, trainers
from utilities import miscellaneous

########################################
########### SETTING STUFF UP ###########
########################################
# Load config
cfg = config.UNSWNB15_Config()  # CICIDS_Config()
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

minmax_txt_path = f"./logs/{timestamps}_minmax_values.txt"

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
    print(torch.norm(x_train_norm))

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

# Saving tensor of mx and Mx
torch.save(
    torch.stack(minmax_list, dim=0),
    f"./logs/{timestamps}_minmax_tensor.pt",
)
