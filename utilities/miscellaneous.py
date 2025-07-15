import torch
from torch import nn

from materials import models

from . import dataset, metrics


def get_device() -> str:
    """
    Returns the best possible device to use, i.e. "cuda" if GPU is available, "mps" if working with MacOS device, "cpu" otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_CF_data(config):
    if config.data_name.lower() == "cic-ids":
        # Define data-path
        if config.normalization_type.lower() == "global":
            data_path = "./data/normalizedCICIDS_2017_small.csv"
        else:
            data_path = "./data/CICIDS_2017_small.csv"

        # Set other parameters
        target_columns = ["Label_bin"]
        info_columns = ["Label", "Time", "Day"]

    elif config.data_name.lower() == "unsw-nb15":
        # Define data-path
        if config.normalization_type.lower() == "global":
            data_path = "./data/normalizedUNSW-NB15.csv"
        else:
            data_path = "./data/UNSW-NB15.csv"

        # Set other parameters
        target_columns = ["Label"]
        info_columns = ["attack_cat"]

    # Load data
    if config.continuous_flow_type.lower() == "daily":
        data = dataset.ContinualDaily(
            data_path=data_path,
            target_columns=target_columns,
            info_columns=info_columns,
            n_data=config.n_data,
            output_type="torch",
        )
    elif config.continuous_flow_type.lower() == "flow":
        data = dataset.ContinualFlow(
            data_path=data_path,
            target_columns=target_columns,
            info_columns=info_columns,
            n_data=config.n_data,
            chunk_size=config.chunk_size,
            stride=config.stride,
        )
    else:
        raise NotImplementedError
    return data


def get_model(config):
    if config.normalization_type.lower() == "emanet":
        model = models.EMAFCNet(
            input_dim=config.input_dim,
            output_dim=1,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
        )
    else:
        model = models.FCNet(
            input_dim=config.input_dim,
            output_dim=1,
            n_layers=config.n_layers,
            hidden_dim=config.hidden_dim,
            dropout_rate=config.dropout_rate,
        )
    return model


def get_prediction(y_pred, threshold=0.5):
    """
    Given a tensor y_pred of shape (N, K) as input generated as the output of
    a neural network model, returns a tensor of shape (N, ) corresponding to the
    position of the Softmax of y_pred, i.e. the class predicted by the network.
    """
    sigmoid_pred = nn.Sigmoid()(y_pred)
    out = torch.zeros_like(sigmoid_pred)
    out[sigmoid_pred > threshold] = 1
    return out


def major_classifier_accuracy(data, timestep):
    sub_df = data(timestep)
    attacks_in_chunck = sub_df["Label"].unique()

    # Count occurences
    n_samples_per_chunck = {
        attack: len(sub_df[sub_df["Label"] == attack]) for attack in attacks_in_chunck
    }

    # Compute accuracy of major classifier
    acc_major = n_samples_per_chunck[attacks_in_chunck[0]] / len(sub_df)
    return acc_major


def random_classifier_accuracy(data, timestep):
    sub_df = data(timestep)
    attacks_in_chunck = sub_df["Label"].unique()

    # Count occurences
    n_samples_per_chunck = {
        attack: len(sub_df[sub_df["Label"] == attack]) for attack in attacks_in_chunck
    }

    # Compute the (expected) accuracy of random classifier, that is:
    # acc = \sum_i p_i^2
    acc_random = 0
    for attack in n_samples_per_chunck:
        acc_random = acc_random + (n_samples_per_chunck[attack] / len(sub_df)) ** 2
    return acc_random


def test_on_previous_experiences(
    model,
    normalizer,
    test_sets,
    timesteps,
):
    # Get device
    device = get_device()

    # Switch model to eval
    model.eval()

    # Test the model on all the previous timesteps
    metrics_t = {
        "Acc": {},
        "FPR": {},
        "F1": {},
        "Precision": {},
        "Recall": {},
        "AUROC": {},
        "ROC": {},
        "Confusion": {},
    }
    total_acc, N_test = 0, 0  # Required to compute avg accuracy
    for t_test, (x_test, y_test) in enumerate(test_sets):
        # Deactivate gradient
        with torch.no_grad():

            # Normalize data if needed
            x_test = normalizer(x_test)

            # Compute prediction on (normalized) test set
            y_pred_test = model(x_test.float().to(device))

            # Update total acc
            total_acc += metrics.accuracy(
                y_pred_test, y_test.float().to(device), threshold=0.5
            ) * len(y_pred_test)
            N_test += len(y_pred_test)

            # Add all the metrics to metrics_t
            metrics_t["Acc"][timesteps[t_test]] = round(
                metrics.accuracy(y_pred_test, y_test.float().to(device), threshold=0.5),
                4,
            )
            metrics_t["FPR"][timesteps[t_test]] = round(
                metrics.FPR(y_pred_test, y_test.float().to(device), threshold=0.5),
                4,
            )
            metrics_t["F1"][timesteps[t_test]] = round(
                metrics.f1_score(y_pred_test, y_test.float().to(device), threshold=0.5),
                4,
            )
            metrics_t["Precision"][timesteps[t_test]] = round(
                metrics.precision(
                    y_pred_test, y_test.float().to(device), threshold=0.5
                ),
                4,
            )
            metrics_t["Recall"][timesteps[t_test]] = round(
                metrics.recall(y_pred_test, y_test.float().to(device), threshold=0.5),
                4,
            )
            metrics_t["AUROC"][timesteps[t_test]] = round(
                metrics.AUROC(y_pred_test, y_test.float().to(device)),
                4,
            )
            metrics_t["Confusion"][timesteps[t_test]] = metrics.confusion_matrix(
                y_pred_test, y_test.float().to(device), threshold=0.5
            )

    # Compute avg accuracy
    avg_acc = total_acc / N_test
    return metrics_t, avg_acc


def compute_ROC(
    model,
    normalizer,
    test_sets,
    timesteps,
):
    # Get device
    device = get_device()

    # Switch model to eval
    model.eval()

    # Initialize roc curve list
    roc_curve = []
    for t_test, (x_test, y_test) in enumerate(test_sets):
        # Deactivate gradient
        with torch.no_grad():

            # Normalize data if needed
            x_test = normalizer(x_test)

            # Compute prediction on (normalized) test set
            y_pred_test = model(x_test.float().to(device))

            # Compute ROC values
            roc_curve.append(
                metrics.ROC_curve(
                    y_pred_test,
                    y_test.long().to(device),
                    n_samples=50,
                )
            )
    return torch.stack(roc_curve)
