import os
from typing import List

import numpy as np
import pandas as pd
import torch

from utilities import metrics


def load_matching_files(
    folder_path: str, norm_substring: str, buffer_substring: str
) -> List[str]:
    """
    Return list of .pt files in `folder_path` whose names contain both
    'norm-{norm_substring}' and 'buffer-{buffer_substring}'.

    Parameters:
        folder_path (str): Directory containing the files.
        norm_substring (str): One of 'no', 'global', 'local', 'EMANet'.
        buffer_substring (str): One of 'no', 'random', 'agem'.

    Returns:
        List[str]: Full paths to matching .pt files.
    """
    norm_key = f"norm-{norm_substring}"
    buffer_key = f"buffer-{buffer_substring}"

    matching_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pt") and norm_key in fname and buffer_key in fname:
            full_path = os.path.join(folder_path, fname)
            matching_files.append(torch.load(full_path))
    return matching_files


def summarize_accuracies(results_list):
    """
    Given a list of 3 dictionaries (results), compute the average and std of
    'accuracies_per_experience' per time step, returning a pandas DataFrame.
    """
    assert len(results_list) == 3, "Expected exactly 3 runs per setting."

    # Extract the list of timesteps (assumed to be the same in all 3 dictionaries)
    timesteps = list(results_list[0]["timesteps"])
    num_experiences = len(timesteps)

    # Prepare result table (each cell will contain a "mean ± std" string or "-")
    table = []

    for i in range(num_experiences):  # row index
        row = []
        for j in range(num_experiences):  # col index
            values = []
            for run in results_list:
                acc_list = run["accuracies_per_experience"]
                if i < len(acc_list) and j < len(acc_list[i]):
                    values.append(acc_list[i][j])
            if values:
                mean = np.mean(values)
                std = np.std(values)
                cell = f"{mean:.4f} ± {std:.4f}"
            else:
                cell = "-"
            row.append(cell)
        table.append(row)

    df = pd.DataFrame(table, columns=timesteps, index=timesteps)
    return df


def summarize_average_accuracy(results_list):
    """
    Given a list of 3 dictionaries (results), compute the average accuracy
    per timestep (row-wise), across 3 runs. Returns a pandas DataFrame.
    """
    assert len(results_list) == 3, "Expected exactly 3 runs per setting."

    timesteps = list(results_list[0]["timesteps"])
    num_experiences = len(timesteps)

    row_means = []

    for i in range(num_experiences):
        run_means = []
        for run in results_list:
            acc_list = run["accuracies_per_experience"]
            if i < len(acc_list):
                row_mean = np.mean(acc_list[i])
                run_means.append(row_mean)
        if run_means:
            mean = np.mean(run_means)
            std = np.std(run_means)
            row_means.append(f"{mean:.4f} ± {std:.4f}")
        else:
            row_means.append("-")

    df = pd.DataFrame(row_means, index=timesteps, columns=["Average Accuracy"])
    return df


def summarize_forgetting(results_list):
    """
    Given a list of 3 dictionaries (results), compute the average and std of
    forgetting computed from 'accuracies_per_experience' per time step, returning a pandas DataFrame.
    """
    assert len(results_list) == 3, "Expected exactly 3 runs per setting."

    # Extract the list of timesteps (assumed to be the same in all 3 dictionaries)
    timesteps = list(results_list[0]["timesteps"])
    num_experiences = len(timesteps)

    # Prepare result table (each cell will contain a "mean ± std" string or "-")
    table = []

    for i in range(num_experiences):  # row index
        row = []
        for j in range(num_experiences):  # col index
            values = []
            for run in results_list:
                forgetting_list = metrics.compute_forgetting(
                    run["accuracies_per_experience"]
                )
                if i < len(forgetting_list) and j < len(forgetting_list[i]):
                    values.append(forgetting_list[i][j])
            if values:
                mean = np.mean(values)
                std = np.std(values)
                cell = f"{mean:.4f} ± {std:.4f}"
            else:
                cell = "-"
            row.append(cell)
        table.append(row)

    df = pd.DataFrame(table, columns=timesteps, index=timesteps)
    return df


def summarize_aurocs(results_list):
    """
    Given a list of 3 dictionaries (results), compute the average and std of
    'aurocs_per_experience' per time step, returning a pandas DataFrame.
    """
    assert len(results_list) == 3, "Expected exactly 3 runs per setting."

    # Extract the list of timesteps (assumed to be the same in all 3 dictionaries)
    timesteps = list(results_list[0]["timesteps"])
    num_experiences = len(timesteps)

    # Prepare result table (each cell will contain a "mean ± std" string or "-")
    table = []

    for i in range(num_experiences):  # row index
        row = []
        for j in range(num_experiences):  # col index
            values = []
            for run in results_list:
                auroc_list = run["aurocs_per_experience"]
                if i < len(auroc_list) and j < len(auroc_list[i]):
                    values.append(auroc_list[i][j])
            if values:
                mean = np.mean(values)
                std = np.std(values)
                cell = f"{mean:.4f} ± {std:.4f}"
            else:
                cell = "-"
            row.append(cell)
        table.append(row)

    df = pd.DataFrame(table, columns=timesteps, index=timesteps)
    return df


########################################
###### LOAD METRICS DATA         #######
########################################
base_path = "./results"

no_norm_no_buffer = load_matching_files(base_path, "no", "no")
global_norm_no_buffer = load_matching_files(base_path, "global", "no")
local_norm_no_buffer = load_matching_files(base_path, "local", "no")
EMANet_norm_no_buffer = load_matching_files(base_path, "EMANet", "no")

no_norm_random_buffer = load_matching_files(base_path, "no", "random")
global_norm_random_buffer = load_matching_files(base_path, "global", "random")
local_norm_random_buffer = load_matching_files(base_path, "local", "random")
EMANet_norm_random_buffer = load_matching_files(base_path, "EMANet", "random")

no_norm_agem_buffer = load_matching_files(base_path, "no", "agem")
global_norm_agem_buffer = load_matching_files(base_path, "global", "agem")
local_norm_agem_buffer = load_matching_files(base_path, "local", "agem")
EMANet_norm_agem_buffer = load_matching_files(base_path, "EMANet", "agem")

print(summarize_accuracies(EMANet_norm_no_buffer))
print(summarize_forgetting(EMANet_norm_no_buffer))
print(summarize_aurocs(EMANet_norm_no_buffer))
print(summarize_average_accuracy(EMANet_norm_no_buffer))
