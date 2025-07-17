import torch
import torchmetrics
import torchmetrics.functional.classification
from torch import nn

from . import miscellaneous


def MR(y_pred, y_true, threshold=0.5):
    """
    Returns the misclassification rate (MR) between the prediction y_pred and the true
    solution y_true.
    """
    y_pred_bin = miscellaneous.get_prediction(y_pred, threshold=threshold)
    MissRate = torch.mean((y_pred_bin != y_true).to(torch.float)).item()
    return MissRate


def accuracy(y_pred, y_true, threshold=0.5):
    """
    Returns the Accuracy of the prediction y_pred compared with the true solution y_true.
    """
    MissRate = MR(y_pred, y_true, threshold=threshold)
    return 1 - MissRate


def confusion_matrix(y_pred, y_true, threshold=0.5):
    """
    Return the quadruplet (TP, TN, FP, FN) of y_pred compared with y_true.
    """
    # Predict given the threshold
    y_pred_bin = miscellaneous.get_prediction(y_pred, threshold=threshold)

    # Get error vector and correct vector
    error_vec = torch.abs(y_pred_bin - y_true)
    correct_vec = 1 - error_vec

    # Compute quantities
    TP = torch.sum(correct_vec[y_pred_bin == 1]).item()
    TN = torch.sum(correct_vec[y_pred_bin == 0]).item()
    FP = torch.sum(error_vec[y_pred_bin == 1]).item()
    FN = torch.sum(error_vec[y_pred_bin == 0]).item()
    return TP, TN, FP, FN


def TPR(y_pred, y_true, threshold=0.5):
    """
    Get the true positive rate, defined as:

    TPR = TP / (TP + FN)
    """
    TP, TN, FP, FN = confusion_matrix(y_pred, y_true, threshold)
    # Avoid division by zero
    if (TP + FN) == 0:
        return 1.0
    return TP / (TP + FN)


def FPR(y_pred, y_true, threshold=0.5):
    """
    Get the false positive rate, defined as:

    FPR = FP / (FP + TN)
    """
    TP, TN, FP, FN = confusion_matrix(y_pred, y_true, threshold)
    # Avoid division by zero
    if (FP + TN) == 0:
        return 1.0  # or return 1.0 depending on your use case
    return FP / (FP + TN)


def AUROC(y_pred, y_true):
    """
    Compute Area Under the Curve (AUROC) for the given input predictions y_pred.
    """
    sigmoid_pred = nn.Sigmoid()(y_pred)

    # If all the targets are positive or negative, just return 1.0
    if torch.sum(y_true) == len(y_true) or torch.sum(y_true) == 0:
        return 1.0

    auroc = torchmetrics.AUROC(task="binary")
    return auroc(sigmoid_pred, y_true).item()


def ROC_curve(y_pred, y_true, n_samples=50):
    """
    Compute ROC curve (FPR, TPR, thresholds) for the given input predictions y_pred.
    """
    sigmoid_pred = nn.Sigmoid()(y_pred)

    # If all the targets are positive or negative, return trivial curve
    if torch.sum(y_true) == len(y_true) or torch.sum(y_true) == 0:
        return torch.stack(
            (
                torch.tensor([0.0, 1.0] + [0.0] * (n_samples - 2)),
                torch.tensor([0.0, 1.0] + [0.0] * (n_samples - 2)),
                torch.tensor([1.0, 0.0] + [0.0] * (n_samples - 2)),
            )
        ).to(y_true.device)

    fpr, tpr, thresholds = torchmetrics.functional.classification.binary_roc(
        sigmoid_pred,
        y_true,
        thresholds=n_samples,
    )
    return torch.stack((fpr, tpr, thresholds))


def precision(y_pred, y_true, threshold=0.5):
    """
    Compute Precision, defined as:

    Precision = TP / (TP + FP)
    """
    TP, TN, FP, FN = confusion_matrix(y_pred, y_true, threshold)
    if (TP + FP) == 0:  # Handle case where no positive predictions are made
        if FN == 0:  # If there are no positive in the data
            return 1.0
        return 0.0  # If there are some positive
    return TP / (TP + FP)


def recall(y_pred, y_true, threshold=0.5):
    """
    Compute Recall (or True Positive Rate), defined as:

    Recall = TP / (TP + FN)
    """
    return TPR(y_pred, y_true, threshold)


def f1_score(y_pred, y_true, threshold=0.5):
    """
    Compute the F1-Score, defined as:

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision(y_pred, y_true, threshold)
    r = recall(y_pred, y_true, threshold)
    if (p + r) == 0:
        return 0.0  # Handle case where both precision and recall are 0
    return 2 * (p * r) / (p + r)


from typing import List


def compute_forgetting(data):
    """Compute the forgetting matrix for jagged accuracy data."""
    F = []
    for i, row in enumerate(data):
        F_row = []
        for j in range(len(row)):
            max_diff = 0.0
            for k in range(i):
                if j < len(data[k]):
                    diff = data[k][j] - row[j]
                    if diff > max_diff:
                        max_diff = diff
            F_row.append(max_diff)
        F.append(F_row)
    return F


def compute_average_forgetting(F):
    """Compute average forgetting from the forgetting matrix."""
    AvgF = [0.0]  # by convention, first value is zero
    for i in range(1, len(F)):
        avg_f = sum(F[i]) / i
        AvgF.append(round(avg_f, 4))
    return AvgF
