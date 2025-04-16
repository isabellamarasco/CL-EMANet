import torch
from . import miscellaneous
import torchmetrics
from torch import nn


def MR(y_pred, y_true, threshold=0.5):
    """
    Returns the missclassification rate (MR) between the prediction y_pred and the true
    solution y_true
    """
    y_pred_bin = miscellaneous.get_prediction(y_pred, threshold=threshold)
    MissRate = torch.mean((y_pred_bin != y_true).to(torch.float)).item()

    return MissRate


def accuracy(y_pred, y_true, threshold=0.5):
    """
    Returns the Accuracy of the prediction y_pred compared with the true solution y_true
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

    # Avoid extreme case
    if (TP + FN) == 0:
        return 1.0
    return TP / (TP + FN)


def FPR(y_pred, y_true, threshold=0.5):
    """
    Get the false positive rate, defined as:

    FPR = FP / (FP + TN)
    """
    TP, TN, FP, FN = confusion_matrix(y_pred, y_true, threshold)

    # Avoid extreme case
    if (FP + TN) == 0:
        return 1.0
    return FP / (FP + TN)


def AUROC(y_pred, y_true):
    """
    Compute Area Under the Curve (AUROC) for the given input predictions y_pred
    """
    # Convert y_pred to sigmoid predictions
    sigmoid_pred = nn.Sigmoid()(y_pred)

    # If all the targets are positive or negative, just return 1.0
    if torch.sum(y_true) == len(y_true) or torch.sum(y_true) == 0:
        return 1.0

    # Compute AUROC
    auroc = torchmetrics.AUROC(task="binary")
    return auroc(sigmoid_pred, y_true).item()
