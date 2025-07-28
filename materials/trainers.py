from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from materials import buffer as buff
from utilities import metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        buffer: buff.Buffer = None,
        normalizer: Any = None,
    ):
        """
        Implements the simplest Training function for a given model in our pipeline.
        In particular, given a model, a buffer, and a normalizer, the training
        works as follows:

        - A batch of previous data gets sampled by the Buffer and concatenated to
            the actual data.
        - Data gets normalized by the normalizer.
        - Data is processed by the neural network model and a step of the optimizer
            is performed.

        Args:
            model (nn.Module): The neural network model to be trained.
            buffer (buff.Buffer): A pre-defined Buffer to use for training.
            normalizer (Any): A function used to normalize data.
        """
        self.model = model
        self.buffer = buffer
        self.normalizer = normalizer

        if self.buffer:
            self.buffer_type = self.buffer.buffer_type.lower()
        else:
            self.buffer_type = "no"

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        n_epochs: int = 20,
    ):
        """
        The function executing the training.
        """
        # Cycle through epochs
        for epoch in range(n_epochs):
            # Initialize tqdm
            progress_bar = tqdm(
                range(len(train_loader)),
                desc=f"Epoch: {epoch+1}/{n_epochs}",
            )

            # Loop through batches
            epoch_avg_loss = 0.0
            epoch_avg_acc = 0.0
            for idx, (x, y) in enumerate(train_loader):

                # If buffer modifies gradient (e.g. A-GEM), compute the modified step
                if self.buffer_type == "agem":
                    # Reset optimizer
                    optimizer.zero_grad()

                    # Compute and project gradient
                    g, loss, y_pred = self.buffer.get_projected_gradient(x, y, loss_fn)

                    # Apply gradient manually
                    k = 0
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.requires_grad:
                                num_params = p.numel()  # Number of tensor elements
                                p.grad = g[k : k + num_params].view(p.shape).clone()
                                k += num_params
                    optimizer.step()

                else:
                    # Get buffer sample (if available)
                    if self.buffer_type == "random":
                        x_b, y_b = self.buffer.get()
                        x_b, y_b = x_b.to(x.device), y_b.to(y.device)

                        # Normalize + concatenate
                        if self.normalizer:
                            x_b = self.normalizer(x_b)
                        x, y = torch.cat((x, x_b)), torch.cat((y, y_b))

                    # Compute prediction
                    y_pred = self.model(x)

                    # Reset optimizer
                    optimizer.zero_grad()

                    # Compute loss and backpropagate
                    loss = loss_fn(y_pred, y)
                    loss.backward()

                    # Step + Reset optimizer
                    optimizer.step()

                # Compute accuracy
                acc = metrics.accuracy(y_pred, y, threshold=0.5)

                # Update loss registry
                epoch_avg_loss = epoch_avg_loss + loss.item()
                epoch_avg_acc = epoch_avg_acc + acc

                progress_bar.set_postfix(
                    {
                        "Loss": epoch_avg_loss / (idx + 1),
                        "Acc": epoch_avg_acc / (idx + 1),
                    }
                )
                progress_bar.update(1)
            progress_bar.close()
