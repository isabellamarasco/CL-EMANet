import torch
from torch import nn


class Buffer:
    def __init__(
        self, buffer_size: int, buffer_features: int, buffer_batch_size: int
    ) -> None:
        """
        Implements the basic module defining a Buffer, which should then be customized.
        It provides some basic functionalities via the functions _update and _get which
        updates the Buffer state by adding new data and returns a batch of data from the
        Buffer, respectively. It expect each sub-class to have at least two functions:
        * update(): to add elements on the Buffer, possibly employing _update().
        * get(): to get a batch of data from the Buffer, possibly employing _get().

        Args:
            buffer_size (int): The maximum dimensionality of the Buffer which can be
                            used before it gets slowly overrided by new data.
            buffer_features (int): The number of features memorized by the Buffer. It
                            should correspond to the number of features of the data.
            buffer_batch_size (int): The batch size for the Buffer. Determines the
                            amount of data returned when _get() is called.
        """
        self.buffer_size: int = buffer_size
        self.buffer_features: int = buffer_features
        self.buffer_batch_size = buffer_batch_size
        self.is_full: bool = False
        self.index_count: int = 0  # Counts which percentage of the Buffer is full

        self.buffer_x = torch.zeros(
            (self.buffer_size, self.buffer_features), dtype=torch.float32
        )
        self.buffer_y = torch.zeros((self.buffer_size, 1), dtype=torch.int64)

    def _update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the status of the Buffer. In particular, if the Buffer is still empty,
        it just add the new data to the Buffer. If the Buffer is full, it samples the
        required amount of data from the Buffer to get overrided by the new samples.

        Args:
            x (torch.Tensor): The new input data to be added to the Buffer, of shape (N, d).
            y (torch.Tensor): The new output data to be added to the Buffer, of shape (N, 1).
        """
        # Convert x to float32
        x = x.to(torch.float32)

        assert x.shape[0] == y.shape[0]
        N = x.shape[0]

        if not self.is_full:
            if self.index_count + N >= self.buffer_size:
                self.buffer_x[self.index_count : self.buffer_size] = x[
                    : self.buffer_size - self.index_count
                ]
                self.buffer_y[self.index_count : self.buffer_size] = y[
                    : self.buffer_size - self.index_count
                ].to(torch.int64)
                self.is_full = True

                self.update(
                    x[self.buffer_size - self.index_count :],
                    y[self.buffer_size - self.index_count :],
                )
            else:
                self.buffer_x[self.index_count : self.index_count + N] = x
                self.buffer_y[self.index_count : self.index_count + N] = y
            self.index_count = self.index_count + N
        else:
            self.index_count = torch.randint(0, self.buffer_size, (N,))
            self.buffer_x[self.index_count] = x
            self.buffer_y[self.index_count] = y.cpu().to(torch.int64)

    def _get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of data randomly sampled from the Buffer.
        """
        # If buffer is not full, just just return all the buffer up to that point
        if (not self.is_full) and (self.index_count <= self.buffer_batch_size):
            return (
                self.buffer_x[: self.index_count],
                self.buffer_y[: self.index_count],
            )

        # If buffer is full and batch_size = None, then return all the dataset
        if self.buffer_batch_size >= self.buffer_size:
            return (self.buffer_x, self.buffer_y)

        # Else, randomly sample a batch from buffer and return it
        idx = torch.randperm(self.buffer_size)[: self.buffer_batch_size]
        return (self.buffer_x[idx], self.buffer_y[idx])


class RandomBuffer(Buffer):
    def __init__(
        self, buffer_size: int, buffer_features: int, buffer_batch_size: int
    ) -> None:
        """
        Implements the RandomBuffer, which is a Buffering technique where at each
        experience, the data on which the model has been trained gets added to the Buffer,
        either randomly or by following a rule. In this implementation, only True
        attacks get added to the Buffer.

        Args:
            buffer_size (int): The maximum dimensionality of the Buffer which can be
                            used before it gets slowly overrided by new data.
            buffer_features (int): The number of features memorized by the Buffer. It
                            should correspond to the number of features of the data.
            buffer_batch_size (int): The batch size for the Buffer. Determines the
                            amount of data returned when _get() is called.
        """
        super().__init__(buffer_size, buffer_features, buffer_batch_size)

        self.buffer_type = "Random"

    def update(
        self, x: torch.Tensor, y: torch.Tensor, attacks_only: bool = True
    ) -> None:
        if attacks_only:
            return self._update(x[y[:, 0].cpu() == 1], y[y[:, 0].cpu() == 1])
        return self._update(x, y)

    def get(self):
        return self._get()


class AGEMBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int,
        buffer_features: int,
        buffer_batch_size: int,
        model: nn.Module,
    ) -> None:
        """
        Implements the A-GEM, which is a Buffering technique where at each
        gradient step of the training, the gradient "g" computed on the data batch
        gets compared with the gradient "g_ref" of the loss function over a batch
        of the Buffer and, if g^T g_ref < 0, then g gets projected over the space defined
        by g_ref. This reduces forgetting as it helps the gradient to never move
        away from the region where the old data was correctly classified.

        Args:
            buffer_size (int): The maximum dimensionality of the Buffer which can be
                            used before it gets slowly overrided by new data.
            buffer_features (int): The number of features memorized by the Buffer. It
                            should correspond to the number of features of the data.
            buffer_batch_size (int): The batch size for the Buffer. Determines the
                            amount of data returned when _get() is called.
            model (nn.Module): The neural network model that is getting trained.
        """
        super().__init__(buffer_size, buffer_features, buffer_batch_size)

        self.buffer_type = "AGEM"
        self.model = model

    def update(
        self, x: torch.Tensor, y: torch.Tensor, attacks_only: bool = True
    ) -> None:
        if attacks_only:
            return self._update(x[y[:, 0].cpu() == 1], y[y[:, 0].cpu() == 1])
        return self._update(x, y)

    def get(self):
        return self._get()

    def _project_gradient(self, g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
        """
        Project "g" over the space defined by "g_ref".
        """
        g_ref_dot_g_ref = torch.dot(g_ref, g_ref)
        if g_ref_dot_g_ref > 0:  # Avoid division by zero
            alpha = torch.dot(g, g_ref) / g_ref_dot_g_ref
            if alpha < 0:
                g -= alpha * g_ref
        return g

    def _compute_gradient(self, x, y, loss_fn):
        """
        Return the gradient of the loss with respect to the neural network
        parameters, computed on the dataset (x, y) given as input.
        """
        self.model.zero_grad()

        y_pred = self.model(x)
        loss = loss_fn(y_pred, y.to(torch.float32))

        # Calcola i gradienti senza modificarli nei parametri
        grads = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=False, retain_graph=False
        )

        # Concatenazione dei gradienti in un unico vettore
        g = torch.cat([grad.view(-1) for grad in grads if grad is not None])

        return g, loss, y_pred

    def get_projected_gradient(
        self, x, y, loss_fn
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the gradient for the current task and projects it
        using memory constraints.
        """
        g, loss, y_pred = self._compute_gradient(x, y, loss_fn)

        if self.index_count > 0:
            # Compute reference gradient g_ref from buffer
            x_ref, y_ref = self._get()
            x_ref, y_ref = x_ref.to(x.device), y_ref.to(y.device)
            g_ref, _, _ = self._compute_gradient(x_ref, y_ref, loss_fn)
            g_ref = g_ref.detach()

            # Project gradient
            g = self._project_gradient(g, g_ref)
        return g, loss, y_pred

    def apply_gradient(self, model, g):
        """
        Injects the computed gradient "g" from the previous functions onto the
        neural network to prepare it to get optimized.
        """
        # Apply gradient manually
        idx = 0
        for p in model.parameters():
            num_params = p.numel()
            if p.grad is not None:
                p.grad.copy_(g[idx : idx + num_params].view(p.shape))
            idx += num_params
