import torch
from torch import nn
from typing import Tuple, Optional


class Buffer:
    """
    Base memory buffer.

    Stores pairs (x, y) as tensors:
      - x: float32, shape (buffer_size, d)
      - y: float32, shape (buffer_size, 1)  (for BCEWithLogitsLoss)
    """

    def __init__(
        self, buffer_size: int, buffer_features: int, buffer_batch_size: int
    ) -> None:
        self.buffer_size: int = int(buffer_size)
        self.buffer_features: int = int(buffer_features)
        self.buffer_batch_size: int = int(buffer_batch_size)

        self.is_full: bool = False
        self.index_count: int = (
            0  # how many items are currently filled (up to buffer_size)
        )
        self.seen_count: int = 0  # used by reservoir policy

        # keep on CPU; trainer moves to device
        self.buffer_x = torch.zeros(
            (self.buffer_size, self.buffer_features), dtype=torch.float32
        )
        self.buffer_y = torch.zeros((self.buffer_size, 1), dtype=torch.float32)

        # For subclasses to identify themselves (used by the trainer)
        self.buffer_type: str = "base"

    # ---------------------
    # Internal helpers
    # ---------------------
    def _append_until_full(self, x: torch.Tensor, y: torch.Tensor) -> int:
        """
        Append sequentially while capacity remains. Returns how many samples were appended.
        """
        if self.is_full or self.index_count >= self.buffer_size:
            self.is_full = True
            return 0

        N = x.shape[0]
        end = min(self.index_count + N, self.buffer_size)
        count = end - self.index_count
        if count > 0:
            self.buffer_x[self.index_count : end] = x[:count]
            self.buffer_y[self.index_count : end] = y[:count]
            self.index_count = end
            if self.index_count >= self.buffer_size:
                self.is_full = True
        return count

    def _rand_overwrite(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Randomly overwrite memory with (x, y), with replacement (ER-style overwrite).
        """
        N = x.shape[0]
        idx = torch.randint(0, self.buffer_size, (N,))
        self.buffer_x[idx] = x
        self.buffer_y[idx] = y
        self.is_full = True
        self.index_count = self.buffer_size

    def _reservoir_insert(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        True reservoir sampling (Vitter's algorithm, batched as a loop).
        """
        N = x.shape[0]
        for k in range(N):
            t = self.seen_count  # 0-based counter of seen items
            if t < self.buffer_size:
                self.buffer_x[t] = x[k]
                self.buffer_y[t] = y[k]
                self.index_count = t + 1
                if self.index_count >= self.buffer_size:
                    self.is_full = True
            else:
                j = torch.randint(0, t + 1, (1,)).item()
                if j < self.buffer_size:
                    self.buffer_x[j] = x[k]
                    self.buffer_y[j] = y[k]
            self.seen_count += 1

    def _get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a batch sampled uniformly from memory.
        """
        if self.index_count == 0:
            return self.buffer_x[:0], self.buffer_y[:0]

        total = self.index_count if not self.is_full else self.buffer_size

        if (not self.is_full) and (self.index_count <= self.buffer_batch_size):
            return self.buffer_x[: self.index_count], self.buffer_y[: self.index_count]

        if self.buffer_batch_size >= total:
            return self.buffer_x[:total], self.buffer_y[:total]

        idx = torch.randint(0, total, (self.buffer_batch_size,))
        return self.buffer_x[idx], self.buffer_y[idx]

    # ---------------------
    # Public API to override
    # ---------------------
    def update(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        raise NotImplementedError

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ERBuffer(Buffer):
    """
    Classic Experience Replay:
      - Store all seen samples (random overwrite when full).
      - Sample uniformly from memory on demand.
    """

    def __init__(
        self, buffer_size: int, buffer_features: int, buffer_batch_size: int
    ) -> None:
        super().__init__(buffer_size, buffer_features, buffer_batch_size)
        self.buffer_type = "er"

    def update(self, x: torch.Tensor, y: torch.Tensor, *_, **__) -> None:
        if x.numel() == 0:
            return
        x = x.to(torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        y = y.to(torch.float32)
        appended = self._append_until_full(x, y)
        if appended < x.shape[0]:
            self._rand_overwrite(x[appended:], y[appended:])

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get()


class ReservoirBuffer(Buffer):
    """
    Reservoir Experience Replay:
      - True reservoir sampling policy to decide which items stay in memory.
      - Sampling uniformly at get().
    """

    def __init__(
        self, buffer_size: int, buffer_features: int, buffer_batch_size: int
    ) -> None:
        super().__init__(buffer_size, buffer_features, buffer_batch_size)
        self.buffer_type = "reservoir"

    def update(self, x: torch.Tensor, y: torch.Tensor, *_, **__) -> None:
        if x.numel() == 0:
            return
        x = x.to(torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        y = y.to(torch.float32)
        self._reservoir_insert(x, y)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get()


class DERBuffer(Buffer):
    """
    Dark Experience Replay buffer:
      - Stores (x, y, logits) where `logits` are the model outputs at storage time.
      - get() returns (x, y, logits).
    DER++ is supported in the trainer by adding CE on (x, y) from memory.
    """

    def __init__(
        self,
        buffer_size: int,
        buffer_features: int,
        buffer_batch_size: int,
        logit_dim: int = 1,
    ) -> None:
        super().__init__(buffer_size, buffer_features, buffer_batch_size)
        self.buffer_type = "der"
        self.logit_dim = int(logit_dim)
        self.buffer_logits = torch.zeros(
            (self.buffer_size, self.logit_dim), dtype=torch.float32
        )

    def _append_logits_until_full(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> int:
        if self.is_full or self.index_count >= self.buffer_size:
            self.is_full = True
            return 0
        N = x.shape[0]
        end = min(self.index_count + N, self.buffer_size)
        count = end - self.index_count
        if count > 0:
            self.buffer_x[self.index_count : end] = x[:count]
            self.buffer_y[self.index_count : end] = y[:count]
            self.buffer_logits[self.index_count : end] = z[:count]
            self.index_count = end
            if self.index_count >= self.buffer_size:
                self.is_full = True
        return count

    def _rand_overwrite_logits(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> None:
        N = x.shape[0]
        idx = torch.randint(0, self.buffer_size, (N,))
        self.buffer_x[idx] = x
        self.buffer_y[idx] = y
        self.buffer_logits[idx] = z
        self.is_full = True
        self.index_count = self.buffer_size

    def update(
        self, x: torch.Tensor, y: torch.Tensor, logits: Optional[torch.Tensor] = None
    ) -> None:
        if x.numel() == 0:
            return
        x = x.to(torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        y = y.to(torch.float32)

        if logits is None:
            # If not provided, store zeros (trainer should normally pass logits)
            logits = torch.zeros((x.shape[0], self.logit_dim), dtype=torch.float32)
        else:
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            logits = logits.to(torch.float32)

        appended = self._append_logits_until_full(x, y, logits)
        if appended < x.shape[0]:
            self._rand_overwrite_logits(x[appended:], y[appended:], logits[appended:])

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.index_count == 0:
            empty = self.buffer_x[:0]
            return empty, empty, empty

        total = self.index_count if not self.is_full else self.buffer_size
        if (not self.is_full) and (self.index_count <= self.buffer_batch_size):
            return (
                self.buffer_x[: self.index_count],
                self.buffer_y[: self.index_count],
                self.buffer_logits[: self.index_count],
            )

        if self.buffer_batch_size >= total:
            return (
                self.buffer_x[:total],
                self.buffer_y[:total],
                self.buffer_logits[:total],
            )

        idx = torch.randint(0, total, (self.buffer_batch_size,))
        return self.buffer_x[idx], self.buffer_y[idx], self.buffer_logits[idx]


class AGEMBuffer(Buffer):
    """
    A-GEM (Averaged GEM) style buffer and gradient projector.

    - Memory: ER-like (store all, random overwrite when full).
    - At each step:
        * Compute current gradient g on the current mini-batch.
        * If memory not empty, sample a memory batch and compute g_ref.
        * If g^T g_ref < 0, project: g <- g - (g^T g_ref / ||g_ref||^2) * g_ref
        * Apply g as the model gradient.
    """

    def __init__(
        self,
        buffer_size: int,
        buffer_features: int,
        buffer_batch_size: int,
        model: nn.Module,
    ) -> None:
        super().__init__(buffer_size, buffer_features, buffer_batch_size)
        self.buffer_type = "agem"
        self.model = model

    def update(self, x: torch.Tensor, y: torch.Tensor, *_, **__) -> None:
        if x.numel() == 0:
            return
        x = x.to(torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        y = y.to(torch.float32)
        appended = self._append_until_full(x, y)
        if appended < x.shape[0]:
            self._rand_overwrite(x[appended:], y[appended:])

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._get()

    # --------- A-GEM helpers ---------
    @torch.no_grad()
    def _project_gradient_inplace(self, g: torch.Tensor, g_ref: torch.Tensor) -> None:
        denom = torch.dot(g_ref, g_ref)
        if denom.item() <= 0:
            return
        dot = torch.dot(g, g_ref)
        if dot.item() < 0:
            g -= (dot / denom) * g_ref  # in-place

    def _flatten_grads(self, params_iter):
        flat = []
        for p in params_iter:
            if p.grad is None:
                flat.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                flat.append(p.grad.view(-1))
        return torch.cat(flat, dim=0)

    def get_projected_gradient(
        self, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.modules.loss._Loss
    ):
        """
        Compute logits, loss, and a projected flat gradient for the *current* batch.
        Returns: flat_g, loss, y_pred
        """
        if y.ndim == 1:
            y = y.unsqueeze(1)
        y = y.to(torch.float32)

        # Current batch grad
        self.model.zero_grad(set_to_none=True)
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        params = [p for p in self.model.parameters() if p.requires_grad]
        flat_g = self._flatten_grads(params)

        # Memory grad
        if self.index_count > 0:
            x_ref, y_ref = self._get()
            x_ref = x_ref.to(x.device, dtype=torch.float32)
            if y_ref.ndim == 1:
                y_ref = y_ref.unsqueeze(1)
            y_ref = y_ref.to(y.device, dtype=torch.float32)

            self.model.zero_grad(set_to_none=True)
            y_ref_pred = self.model(x_ref)
            ref_loss = loss_fn(y_ref_pred, y_ref)
            ref_loss.backward()
            flat_g_ref = self._flatten_grads(params).detach()

            flat_g_proj = flat_g.clone()
            self._project_gradient_inplace(flat_g_proj, flat_g_ref)
            return flat_g_proj, loss, y_pred

        return flat_g, loss, y_pred
