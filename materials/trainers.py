from typing import Any, Callable, Optional, List, Dict
from copy import deepcopy

import torch
from torch import nn
from tqdm import tqdm

from materials import buffer as buff
from utilities import metrics


# ---------- OGD helper ----------
class _OGDProjector:
    def __init__(self, max_bases: int = 50, eps: float = 1e-12):
        self.max_bases = max_bases
        self.eps = eps
        self.bases: List[torch.Tensor] = []
        self._sum: Optional[torch.Tensor] = None
        self._count: int = 0

    def reset_accumulator(self):
        self._sum = None
        self._count = 0

    def accumulate(self, g: torch.Tensor):
        if self._sum is None:
            self._sum = g.detach().clone()
        else:
            self._sum.add_(g.detach())
        self._count += 1

    def finalize_and_store(self):
        if self._count == 0 or self._sum is None:
            return
        mean_g = self._sum / float(self._count)
        norm = torch.linalg.norm(mean_g)
        if norm.item() > self.eps:
            b = mean_g / norm
            self.bases.append(b)
            if len(self.bases) > self.max_bases:
                self.bases.pop(0)
        self.reset_accumulator()

    @torch.no_grad()
    def project_inplace(self, g: torch.Tensor):
        if not self.bases:
            return
        for b in self.bases:
            coef = torch.dot(g, b)  # b is unit-norm
            if coef.abs().item() > 0:
                g -= coef * b


class Trainer:
    """
    Trainer supporting:
      - No buffer           : "no"
      - ER                  : "er"
      - Reservoir ER        : "reservoir"
      - A-GEM               : "agem"
      - OGD                 : "ogd"
      - DER / DER++         : "der"
      - EWC                 : "ewc"
      - LwF                 : "lwf"

    Also supports an `on_epoch_end(epoch_idx)` callback to run evaluations and store per-epoch metrics.
    """

    # ---- Hyperparams for special methods (tweak here or expose via CLI later) ----
    DER_KD_WEIGHT = 1.0  # weight for KD loss on replay logits
    DER_PP_CE_WEIGHT = 0.5  # extra CE on replay labels (DER++)
    LWF_ALPHA = 0.5  # weight for distillation vs label loss
    LWF_TEMPERATURE = 1.0  # T=1 -> plain MSE on logits
    EWC_LAMBDA = 1.0  # regularization strength

    def __init__(
        self,
        model: nn.Module,
        buffer: Optional[buff.Buffer] = None,
        normalizer: Optional[Any] = None,
        buffer_type: Optional[str] = None,
    ):
        self.model = model
        self.buffer = buffer
        self.normalizer = normalizer

        if buffer_type is not None:
            self.buffer_type = buffer_type.lower()
        else:
            if buffer is None:
                self.buffer_type = "no"
            else:
                self.buffer_type = getattr(self.buffer, "buffer_type", "no").lower()

        # OGD projector (used only if buffer_type == "ogd")
        self._ogd = _OGDProjector(max_bases=50)

        # EWC state
        self._ewc_tasks: List[Dict[str, torch.Tensor]] = (
            []
        )  # list of dicts: {'theta': {...}, 'fisher': {...}}
        self._ewc_running_squares: Optional[List[torch.Tensor]] = None
        self._ewc_running_count: int = 0

        # LwF teacher model (frozen)
        self._teacher: Optional[nn.Module] = None

    # ---------------- Utils ----------------
    def _flatten_grads(self):
        flat = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                flat.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                flat.append(p.grad.view(-1))
        if flat:
            return torch.cat(flat, dim=0)
        dev = next(self.model.parameters()).device
        return torch.tensor([], device=dev)

    def _param_dict(self) -> Dict[str, torch.Tensor]:
        return {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _zeros_like_params(self) -> Dict[str, torch.Tensor]:
        return {
            n: torch.zeros_like(p, device=p.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _ewc_penalty(self) -> torch.Tensor:
        if not self._ewc_tasks:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for task in self._ewc_tasks:
            theta_star = task["theta"]  # dict of param snapshots
            fisher = task["fisher"]  # dict of fisher diagonals
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                diff = p - theta_star[n]
                penalty = penalty + (fisher[n] * diff.pow(2)).sum()
        return 0.5 * self.EWC_LAMBDA * penalty

    # ------------- Training ----------------
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.modules.loss._Loss,
        n_epochs: int = 20,
        on_epoch_end: Optional[Callable[[int], None]] = None,
    ):
        """
        Train for `n_epochs`. At the end of each epoch, if provided, `on_epoch_end(epoch_idx)` is invoked.
        """
        device = next(self.model.parameters()).device

        # Initialize method-specific state at start of experience
        if self.buffer_type == "ogd":
            self._ogd.reset_accumulator()

        if self.buffer_type == "ewc":
            # reset running squares for Fisher estimation during this experience
            self._ewc_running_squares = [
                torch.zeros_like(p) for p in self.model.parameters() if p.requires_grad
            ]
            self._ewc_running_count = 0

        if self.buffer_type == "lwf":
            # teacher is previous model snapshot; keep frozen
            if self._teacher is not None:
                self._teacher.eval()

        for epoch in range(n_epochs):
            progress_bar = tqdm(
                range(len(train_loader)),
                desc=f"Epoch: {epoch+1}/{n_epochs}",
            )

            epoch_avg_loss = 0.0
            epoch_avg_acc = 0.0

            for idx, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32)
                if y.ndim == 1:
                    y = y.unsqueeze(1)
                y = y.to(device, dtype=torch.float32)

                if self.buffer_type == "agem":
                    # ---- A-GEM step ----
                    flat_g, loss, y_pred = self.buffer.get_projected_gradient(
                        x, y, loss_fn
                    )
                    optimizer.zero_grad(set_to_none=True)
                    offset = 0
                    for p in self.model.parameters():
                        if not p.requires_grad:
                            continue
                        n = p.numel()
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.copy_(flat_g[offset : offset + n].view_as(p))
                        offset += n
                    optimizer.step()

                elif self.buffer_type in ["er", "reservoir"]:
                    # ---- ER / Reservoir ----
                    if self.buffer is not None:
                        x_b, y_b = self.buffer.get()
                        x_b = x_b.to(device, dtype=torch.float32)
                        if y_b.ndim == 1:
                            y_b = y_b.unsqueeze(1)
                        y_b = y_b.to(device, dtype=torch.float32)
                        if self.normalizer is not None:
                            x_b = self.normalizer(x_b)
                        if x_b.shape[0] > 0:
                            x = torch.cat((x, x_b), dim=0)
                            y = torch.cat((y, y_b), dim=0)

                    y_pred = self.model(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(y_pred, y)
                    # EWC penalty can also be active with ER if you want (but default only in ewc mode)
                    if self.buffer_type == "ewc":  # unlikely; left for completeness
                        loss = loss + self._ewc_penalty()
                    loss.backward()
                    # EWC Fisher accumulation (only when in ewc mode)
                    if (
                        self.buffer_type == "ewc"
                        and self._ewc_running_squares is not None
                    ):
                        k = 0
                        for p in self.model.parameters():
                            if not p.requires_grad:
                                continue
                            self._ewc_running_squares[k] += p.grad.detach() ** 2
                            k += 1
                        self._ewc_running_count += 1
                    optimizer.step()

                elif self.buffer_type == "der":
                    # ---- DER / DER++ ----
                    # Sample memory: (x_b, y_b, logits_b)
                    x_b, y_b, z_b = self.buffer.get()
                    x_b = x_b.to(device, dtype=torch.float32)
                    z_b = z_b.to(device, dtype=torch.float32)
                    if y_b.ndim == 1:
                        y_b = y_b.unsqueeze(1)
                    y_b = y_b.to(device, dtype=torch.float32)

                    # Normalize memory if needed
                    if self.normalizer is not None and x_b.shape[0] > 0:
                        x_b = self.normalizer(x_b)

                    # Forward current + (optionally) memory for DER++
                    y_pred = self.model(x)  # current logits
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(y_pred, y)  # label loss on current batch

                    if x_b.shape[0] > 0:
                        # KD loss on memory logits (DER)
                        y_b_pred = self.model(x_b)
                        kd = nn.functional.mse_loss(y_b_pred, z_b)
                        loss = loss + self.DER_KD_WEIGHT * kd

                        # DER++: also CE on memory labels
                        ce_b = loss_fn(y_b_pred, y_b)
                        loss = loss + self.DER_PP_CE_WEIGHT * ce_b

                    # Optional EWC penalty if jointly enabled (not typical)
                    if self.buffer_type == "ewc":
                        loss = loss + self._ewc_penalty()

                    loss.backward()
                    # EWC fisher accumulation (only in pure ewc mode)
                    if (
                        self.buffer_type == "ewc"
                        and self._ewc_running_squares is not None
                    ):
                        k = 0
                        for p in self.model.parameters():
                            if not p.requires_grad:
                                continue
                            self._ewc_running_squares[k] += p.grad.detach() ** 2
                            k += 1
                        self._ewc_running_count += 1
                    optimizer.step()

                elif self.buffer_type == "ogd":
                    # ---- OGD ----
                    y_pred = self.model(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(y_pred, y)
                    # EWC penalty if someone wants OGD+EWC (not default)
                    if self.buffer_type == "ewc":
                        loss = loss + self._ewc_penalty()
                    loss.backward()

                    flat_g = self._flatten_grads()
                    self._ogd.accumulate(flat_g)
                    g_proj = flat_g.clone()
                    self._ogd.project_inplace(g_proj)

                    offset = 0
                    for p in self.model.parameters():
                        if not p.requires_grad:
                            continue
                        n = p.numel()
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.copy_(g_proj[offset : offset + n].view_as(p))
                        offset += n
                    optimizer.step()

                elif self.buffer_type == "ewc":
                    # ---- EWC (no memory) ----
                    y_pred = self.model(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(y_pred, y) + self._ewc_penalty()
                    loss.backward()
                    # accumulate fisher squares for this experience
                    if self._ewc_running_squares is not None:
                        k = 0
                        for p in self.model.parameters():
                            if not p.requires_grad:
                                continue
                            self._ewc_running_squares[k] += p.grad.detach() ** 2
                            k += 1
                        self._ewc_running_count += 1
                    optimizer.step()

                elif self.buffer_type == "lwf":
                    # ---- LwF (no memory): distill from frozen teacher on *current data* ----
                    y_pred = self.model(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(y_pred, y)  # label loss
                    if self._teacher is not None:
                        with torch.no_grad():
                            t_logits = self._teacher(x)  # teacher logits
                        # simple MSE on logits; could use temperature if desired
                        distill = nn.functional.mse_loss(
                            y_pred / self.LWF_TEMPERATURE,
                            t_logits / self.LWF_TEMPERATURE,
                        )
                        loss = (1.0 - self.LWF_ALPHA) * loss + self.LWF_ALPHA * distill
                    loss.backward()
                    optimizer.step()

                else:
                    # ---- No buffer ----
                    y_pred = self.model(x)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()

                # ---- batch metrics ----
                acc = metrics.accuracy(y_pred, y, threshold=0.5)
                epoch_avg_loss += loss.item()
                epoch_avg_acc += acc

                progress_bar.set_postfix(
                    {
                        "Loss": epoch_avg_loss / (idx + 1),
                        "Acc": epoch_avg_acc / (idx + 1),
                    }
                )
                progress_bar.update(1)

            progress_bar.close()

            if on_epoch_end is not None:
                try:
                    on_epoch_end(epoch)
                except Exception as e:
                    print(
                        f"[WARN] on_epoch_end callback failed at epoch {epoch+1}: {e}"
                    )

        # ---- End of experience hooks ----
        if self.buffer_type == "ogd":
            self._ogd.finalize_and_store()

        if self.buffer_type == "ewc":
            # Build Fisher diag and store snapshot
            if self._ewc_running_squares is not None and self._ewc_running_count > 0:
                # Average accumulated squares
                avg_squares = [
                    s / float(self._ewc_running_count)
                    for s in self._ewc_running_squares
                ]
                fisher: Dict[str, torch.Tensor] = {}
                idx = 0
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    fisher[n] = avg_squares[idx].detach().clone()
                    idx += 1
                theta = self._param_dict()
                self._ewc_tasks.append({"theta": theta, "fisher": fisher})
            self._ewc_running_squares = None
            self._ewc_running_count = 0

        if self.buffer_type == "lwf":
            # Update teacher snapshot to the just-trained student
            self._teacher = deepcopy(self.model).eval()
            for p in self._teacher.parameters():
                p.requires_grad = False
