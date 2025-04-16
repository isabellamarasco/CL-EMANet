import torch
from torch import nn


class SimpleNormalization:
    def __init__(self, normalization_type: str):
        """
        A module implementing basics normalization methods for given data.

        Args:
            normalization_type (str): The type of normalization to consider. Currently
                                    available: {"no", "global", "local", "up_to", "emanet"}.
        """
        self.normalization_type = normalization_type
        self.normalizer = self.get_normalization_by_type()

        # Initialize Mx, mx
        self.Mx, self.mx = None, None

    def get_normalization_by_type(self):
        """
        Provide normalization functions given the required type. In particular:
        - "local", "up_to" -> minmax normalization.
        - "no", "global", "emanet" -> identity normalization (i.e. do nothing).
        """
        if self.normalization_type.lower() in ["local", "up_to"]:
            return self.minmax
        elif self.normalization_type.lower() in ["no", "global", "emanet"]:
            return self.identity
        else:
            raise NotImplementedError

    def identity(self, x, M=None, m=None):
        """
        Apply identity normalization to input tensor x, i.e. does not modify x.
        """
        return x

    def minmax(self, x: torch.Tensor, M=None, m=None):
        """
        Apply min-max normalization to input tensor x. If M, m are None, then
        the maximum and minimum of x are used as normalization constants.
        """
        if M is None:
            M = x.max(dim=0, keepdim=True)[0]
        if m is None:
            m = x.min(dim=0, keepdim=True)[0]

        # Send to device
        M, m = M.to(torch.float32).to(x.device), m.to(torch.float32).to(x.device)

        return (x - m) / (M - m)

    def get_minmax(self, x: torch.Tensor):
        M = x.max(dim=0, keepdim=True)[0]
        m = x.min(dim=0, keepdim=True)[0]
        return M, m

    def __call__(self, x: torch.Tensor):
        """
        Applies the selected normalization method to a tensor x.

        Args:
            x (torch.Tensor): The tensor on which to apply normalization.
        """
        x_normalized = self.normalizer(x, self.Mx, self.mx)
        return x_normalized

    def update(self, Mx: torch.Tensor, mx: torch.Tensor):
        """
        Update the information of the normalizer based on the type. For example, if
        "local" normalization is used, then this methods override the actual value of
        mx and Mx. If "up_to" is selected instead, this method substitute the values of
        mx and Mx with min(mx, mx') and max(Mx, Mx'), respectively.

        Args:
            Mx (torch.Tensor): A tensor of shape (1, d) containing the Max of each feature.
            Mx (torch.Tensor): A tensor of shape (1, d) containing the Min of each feature.
        """
        # If (Mx, mx) are empty, just set them as (Mx, mx)
        if self.Mx is None:
            assert self.mx is None
            self.Mx = Mx
            self.mx = mx

        # Otherwise, update based on normalization_type
        if self.normalization_type.lower() == "local":
            self.Mx = Mx
            self.mx = mx
        elif self.normalization_type.lower() == "up_to":
            self.Mx = torch.maximum(self.Mx, Mx)
            self.mx = torch.minimum(self.mx, mx)


class EMANet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_type: str = "minmax",
        use_running_stats: bool = True,
        momentum: float = 0.99,
        eps: float = 1e-6,
    ):
        """
        Faster dataset-aware normalization layer using batch statistics.

        Args:
            input_dim (int): Input feature dimensionality.
            n_type (str): Type of normalization employed.
                          Currently available: {"minmax", "gaussian"}. Default: "minmax".
            use_running_stats (bool): Whether running statistics are used instead of
                                      batch. Default: True.
            momentum (float): Decay factor for moving average.
            eps (float): Small value to prevent division by zero.
        """
        super(EMANet, self).__init__()
        self.n_type = n_type
        self.use_running_stats = use_running_stats
        self.eps = eps
        self.momentum = momentum

        # Running statistics for dataset-aware normalization
        self.register_buffer("running_mx", torch.zeros((input_dim,)))
        self.register_buffer("running_Mx", torch.ones((input_dim,)))

        # Learnable scale and bias
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        """
        Fast dataset-aware normalization.

        Args:
            x (torch.Tensor): Shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Normalized output (batch_size, input_dim)
        """
        if self.training:
            # Compute per-batch statistics
            if self.n_type == "minmax":
                batch_mx = x.min(dim=0, keepdim=True)[0]  # Shape: (input_dim,)
                batch_Mx = x.max(dim=0, keepdim=True)[0]  # Shape: (input_dim,)
            elif self.n_type == "gaussian":
                batch_mx = x.mean(dim=0)  # Shape: (input_dim,)
                batch_Mx = x.std(dim=0, unbiased=False)  # Shape: (input_dim,)

            # Update running statistics (EMA) - Detach & Clone to avoid errors
            self.running_mx = self.running_mx * self.momentum + batch_mx * (
                1 - self.momentum
            )
            self.running_Mx = self.running_Mx * self.momentum + batch_Mx * (
                1 - self.momentum
            )

        else:
            # Use precomputed dataset-wide statistics in inference
            batch_mx = self.running_mx
            batch_Mx = self.running_Mx

        # Normalize and apply learnable scale/bias
        if self.n_type.lower() == "minmax":
            if self.use_running_stats:
                norm_x = (x - self.running_mx) / (self.running_Mx - self.running_mx)
            else:
                norm_x = (x - batch_mx) / (batch_Mx - batch_mx)

        elif self.n_type.lower() == "gaussian":
            if self.use_running_stats:
                norm_x = (x - self.running_mx) / (self.running_Mx + self.eps)
            else:
                norm_x = (x - batch_mx) / (batch_Mx + self.eps)

        return self.scale * norm_x + self.bias
