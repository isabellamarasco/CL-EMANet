import torch
import torch.nn as nn
from tqdm import tqdm

from materials import buffer as buff
from utilities import metrics

from . import normalizers


class FCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 128,
        dropout_rate: int = 0.5,
    ) -> None:
        """
        Implements a very simple fully-connected neural network, with the provided
        of layers, each with the given number of neurons. To reduce overfitting,
        Dropout layers are employed, with the specified rate. Note that the final layer
        of the network is a Linear layer with no activation, therefore it is recommended
        to apply either sigmoid or softmax layers after the newtork output.

        Also, recall to set the model in "evaluation" mode before using it for testing,
        to deactivate Dropout layers, by calling the model.eval() method.

        Args:
            input_dim (int): The number of input neurons. It should correspond to the
                            number of input features in the data.
            output_dim (int): The number of output neurons. It should correspond to the
                            number of classes.
            n_layers (int): The amount of fully-connected layers to use. Default: 4.
            hidden_dim (int): The amount of neurons to use in the hidden layers of the
                            network. Default: 128.
            dropout_rate (float): The percentage of neurons deactivated during training.
                            Default: 0.5.
        """
        super().__init__()

        # Defining fully-connected layers
        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
            + [nn.Linear(hidden_dim, output_dim)]
        )

        # Set parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Define dropout
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.fc_layers[0](x))
        for i in range(1, self.n_layers - 1):
            x = self.dropout(x)
            x = nn.ReLU()(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)  # IMPORTANT: No Dropout on last layer
        return x


class EMAFCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 4,
        hidden_dim: int = 128,
        dropout_rate: float = 0.5,
        eta: float = 0.99,
    ) -> None:
        """
        Implements the proposed EMAFCNet, with the provided number of layers,
        each with the given number of neurons. It has the same structure of FCNet, but
        the input is pre-processed by passing through the EMA normalization layer.

        Args:
            input_dim (int): The number of input neurons. It should correspond to the
                            number of input features in the data.
            output_dim (int): The number of output neurons. It should correspond to the
                            number of classes.
            n_layers (int): The amount of fully-connected layers to use. Default: 4.
            hidden_dim (int): The amount of neurons to use in the hidden layers of the
                            network. Default: 128.
            dropout_rate (float): The percentage of neurons deactivated during training.
                            Default: 0.5.
        """
        super().__init__()

        # Initialize layers
        self.normalization_layer = normalizers.EMANet(
            input_dim=input_dim,
            n_type="minmax",
            use_running_stats=True,
            momentum=eta,
            eps=1e-6,
        )
        self.model = FCNet(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x = self.normalization_layer(x)

        # Process input
        x = self.model(x)
        return x
