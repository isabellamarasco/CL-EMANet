from dataclasses import dataclass

from utilities import miscellaneous


@dataclass
class CICIDS_Config:
    # Support parameters
    device = miscellaneous.get_device()

    # Dataset values
    data_name = "CIC-IDS"  # UNSW-NB15
    n_data = 2_827_876  # 2_540_047
    input_dim = 68  # 53

    # ContinuousFlow parameters
    continuous_flow_type = "daily"  # in {"daily", "flow"}
    normalization_type = "local"  # in {"no", "global", "local", "EMANet", "up_to"}
    buffer_type = "random"  # in {"no", "random", "agem"}

    # (Optional) parameters
    chunk_size = 300_000
    stride = 100_000

    buffer_size = 500_000
    buffer_batch_size = 20_000

    # Model configurations
    n_layers = 4
    hidden_dim = 128
    dropout_rate = 0.5

    # Training parameters
    batch_size = 20_000
    n_epochs = 20
    learning_rate = 5e-4


@dataclass
class UNSWNB15_Config:
    # Support parameters
    device = miscellaneous.get_device()

    # Dataset values
    data_name = "UNSW-NB15"
    n_data = 2_540_047
    input_dim = 51

    # ContinuousFlow parameters
    continuous_flow_type = "flow"  # in {"flow"}
    normalization_type = "up_to"  # in {"no", "global", "local", "EMANet", "up_to"}
    buffer_type = "random"  # in {"no", "random", "agem"}

    # (Optional) parameters
    chunk_size = 600_000
    stride = 600_000

    buffer_size = 500_000
    buffer_batch_size = 20_000

    # Model configurations
    n_layers = 4
    hidden_dim = 128
    dropout_rate = 0.5

    # Training parameters
    batch_size = 20_000
    n_epochs = 20
    learning_rate = 1e-3  # 5e-4
