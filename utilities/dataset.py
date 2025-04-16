import math
import typing

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


########### BASE ContinualLearning-Data
class ContinualLearningData(Dataset):
    def __init__(
        self,
        data_path: str,
        target_columns: list[str],
        info_columns: list[str],
        transforms: typing.Any | None = None,
        output_type: str = "torch",
        *args,
        **kwargs,
    ) -> None:
        # Initialization of input parameters
        self.data_path = data_path
        self.target_columns = target_columns
        self.info_columns = info_columns
        self.transforms = transforms
        self.output_type = output_type

        # Read a few rows to get the column names and the list of categorical columns
        data_chunk = pd.read_csv(self.data_path, skiprows=0, nrows=10, header=0)
        self.column_names = data_chunk.columns.values
        self.d = (
            len(self.column_names) - len(self.target_columns) - len(self.info_columns)
        )

    def __call__(self, t: None = None):
        return None

    def input_output_split(
        self, t: int = None
    ) -> tuple[torch.Tensor] | tuple[np.ndarray]:
        """
        From the pandas DataFrame "data", separate the input and the output as pytorch
        tensors named "x" and "y", where "x" represents all the columns at the exception
        of the last, while "y" is the last column.

        NOTE: If data has shape (N, d+1), then:
            - x has shape (N, d) and,
            - y has shape (N, 1).
        """
        data = self.__call__(t)

        x = torch.tensor(
            data.drop(columns=self.target_columns + self.info_columns).to_numpy()
        )
        y = torch.tensor(data[self.target_columns].to_numpy())

        if self.transforms:
            x = self.transforms(x)

        if self.output_type == "numpy":
            return x.numpy(), y.numpy()
        elif self.output_type == "torch":
            return x, y
        else:
            raise NotImplementedError

    def train_test_split(self, x, y, train_split=0.8, shuffle=True):
        # Compute train and test shapes based on inputs
        N, d = x.shape
        N_train = int(N * train_split)

        # Randomly sample indices
        idx = np.arange(0, N, step=1)
        if shuffle:
            np.random.shuffle(idx)

        # Extract train and test set
        x_train, y_train = x[idx[:N_train], :], y[idx[:N_train]]
        x_test, y_test = x[idx[N_train:], :], y[idx[N_train:]]
        return (x_train, y_train), (x_test, y_test)

    def create_dataloader(self, xt, yt, batch_size=128, shuffle=True):
        """
        Given the tensors xt and yt of shapes (N, d) and (N, 1) respectively, return the
        dataloader associated with them with given batch_size
        """
        Dt = TensorDataset(xt, yt)
        return DataLoader(Dt, batch_size=batch_size, shuffle=shuffle)


########### ContinualFlow
class ContinualFlow(ContinualLearningData):
    def __init__(
        self,
        data_path: str,
        target_columns: list[str],
        info_columns: list[str],
        n_data: int,
        chunk_size: int,
        stride: int | None = None,
        transforms: typing.Any | None = None,
        output_type: str = "torch",
    ) -> None:
        super().__init__(
            data_path, target_columns, info_columns, transforms, output_type
        )

        self.n_data = n_data
        self.chunk_size = chunk_size
        if stride is None:
            stride = chunk_size
        self.stride = stride

        self.timesteps = torch.arange(
            0, math.ceil((n_data - chunk_size) / stride) + 1, step=1
        )

    def __call__(self, t=None):
        # Loading chunk in memory (if required)
        if self.chunk_size == -1:
            self.data = pd.read_csv(self.data_path)
        else:
            self.data = pd.read_csv(
                self.data_path,
                skiprows=t * self.stride,
                nrows=self.chunk_size,
                header=0,
                names=self.column_names,
            )
        return self.data


########### ContinualDaily
class ContinualDaily(ContinualLearningData):
    def __init__(
        self,
        data_path: str,
        target_columns: list[str],
        info_columns: list[str],
        n_data: int,
        transforms: typing.Any | None = None,
        output_type: str = "torch",
    ) -> None:
        super().__init__(
            data_path, target_columns, info_columns, transforms, output_type
        )
        self.n_data = n_data

        # Load data in memory
        self.data = pd.read_csv(
            self.data_path,
        )

        # Define a list of days to load
        self.timesteps = self.data["Day"].unique()

    def __call__(self, day):
        # Filter data
        self.daily_data = self.data[self.data["Day"] == self.timesteps[day]]

        return self.daily_data
