from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    def __init__(self, input_data: Tensor, truth_data: Tensor) -> None:
        self.input: Tensor = input_data.clone().detach()
        self.truth: Tensor = truth_data.clone().detach()

    def to(self, device: torch.device | str) -> None:
        self.input = self.input.to(device)
        self.truth = self.truth.to(device)

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.truth[index]

    def __getstate__(self) -> dict[str, Any]:
        states = self.__dict__.copy()
        states['input'] = self.input.cpu()
        states['truth'] = self.truth.cpu()
        return states


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.phase_encoder = nn.Sequential(
            nn.Linear(48, 40),
            nn.ReLU(),
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 16)
        )

        self.amplitude_encoder = nn.Sequential(
            nn.Linear(48, 40),
            nn.ReLU(),
            nn.Linear(40, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 16)
        )

        self.ap_encoder = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.ap_decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.output = nn.Linear(6, 2)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        is_training = len(x.shape) == 3
        start_dim = 1 if is_training else 0

        phases = x[:, 0, :] if is_training else x[0, :]
        amplitudes = x[:, 1, :] if is_training else x[1, :]

        encoded_phases: Tensor = self.phase_encoder(phases)
        encoded_amplitudes: Tensor = self.amplitude_encoder(amplitudes)

        # print(encoded_amplitudes.shape, encoded_phases.shape)

        ap = torch.cat([encoded_phases, encoded_amplitudes], dim=start_dim)

        # print(ap.shape)

        x: Tensor = self.ap_encoder(ap)
        # print(x.shape)
        x: Tensor = self.ap_decoder(x)
        # print(x.shape)

        x = x[:, None, :]
        # print(x.shape)
        x: Tensor = self.conv(x)
        # print(x.shape)
        # x = torch.squeeze(x)
        # print(5, x.shape)

        # x = torch.flatten(x, start_dim=1)
        x = x.transpose(1, 2)
        #
        # print(6, x.shape)

        x = self.fc(x)

        x = x.transpose(1, 2)

        x = self.output(x)

        # print('out', x.shape)

        return x
