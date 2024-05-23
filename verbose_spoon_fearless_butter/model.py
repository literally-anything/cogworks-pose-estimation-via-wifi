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
            nn.Linear(192, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )

        self.phase_diff_encoder = nn.Sequential(
            nn.Linear(192, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )

        self.amplitude_encoder = nn.Sequential(
            nn.Linear(192, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )

        self.ap_encoder = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )

        # self.ap_decoder = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(16, 24),
        #     nn.ReLU(),
        #     nn.Linear(24, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32)
        # )

        # self.conv = nn.Sequential(
        #     nn.Conv1d(1, 16, 4),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Conv1d(16, 32, 3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2)
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(64, 24),
        #     nn.ReLU(),
        #     nn.Linear(24, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1)
        # )

        # self.output = nn.Linear(6, 2)

        self.last_phases: Tensor | None = None

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        phases = x[:, :, 0, :].flatten(start_dim=1)
        amplitudes = x[:, :, 1, :].flatten(start_dim=1)

        if self.last_phases is not None:
            phase_differences = phases - self.last_phases
            encoded_phases: Tensor = self.phase_encoder(phases)
            encoded_phase_diffs: Tensor = self.phase_diff_encoder(phase_differences)
            encoded_amplitudes: Tensor = self.amplitude_encoder(amplitudes)

            ap = torch.cat([encoded_phases, encoded_phase_diffs, encoded_amplitudes], dim=1)

            x: Tensor = self.ap_encoder(ap)
            # x: Tensor = self.ap_decoder(x)

            # x = x[:, None, :]
            # print('1', x.shape)
            # x: Tensor = self.conv(x)

            # x = x.transpose(1, 2)
            #
            # x = self.fc(x)
            #
            # x = x.transpose(1, 2)
            #
            # x = self.output(x)
        self.last_phases = phases

        return x
