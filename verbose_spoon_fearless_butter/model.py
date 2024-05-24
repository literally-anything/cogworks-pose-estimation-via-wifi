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
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 160),
            nn.ReLU(),
            nn.Linear(160,192)
        )

        self.phase_diff_encoder = nn.Sequential(
            nn.Linear(192, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 160),
            nn.ReLU(),
            nn.Linear(160, 192)
        )

        self.amplitude_encoder = nn.Sequential(
            nn.Linear(192, 160),
            nn.ReLU(),
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 96),
            nn.ReLU(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 160),
            nn.ReLU(),
            nn.Linear(160, 192)
        )

        self.ap_encoder = nn.Sequential(
            nn.Linear(576, 480),
            nn.ReLU(),
            nn.Linear(480, 384),
            nn.ReLU(),
            nn.Linear(384, 288),
            nn.ReLU(),
            nn.Linear(288, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 96),
            nn.ReLU(),
            nn.Linear(96, 192),
            nn.ReLU(),
            nn.Linear(192, 288),
            nn.ReLU(),
            nn.Linear(288, 384),
            nn.ReLU(),
            nn.Linear(384, 480),
            nn.ReLU(),
            nn.Linear(480, 576),
            nn.ReLU(),
            nn.Linear(576, 480),
            nn.ReLU(),
            nn.Linear(480, 384),
            nn.ReLU(),
            nn.Linear(384, 288),
            nn.ReLU(),
            nn.Linear(288, 192),
            nn.ReLU(),
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

        if self.last_phases is None:
            self.last_phases = phases

        phase_differences = phases - self.last_phases
        encoded_phases: Tensor = self.phase_encoder(phases)
        encoded_phase_diffs: Tensor = self.phase_diff_encoder(phase_differences)
        encoded_amplitudes: Tensor = self.amplitude_encoder(amplitudes)

        # print(1, encoded_amplitudes.shape, encoded_phases.shape, encoded_phase_diffs.shape)

        ap = torch.cat([encoded_phases, encoded_phase_diffs, encoded_amplitudes], dim=1)

        # print(2, ap.shape)

        x: Tensor = self.ap_encoder(ap)

        # print(3, x.shape)

        self.last_phases = phases

        return x
