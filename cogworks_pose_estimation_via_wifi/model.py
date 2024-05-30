from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
from torch.utils.data import Dataset

# from efficient_kan import KANLinear
# nn.Linear = KANLinear


class CSIDataset(Dataset):
    def __init__(self, input_data: Tensor, truth_data: Tensor) -> None:
        self.input: Tensor = input_data.clone().detach()
        self.truth: Tensor = truth_data.clone().detach()

    def to(self, device: torch.device | str) -> None:
        self.input = self.input.to(device)
        self.truth = self.truth.to(device)

    def __len__(self) -> int:
        return len(self.input)

    @jit.export
    def __getitem__(self, index):
        if index > 0:
            last_phases = self.input[index - 1, :, 0:1, :]
        else:
            last_phases = torch.zeros((self.input.shape[1], 1, self.input.shape[3]), device=self.input.device)
        inp = self.input[index]
        diff = inp[:, 0, :] - last_phases
        inp = torch.cat([inp, diff], dim=1)
        return inp, self.truth[index]

    def __getstate__(self) -> dict[str, Any]:
        states = self.__dict__.copy()
        states['input'] = self.input.cpu()
        states['truth'] = self.truth.cpu()
        return states


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.phase_encoder = nn.Sequential(
        #     nn.Linear(192, 160),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(160, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 160),
        #     nn.SELU(),
        #     nn.Linear(160,192)
        # )
        #
        # self.phase_diff_encoder = nn.Sequential(
        #     nn.Linear(192, 160),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(160, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 160),
        #     nn.SELU(),
        #     nn.Linear(160, 192)
        # )
        #
        # self.amplitude_encoder = nn.Sequential(
        #     nn.Linear(192, 160),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(160, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(128, 128),
        #     nn.SELU(),
        #     nn.Linear(128, 160),
        #     nn.SELU(),
        #     nn.Linear(160, 192)
        # )
        #
        # self.ap_encoder = nn.Sequential(
        #     nn.Linear(576, 480),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(480, 384),
        #     nn.SELU(),
        #     nn.Linear(384, 512),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(512, 1024),
        #     nn.SELU(),
        #
        #     nn.Linear(1024, 1024),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(1024, 1024),
        #     nn.SELU(),
        #     nn.Linear(1024, 1024),
        #     nn.SELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(1024, 1024),
        #     nn.SELU(),
        #     nn.Linear(1024, 512)
        # )
        #
        # self.lstm = nn.LSTM(512, 512, 16, dropout=0.4)
        self.lstm = nn.LSTM(3, 64, 32, dropout=0.0, batch_first=True)
        #
        # self.rnn = nn.RNN(512, 512, 16, dropout=0.4)
        self.rnn = nn.RNN(64, 64, 32, dropout=0.0, batch_first=True)
        # self.rnn = nn.RNN(3, 64, 32, dropout=0.0, batch_first=True)
        #
        self.ap_decoder = nn.Sequential(
            nn.Linear(64, 192),
            nn.SELU(),
            nn.Dropout(0.0),
            nn.Linear(192, 96),
            nn.SELU(),
            nn.Linear(96, 48),
            nn.SELU(),
            nn.Linear(48, 24),
            nn.SELU(),
            nn.Dropout(0.0),
            nn.Linear(24, 12),
            nn.SELU(),
            nn.Linear(12, 6),
            nn.SELU(),
            nn.Linear(6, 1)
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

        # self.output = nn.Linear(192, 2)
        self.output = nn.Linear(144, 2)

        # self.last_phases: Tensor | None = None

        self.lstm_last_hidden = None
        self.last_hidden = None

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        phases = x[:, :, 0, :].flatten(start_dim=1)
        amplitudes = x[:, :, 1, :].flatten(start_dim=1)
        phase_differences = x[:, :, 2, :].flatten(start_dim=1)

        # if self.last_phases is None:
        #     self.last_phases = phases

        # print(phases.shape, self.last_phases.shape)
        # phase_differences = phases - self.last_phases
        # self.last_phases = phases
        # encoded_phases: Tensor = self.phase_encoder(phases)
        # encoded_phase_diffs: Tensor = self.phase_diff_encoder(phase_differences)
        # encoded_amplitudes: Tensor = self.amplitude_encoder(amplitudes)

        # print(1, encoded_amplitudes.shape, encoded_phases.shape, encoded_phase_diffs.shape)

        # ap = torch.cat([encoded_phases, encoded_phase_diffs, encoded_amplitudes], dim=1)

        # print(2, ap.shape)

        # x: Tensor = self.ap_encoder(ap)
        # x = torch.cat([phases, amplitudes, phase_differences], dim=1)
        x = torch.stack([phases, amplitudes, phase_differences], dim=2)
        # print(x.shape)

        # if self.lstm_last_hidden is None:
        #     self.lstm_last_hidden = torch.zeros((16, 512), device=x.device)
        x, hidden = self.lstm(x, self.lstm_last_hidden)
        self.lstm_last_hidden = (hidden[0].detach(), hidden[1].detach())
        # print(len(hidden), 'h1')

        # print(x.shape)

        # if self.last_hidden is None:
        #     self.last_hidden = torch.zeros((64, 192*3), device=x.device)
        x, hidden = self.rnn(torch.squeeze(x), self.last_hidden)
        # print(len(hidden), 'h')
        self.last_hidden = hidden.detach()

        # print(x.shape)

        x: Tensor = self.ap_decoder(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)

        # print(x.shape)
        x = self.output(x.squeeze(dim=1))

        # # print(3, x.shape)
        # print(self.last_phases.shape)

        # print(x)

        return x
