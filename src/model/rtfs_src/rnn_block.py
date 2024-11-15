import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sru import SRU


class DualPathRNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
        num_layers: int = 1,
        bidirectional: bool = True,
        apply_to_audio: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(DualPathRNN, self).__init__()
        # conv params
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # sru params
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.apply_to_audio = apply_to_audio

        self.unfolded_channels = self.in_channels * self.kernel_size
        self.rnn_out_channels = (
            self.hidden_channels * 2 if bidirectional else self.unfolded_channels
        )

        # TODO: fix
        self.norm = nn.LayerNorm([self.in_channels, 1, 1])

        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(self.stride, 1))

        self.sru = SRU(
            input_size=self.unfolded_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        self.linear = nn.ConvTranspose1d(
            self.rnn_out_channels,
            self.in_channels,
            self.kernel_size,
            stride=self.stride,
        )

    def _get_shape_after_conv(self, size_: int) -> int:
        return (
            np.ceil((size_ - self.kernel_size) / self.stride) * self.stride
            + self.kernel_size
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass  # todo
