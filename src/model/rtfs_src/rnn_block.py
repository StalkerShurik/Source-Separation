import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sru import SRU


# CHECKED
class DualPathSRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 8,
        stride: int = 1,
        num_layers: int = 1,
        bidirectional: bool = True,
        apply_to_time: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(DualPathSRU, self).__init__()
        # conv params
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # sru params
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.apply_to_time = apply_to_time

        self.unfolded_channels = self.in_channels * self.kernel_size
        self.rnn_out_channels = (
            self.hidden_channels * 2 if bidirectional else self.unfolded_channels
        )

        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(self.stride, 1))

        self.sru = SRU(
            input_size=self.unfolded_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        )

        self.conv = nn.ConvTranspose1d(
            self.rnn_out_channels,
            self.in_channels,
            self.kernel_size,
            stride=self.stride,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features dim: B x D x T x F
        if self.apply_to_time:
            features = features.permute(0, 1, 3, 2)  # B x D x F x T

        features_residual = features

        batch_dim, channels_dim, time_dim, features_dim = features.shape

        # (B * T) x D x F x 1
        features = features.permute(0, 2, 1, 3).reshape(
            batch_dim * time_dim, channels_dim, features_dim, 1
        )  # according to the artilce we apply layers for every time moment independently

        features = self.unfold(features)  # (B * T) x 8D x F'

        # (B * T) x 2h x F'
        features = self.sru(features.permute(0, 2, 1))[0].permute(0, 2, 1)  # apply SRU

        # (B * T) x D x F'

        features = self.conv(features)

        features = features.reshape(
            batch_dim, time_dim, channels_dim, features_dim
        ).permute(
            0, 2, 1, 3
        )  # separate batch and time

        features += features_residual

        if self.apply_to_time:
            features = features.permute(0, 1, 3, 2)

        return features
