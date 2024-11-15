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

    # def _get_shape_after_conv(self, size_: int) -> int:
    #     return int(
    #         np.ceil((size_ - self.kernel_size) / self.stride) * self.stride
    #         + self.kernel_size
    #     )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_residual = features

        print(f"input shape {features.shape}")

        batch_size, channels, time, features_size = features.shape

        features = features.permute(0, 2, 1, 3).reshape(
            batch_size * time, channels, features_size, 1
        )

        features = self.unfold(features)

        print(f"unfolded shape {features.shape}")

        features = self.sru(features.permute(0, 2, 1))[0]

        print(f"sru shape {features.shape}")

        features = self.linear(features.permute(0, 2, 1))

        print(f"convT shape {features.shape}")

        features = features.reshape(batch_size, time, channels, features_size).permute(
            0, 2, 1, 3
        )

        features += features_residual

        print(f"final shape {features.shape}")

        return features


def test_shape():
    batch_size = 7
    channels = 11
    time = 17
    features = 19

    input = torch.rand(batch_size, channels, time, features)

    layer = DualPathRNN(channels, 23, apply_to_audio=True)

    out_bad = layer.forward_bad(input)

    out = layer(input)

    print(torch.linalg.norm(out_bad - out) / torch.linalg.norm(out_bad))


test_shape()
