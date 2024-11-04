from typing import Union

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: tuple,
        activation_type: str,
        norm_type: str,
        stride: int = 1,
        padding: Union[str, int] = "same",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_channels: int = input_channels
        self.output_channels: int = output_channels
        self.kernel_size: tuple = kernel_size
        self.stride: int = stride
        self.padding: Union[str, int] = padding

        self.encoder = nn.Conv2d(
            input_channels, output_channels, kernel_size, stride, padding
        )
        self.activation = getattr(nn, activation_type)()
        # self.norm = getattr(nn, norm_type)()

    def forward(self, x):
        """
        input:  B x 2 x T x F
        output: B x C x T x F
        """
        x = self.encoder(x)
        # is act and norm required???
        # x = self.activation(x)
        # x = self.norm(x)
        return x


class RTFS_AudioEncoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int = 128,
        win_length: int = 256,
        output_channels: int = 256,
        kernel_size: tuple = (3, 3),
        activation_type="ReLU",
        norm_type="LayerNorm",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.output_channels: int = output_channels
        self.kernel_size: tuple = kernel_size

        self.activation_type = activation_type
        self.norm_type = norm_type

        self.conv_enocder: nn.Module = ConvEncoder(
            input_channels=2,
            output_channels=self.output_channels,
            kernel_size=self.kernel_size,
            activation_type=self.activation_type,
            norm_type=self.norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: tensor with audio seqs: B x L
        Output: B x C x T x F
        """

        x_spectr = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        x_spectr = torch.stack((x_spectr.real, x_spectr.imag), 1).transpose(
            2, 3
        )  # B x 2 x F x T -> B x 2 x T x F
        return self.conv_enocder(x_spectr)
