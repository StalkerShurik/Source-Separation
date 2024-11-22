import torch
import torch.nn as nn

from .conv_blocks import ConvBlockWithActivation


class SpectralSourceSeparationDecoder(nn.Module):
    # Contains S3 and decoder

    def __init__(
        self,
        input_channels: int,
        hop_length: int,
        features: int,
        length: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.length = length

        self.preact = nn.PReLU()

        self.masker: nn.Module = ConvBlockWithActivation(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            is_conv_2d=True,
        )
        self.input_channels: int = input_channels

        self.map_to_stft: nn.Module = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=2,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )

        self.features = features
        self.hop_length = hop_length

        self.window = nn.parameter.Parameter(
            torch.hann_window(window_length=self.features), requires_grad=False
        )

    def forward(self, processed_audio: torch.Tensor, original_audio: torch.Tensor):
        processed_audio = self.preact(processed_audio)
        processed_audio = self.masker(processed_audio)

        processed_real = processed_audio[:, : self.input_channels // 2]
        processed_img = processed_audio[:, self.input_channels // 2 :]

        original_real = original_audio[:, : self.input_channels // 2]
        original_img = original_audio[:, self.input_channels // 2 :]

        decoded_real = processed_real * original_real - processed_img * original_img
        decoded_img = processed_real * original_img + processed_img * original_real

        decoded = torch.concat([decoded_real, decoded_img], dim=1)

        decoded = self.map_to_stft(decoded)

        decoded_complex = torch.complex(decoded[:, 0], decoded[:, 1]).transpose(-1, -2)

        decoded_vaw = torch.istft(
            decoded_complex,
            n_fft=self.features,
            hop_length=self.hop_length,
            win_length=self.features,
            window=self.window,
            length=self.length,
        )

        return decoded_vaw
