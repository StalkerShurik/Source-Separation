import torch
import torch.nn as nn

from .conv_blocks import ConvBlockWithActivation


class SpectralSourceSeparationDecoder(nn.Module):
    # Contains S3 and decoder

    def __init__(
        self,
        input_channels: int,
        hop_length: int = 128,
        features: int = 256,
        length=32000,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.length = length

        self.preact = nn.PReLU()

        self.M: nn.Module = ConvBlockWithActivation(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            is_conv_2d=True,
        )
        self.input_channels: int = input_channels

        self.map_to_stft: nn.Module = ConvBlockWithActivation(
            in_channels=input_channels,
            out_channels=2,
            kernel_size=3,
            padding="same",
            is_conv_2d=True,
            activation_function=nn.Identity,
        )

        self.features = features
        self.hop_length = hop_length

    def forward(self, processed_audio: torch.Tensor, original_audio: torch.Tensor):
        processed_audio = self.preact(processed_audio)
        processed_audio = self.M(processed_audio)

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
            window=torch.hann_window(window_length=self.features),
            length=self.length,
        )

        return decoded_vaw
