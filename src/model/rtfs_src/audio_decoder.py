import torch
import torch.nn as nn

from .conv_blocks import ConvBlockWithActivation


class SpectralSourceSeparationDecoder(nn.Module):
    # Contains S3 and decoder

    def __init__(
        self,
        input_channels: int,
        hop_length: int = 128,
        win_length: int = 256,
        output_channels: int = 256,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

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
            kernel_size=1,
            is_conv_2d=True,
            activation_function=nn.Identity,
        )

        self.hop_length = hop_length
        self.win_length = win_length
        self.output_channels = output_channels

    def forward(self, processed_audio: torch.Tensor, original_audio: torch.Tensor):
        processed_audio = nn.PReLU(processed_audio)
        processed_audio = self.M(processed_audio)

        processed_real = processed_audio[:, : self.input_channels // 2]
        processed_img = processed_audio[:, self.input_channels // 2 :]

        original_real = original_audio[:, : self.input_channels // 2]
        original_img = original_audio[:, self.input_channels // 2 :]

        decoded_real = processed_real * original_real - processed_img * original_img
        decoded_img = processed_real * original_img + processed_img * original_real

        decoded = torch.concat([decoded_real, decoded_img], dim=1)

        decoded = self.map_to_stft(decoded)

        decoded_vaw = torch.istft(
            decoded,
            n_fft=self.output_channels,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        # TODO nasrano

        return decoded_vaw
