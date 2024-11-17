import torch
import torch.nn as nn


class RTFS_AudioEncoder(nn.Module):
    def __init__(
        self,
        hop_length: int = 128,
        features: int = 256,
        output_channels: int = 256,
        kernel_size: int = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hop_length: int = hop_length
        self.features: int = features

        self.layers: nn.Module = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=False,
            ),
            # TODO: try to use Global normalization
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            # nn.Conv2d(
            #     in_channels=output_channels,
            #     out_channels=output_channels,
            #     kernel_size=1,
            #     stride=1,
            # ),
        )

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        Input: tensor with audio seqs: B x L
        Output: B x C x T x F
        """
        x_spectr = torch.stft(
            raw_audio,
            n_fft=self.features,
            hop_length=self.hop_length,
            win_length=self.features,
            window=torch.hann_window(window_length=self.features),
            return_complex=True,
        )

        x_spectr = torch.stack((x_spectr.real, x_spectr.imag), 1).transpose(
            2, 3
        )  # B x 2 x F x T -> B x 2 x T x F

        x_more_channels = self.layers(x_spectr)  # B x 2 x T x F -> B x C x T x F

        return x_more_channels
