import torch
import torch.nn as nn


class RTFS_AudioEncoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int = 128,
        win_length: int = 256,
        output_channels: int = 256,
        kernel_size: int = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length

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
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
            ),
            # TODO: try to add normalization here too
        )

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        Input: tensor with audio seqs: B x L
        Output: B x C x T x F
        """

        x_spectr = torch.stft(
            raw_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        x_spectr = torch.stack((x_spectr.real, x_spectr.imag), 1).transpose(
            2, 3
        )  # B x 2 x F x T -> B x 2 x T x F
        return self.layers(x_spectr)  # B x 2 x T x F -> B x C x T x F
