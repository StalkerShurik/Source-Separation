import torch
import torch.nn as nn


class RTFS_AudioEncoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int = 128,
        win_length: int = 256,
        output_channels: int = 256,
        kernel_size: tuple = (3, 3),
        activation_type: nn.Module = nn.ReLU,
        norm_type: nn.Module = nn.BatchNorm2d,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length

        self.layers: nn.Module = nn.Sequential(
            nn.Conv2d(
                2, output_channels, kernel_size, stride=1, padding="same", bias=False
            ),
            norm_type(output_channels),
            activation_type(inplace=True),
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
        return self.layers(x_spectr)  # B x 2 x T x F -> B x C x T x F
