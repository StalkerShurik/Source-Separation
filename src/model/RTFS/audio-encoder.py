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
        activation_type=nn.ReLU,
        norm_type=nn.BatchNorm2d,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.output_channels: int = output_channels  # C
        self.kernel_size: tuple = kernel_size

        self.activation_type: nn.Module = activation_type
        self.norm_type: nn.Module = norm_type

        self.conv_enocder: nn.Module = nn.Sequential(
            nn.Conv2d(
                2, self.output_channels, self.kernel_size, stride=1, padding="same"
            ),
            self.activation_type(inplace=True),
            self.norm_type(self.output_channels),
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
        return self.conv_enocder(x_spectr)  # B x 2 x T x F -> B x C x T x F


enc = RTFS_AudioEncoder(256)
print(enc(torch.rand(8, 32000)).shape)
