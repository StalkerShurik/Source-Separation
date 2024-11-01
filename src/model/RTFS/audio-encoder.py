import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation_type: str,
        norm_type: str,
        stride: int = 1,
        padding: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding

        self.encoder = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.activation = getattr(nn, activation_type)
        self.norm = getattr(nn, norm_type)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class RTFS_AudioEncoder(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.win_length: int = win_length

        self.conv_enocder: nn.Module = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: tensor with audio seqs: Batch x Seq
        """

        x_spectr = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        # cringe should be fixed
        x_spectr = torch.stack((x_spectr.real, x_spectr.imag), 1)
        print(x_spectr.shape)
        return self.conv_enocder(x_spectr)
