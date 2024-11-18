import torch
import torch.nn as nn

from .conv_blocks import ConvBlockWithActivation
from .rtfs_block import RTFSBlock


class CAF(nn.Module):
    def __init__(self, audio_channels: int, video_channels: int, n_heads=8):
        super(CAF, self).__init__()

        self.nheads = n_heads

        self.P1 = ConvBlockWithActivation(
            in_channels=audio_channels,
            out_channels=audio_channels,
            kernel_size=1,
            is_conv_2d=True,
            groups=audio_channels,
            activation_function=torch.nn.Identity,
        )
        self.P2 = ConvBlockWithActivation(
            in_channels=audio_channels,
            out_channels=audio_channels,
            kernel_size=1,
            is_conv_2d=True,
            groups=audio_channels,
        )

        self.F1 = ConvBlockWithActivation(
            in_channels=video_channels,
            out_channels=audio_channels * n_heads,
            kernel_size=1,
            is_conv_2d=False,
            groups=audio_channels,
            activation_function=torch.nn.Identity,
        )
        self.F2 = ConvBlockWithActivation(
            in_channels=video_channels,
            out_channels=audio_channels,
            kernel_size=1,
            is_conv_2d=False,
            groups=audio_channels,
            activation_function=torch.nn.Identity,
        )

    def forward(
        self, audio_features: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        """
        audio_features: B x Ca x Ta x F
        video_features: B x Cv x Ta
        """

        B, Ca, Ta, F = audio_features.shape

        audio_value = self.P1(audio_features)  # B x Ca x Ta x F
        audio_gate = self.P2(audio_features)  # B x Ca x Ta x F

        video_attn = self.F1(video_features)

        video_attn = video_attn.reshape(B, Ca, self.nheads, -1).mean(dim=2)
        video_attn = torch.softmax(video_attn, -1)
        video_attn = torch.nn.functional.interpolate(video_attn, size=Ta)

        video_key = self.F2(video_features)
        video_key = torch.nn.functional.interpolate(video_key, size=Ta)

        return (
            video_key.unsqueeze(-1) * audio_gate
            + video_attn.unsqueeze(-1) * audio_value
        )


class SeparationNetwork(nn.Module):
    def __init__(
        self,
        audio_channels: int,
        video_channels: int,
        audio_network: nn.Module = RTFSBlock,
        video_network: nn.Module = RTFSBlock,
        ap_hid_channels=64,
        audio_repeats: int = 4,
    ) -> None:
        super(SeparationNetwork, self).__init__()
        self.audio_network = audio_network(
            in_channels=audio_channels, hid_channels=ap_hid_channels, is_conv_2d=True
        )
        self.video_network = video_network

        self.audio_bn_chan = audio_channels
        self.video_bn_chan = video_channels

        self.audio_repeats = audio_repeats

        self.CAE = CAF(
            audio_channels, video_channels
        )  # CHECK IF IT IS APPROPRIATE CHANNELS

    def forward(
        self, audio_features: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        audio_features_residual = audio_features

        audio_features = self.audio_network(audio_features)

        video_features = self.video_network(video_features)

        audio_features = self.CAE(
            audio_features=audio_features, video_features=video_features
        )

        for j in range(self.audio_repeats):
            audio_features = self.audio_net(audio_features + audio_features_residual)

        return audio_features
