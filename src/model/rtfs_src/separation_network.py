import torch
import torch.nn as nn

from .attention_block import Attention2D, GlobalAttention1d
from .conv_blocks import ConvBlockWithActivation
from .rnn_block import DualPathRNN
from .rtfs_block import RTFSBlock


class CAF(nn.Module):
    def __init__(self, audio_channels: int, video_channels: int, num_headss=8):
        super(CAF, self).__init__()

        self.nheads = num_headss

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
            out_channels=audio_channels * num_headss,
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
        rtfs_repeats: int = 4,
    ) -> None:
        super(SeparationNetwork, self).__init__()
        self.audio_network = RTFSBlock(
            in_channels=audio_channels,
            hid_channels=64,
            kernel_size=4,
            stride=2,
            is_conv_2d=True,
            attention_layers=nn.Sequential(
                DualPathRNN(
                    in_channels=64,
                    hidden_channels=32,
                    kernel_size=8,
                    stride=1,
                    num_layers=4,
                    bidirectional=True,
                    apply_to_time=False,
                ),
                DualPathRNN(
                    in_channels=64,
                    hidden_channels=32,
                    kernel_size=8,
                    stride=1,
                    num_layers=4,
                    bidirectional=True,
                    apply_to_time=True,
                ),
                Attention2D(
                    in_channels=64, features_dim=64, hidden_channels=4, num_heads=4
                ),
            ),
        )
        self.video_network = RTFSBlock(
            in_channels=video_channels,
            hid_channels=64,
            kernel_size=3,
            stride=2,
            is_conv_2d=False,
            attention_layers=nn.Sequential(
                GlobalAttention1d(
                    in_channels=64, kernel_size=3, num_heads=8, dropout=0.1
                ),
            ),
        )
        self.rtfs_repeats = rtfs_repeats

        self.caf = CAF(
            audio_channels=audio_channels,
            video_channels=video_channels,
        )  # CHECK IF IT IS APPROPRIATE CHANNELS

    def forward(
        self, audio_features: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        audio_features_residual = audio_features

        audio_features = self.audio_network(audio_features)

        video_features = self.video_network(video_features)

        audio_features = self.caf(
            audio_features=audio_features, video_features=video_features
        )

        for _ in range(self.rtfs_repeats):
            audio_features = self.audio_network(
                audio_features + audio_features_residual
            )

        return audio_features
