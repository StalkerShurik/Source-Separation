import torch
import torch.nn as nn
from sru import SRU

from .attention_block import Attention2D, GlobalAttention1d
from .fusion import CAF
from .rnn_block import DualPathSRU
from .rtfs_block import RTFSBlock


class SeparationNetwork(nn.Module):
    def __init__(
        self,
        audio_channels: int,
        video_channels: int,
        rtfs_hid_channels: int,
        rtfs_repeats: int,
        downsample_rate_2d: int,
        dual_path_rnn_params: dict,
        attention2d_params: dict,
        attention1d_params: dict,
        CAF_params: dict,
    ) -> None:
        super(SeparationNetwork, self).__init__()
        self.audio_network = RTFSBlock(
            in_channels=audio_channels,
            hid_channels=rtfs_hid_channels,
            kernel_size=4,
            stride=2,
            downsample_layers_count=downsample_rate_2d,
            is_conv_2d=True,
            attention_layers=nn.Sequential(
                DualPathSRU(
                    **dual_path_rnn_params,
                    rnn_type=SRU,
                    kernel_size=8,
                    stride=1,
                    bidirectional=True,
                    apply_to_time=False,
                ),
                DualPathSRU(
                    **dual_path_rnn_params,
                    rnn_type=SRU,
                    kernel_size=8,
                    stride=1,
                    bidirectional=True,
                    apply_to_time=True,
                ),
                Attention2D(
                    **attention2d_params,
                ),
            ),
        )
        self.video_network = RTFSBlock(
            in_channels=video_channels,
            hid_channels=rtfs_hid_channels,
            kernel_size=3,
            stride=2,
            downsample_layers_count=4,
            is_conv_2d=False,
            attention_layers=nn.Sequential(
                GlobalAttention1d(
                    **attention1d_params,
                    kernel_size=3,
                ),
            ),
        )
        self.rtfs_repeats = rtfs_repeats

        self.caf = CAF(
            **CAF_params,
            audio_channels=audio_channels,
            video_channels=video_channels,
        )

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
