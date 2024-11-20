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
        rtfs_repeats: int = 4,
        rtfs_hid_channels: int = 64,
        dual_path_hidden_channels: int = 32,
        dual_path_num_layers: int = 4,
        dual_path_rnn_type: nn.Module = SRU,
        attention2d_features_dim: int = 64,
        attention2d_hidden_channels: int = 4,
        attention2d_num_heads: int = 4,
        fusion_num_heads: int = 8,
    ) -> None:
        super(SeparationNetwork, self).__init__()
        self.audio_network = RTFSBlock(
            in_channels=audio_channels,
            hid_channels=rtfs_hid_channels,
            kernel_size=4,
            stride=2,
            downsample_layers_count=2,
            is_conv_2d=True,
            attention_layers=nn.Sequential(
                DualPathSRU(
                    in_channels=rtfs_hid_channels,
                    hidden_channels=dual_path_hidden_channels,
                    kernel_size=8,
                    num_layers=dual_path_num_layers,
                    stride=1,
                    bidirectional=True,
                    apply_to_time=False,
                    rnn_type=dual_path_rnn_type,
                ),
                DualPathSRU(
                    in_channels=rtfs_hid_channels,
                    hidden_channels=dual_path_hidden_channels,
                    kernel_size=8,
                    stride=1,
                    num_layers=dual_path_num_layers,
                    bidirectional=True,
                    apply_to_time=True,
                    rnn_type=dual_path_rnn_type,
                ),
                Attention2D(
                    in_channels=rtfs_hid_channels,
                    features_dim=attention2d_features_dim,
                    hidden_channels=attention2d_hidden_channels,
                    num_heads=attention2d_num_heads,
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
                    in_channels=rtfs_hid_channels,
                    kernel_size=3,
                    num_heads=8,
                    dropout=0.1,
                ),
            ),
        )
        self.rtfs_repeats = rtfs_repeats

        self.caf = CAF(
            audio_channels=audio_channels,
            video_channels=video_channels,
            num_heads=fusion_num_heads,
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
