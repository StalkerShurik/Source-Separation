import typing as tp

import torch
import torch.nn as nn
from sru import SRU

from .attention_block import Attention2D, GlobalAttention1d
from .fusion import CAF
from .rnn_block import DualPathSRU
from .rtfs_block import RTFSBlock


def _get_rnn(rnn_type_str: str):
    possible_types = {"SRU": SRU, "GRU": nn.GRU}
    if rnn_type_str not in possible_types.keys():
        raise Exception("Unsupported RNN type")
    return possible_types[rnn_type_str]


class SeparationNetwork(nn.Module):
    def __init__(
        self,
        audio_channels: int,
        video_channels: int,
        rtfs_repeats: int,
        rnn_type_str: str,
        audio_block_params: dict[str, tp.Any],
        dual_path_rnn_params: dict,
        audio_attention_params: dict[str, tp.Any],
        video_block_params: dict[str, tp.Any],
        video_attention_params: dict[str, tp.Any],
        caf_params: dict[str, tp.Any],
    ) -> None:
        super(SeparationNetwork, self).__init__()

        rnn_type = _get_rnn(rnn_type_str)

        self.audio_network = RTFSBlock(
            in_channels=audio_channels,
            **audio_block_params,
            attention_layers=nn.Sequential(
                DualPathSRU(
                    **dual_path_rnn_params,
                    rnn_type=rnn_type,
                    apply_to_time=False,
                ),
                DualPathSRU(
                    **dual_path_rnn_params,
                    rnn_type=rnn_type,
                    apply_to_time=True,
                ),
                Attention2D(
                    **audio_attention_params,
                ),
            ),
        )
        self.video_network = RTFSBlock(
            in_channels=video_channels,
            **video_block_params,
            attention_layers=nn.Sequential(
                GlobalAttention1d(
                    **video_attention_params,
                ),
            ),
        )
        self.rtfs_repeats = rtfs_repeats

        self.caf = CAF(
            **caf_params,
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
