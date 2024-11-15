import torch
import torch.nn as nn

from .rtfs_block import RTFSBlock


class MultiModalBlock(nn.Module):
    def __init__(self, audio_channels: int, video_channels: int):
        super(MultiModalBlock, self).__init__()
        # TODO

    def forward(
        self, audio_features: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        # TODO
        pass


class SeparationNetwork(nn.Module):
    def __init__(
        self,
        audio_network: RTFSBlock,
        video_network: RTFSBlock,
        audio_channels: int,
        video_channels: int,
        audio_repeats: int = 4,
    ) -> None:
        super(SeparationNetwork, self).__init__()
        self.audio_network = audio_network
        self.video_network = video_network

        self.audio_bn_chan = audio_channels
        self.video_bn_chan = video_channels

        self.audio_repeats = audio_repeats

        self.crossmodal_fusion = MultiModalBlock()  # TODO

    def forward(
        self, audio_features: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        audio_features_residual = audio_features

        # cross modal fusion
        audio_features = self.audio_network(audio_features)
        video_features = self.video_network(video_features)

        audio_features = self.crossmodal_fusion(
            audio_features=audio_features, video_features=video_features
        )

        for j in range(self.audio_repeats):
            audio_features = self.audio_net(audio_features + audio_features_residual)

        return audio_features
