import torch

from .rtfs_src.audio_decoder import SpectralSourceSeparationDecoder
from .rtfs_src.audio_encoder import RTFS_AudioEncoder
from .rtfs_src.separation_network import SeparationNetwork


class RTFSModel(torch.nn.Module):
    def __init__(
        self,
        # audio_encoder: torch.nn.Module,
        # s3_decoder_block: torch.nn.Module,
        # separation_network: torch.nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super(RTFSModel, self).__init__(*args, **kwargs)

        self.audio_encoder = RTFS_AudioEncoder()

        self.separation_network = SeparationNetwork(
            audio_channels=256, video_channels=512
        )
        self.s3_decoder_block = SpectralSourceSeparationDecoder(
            input_channels=256,
        )

    def forward(
        self, raw_audio: torch.Tensor, video_features: torch.Tensor = None  # B, N, T, F
    ) -> torch.Tensor:
        print(f"raw audio shape {raw_audio.shape}")

        audio_features = self.audio_encoder(
            raw_audio=raw_audio
        )  # B, 1, L -> B, N, T, (F)

        print(f"audio features shape {audio_features.shape}")

        # CAF BLOCK + R-stacked RTFS blocks

        video_features = video_features.transpose(-1, -2)

        print(f"video features shape {video_features.shape}")

        separated_features = self.separation_network(
            audio_features=audio_features, video_features=video_features
        )

        return self.s3_decoder_block(
            processed_audio=separated_features, original_audio=audio_features
        )


def test_shape():
    batch_size = 16
    audio_size = 32000

    video_time = 50
    video_features_size = 512

    audio = torch.rand(batch_size, audio_size)
    video_features = torch.rand(batch_size, video_time, video_features_size)

    model = RTFSModel()

    out = model(audio, video_features)

    print(out.shape)


test_shape()
