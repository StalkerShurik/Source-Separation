import torch
from rtfs_src import separation_network as separation_network_module
from rtfs_src.audio_decoder import SpectralSourceSeparationDecoder
from rtfs_src.audio_encoder import RTFS_AudioEncoder


class RTFSModel(torch.nn.Module):
    def __init__(
        self,
        audio_encoder: torch.nn.Module = RTFS_AudioEncoder,
        separation_network: torch.nn.Module = separation_network_module.SeparationNetwork,
        s3_decoder_block: torch.nn.Module = SpectralSourceSeparationDecoder,
        input_audio_channels=256,
        input_video_channels=512,
        *args,
        **kwargs,
    ) -> None:
        super(RTFSModel, self).__init__(*args, **kwargs)

        self.audio_encoder = audio_encoder()

        self.separation_network = separation_network(
            audio_channels=input_audio_channels, video_channels=input_video_channels
        )  # TODO add delete hardcode
        self.s3_decoder_block = s3_decoder_block

        # self.init_modules()  # TODO

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

        # S^3 block
        final_audio_features = self.s3_block(
            separated_features=separated_features, audio_features=audio_features
        )  # B, n_src, N, T, (F)

        # audio decoder
        return self.audio_decoder(
            final_audio_features, shape=raw_audio.shape
        )  # B, n_src, L


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
