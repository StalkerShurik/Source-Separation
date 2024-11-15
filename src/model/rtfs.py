import torch

from .rtfs_src import audio_encoder as audio_encoder_module
from .rtfs_src import separation_network as separation_network_module


class RTFSModel(torch.nn.Module):
    def __init__(
        self,
        n_src: int,
        audio_encoder: audio_encoder_module.RTFS_AudioEncoder,
        separation_network: separation_network_module.SeparationNetwork,
        s3_block,  # TODO
        audio_decoder,  # TODO
        *args,
        **kwargs
    ) -> None:
        super(RTFSModel, self).__init__(*args, **kwargs)

        self.n_src = n_src
        self.audio_encoder = audio_encoder

        self.separation_network = separation_network
        self.s3_block = s3_block
        self.audio_decoder = audio_decoder

        self.s3_block = s3_block

        self.init_modules()  # TODO

    def forward(
        self, raw_audio: torch.Tensor, video_features: torch.Tensor = None  # B, N, T, F
    ) -> torch.Tensor:
        audio_features = self.audio_encoder(
            raw_audio=raw_audio
        )  # B, 1, L -> B, N, T, (F)

        # CAF BLOCK + R-stacked RTFS blocks

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
