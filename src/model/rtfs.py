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
        self, mix: torch.Tensor, video_embed: torch.Tensor = None, **kwargs  # B, N, T, F
    ) -> torch.Tensor:

        mix = torch.concat([mix, mix], dim=0)

        audio_features = self.audio_encoder(
            raw_audio=mix
        )  # B, 1, L -> B, N, T, (F)

        # CAF BLOCK + R-stacked RTFS blocks

        video_features = video_embed.transpose(-1, -2)

        separated_features = self.separation_network(
            audio_features=audio_features, video_features=video_features
        )

        decoded = self.s3_decoder_block(
            processed_audio=separated_features, original_audio=audio_features
        )

        decoded_dict = {}

        batch_size, seq_len = decoded.shape

        if batch_size % 2 != 0:
            raise Exception("batch size is not even")

        decoded_dict["predict"] = torch.empty((batch_size // 2, 2, seq_len)).to(decoded.device)

        decoded_dict["predict"][:,0,:] = decoded[:batch_size // 2]
        decoded_dict["predict"][:,1,:] = decoded[batch_size // 2:]

        return decoded_dict
