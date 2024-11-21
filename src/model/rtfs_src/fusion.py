import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import ConvBlockWithActivation


# CHECKED!
class CAF(nn.Module):
    def __init__(self, audio_channels: int, video_channels: int, num_heads: int):
        super(CAF, self).__init__()

        self.num_heads = num_heads

        self.audio_conv_p1 = ConvBlockWithActivation(
            in_channels=audio_channels,
            out_channels=audio_channels,
            kernel_size=1,
            is_conv_2d=True,
            groups=audio_channels,
            activation_function=torch.nn.Identity,
        )

        self.audio_conv_p2 = ConvBlockWithActivation(
            in_channels=audio_channels,
            out_channels=audio_channels,
            kernel_size=1,
            is_conv_2d=True,
            groups=audio_channels,
            activation_function=nn.ReLU,
        )

        self.video_conv_f1 = ConvBlockWithActivation(
            in_channels=video_channels,
            out_channels=audio_channels * num_heads,
            kernel_size=1,
            is_conv_2d=False,
            groups=audio_channels,
            activation_function=torch.nn.Identity,
        )
        self.video_conv_f2 = ConvBlockWithActivation(
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

        batch_dim, audio_channels_dim, time_dim, features_dim = audio_features.shape

        audio_value = self.audio_conv_p1(audio_features)  # B x Ca x Ta x F
        audio_gate = self.audio_conv_p2(audio_features)  # B x Ca x Ta x F

        video_attn = self.video_conv_f1(video_features)  # B x (Ca x h) x Tv

        # B x Ca x Tv
        video_attn = video_attn.reshape(
            batch_dim, audio_channels_dim, self.num_heads, -1
        ).mean(dim=2)

        video_attn = torch.softmax(video_attn, -1)
        video_attn = F.interpolate(video_attn, size=time_dim)

        video_key = self.video_conv_f2(video_features)
        video_key = F.interpolate(video_key, size=time_dim)

        return (
            video_key.unsqueeze(-1) * audio_gate
            + video_attn.unsqueeze(-1) * audio_value
        )
