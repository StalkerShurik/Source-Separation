import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import ConvBlockWithActivation


class ReconstructionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        is_conv_2d: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(ReconstructionBlock, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.is_conv_2d = is_conv_2d

        self.local_embed_block = ConvBlockWithActivation(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            groups=self.in_channels,
            is_conv_2d=self.is_conv_2d,
        )  # diff: add RELU

        self.global_embed_block = ConvBlockWithActivation(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            groups=self.in_channels,
            is_conv_2d=self.is_conv_2d,
        )  # diff: add RELU

        self.coeff_block = ConvBlockWithActivation(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            groups=self.in_channels,
            is_conv_2d=self.is_conv_2d,
            activation_function=nn.Sigmoid,
        )

    def forward(
        self, local_features: torch.Tensor, global_features: torch.Tensor
    ) -> torch.Tensor:
        is_4d = len(global_features.shape) == 4
        prev_shape = global_features.shape[-2 if is_4d else -1 :]
        next_shape = local_features.shape[-2 if is_4d else -1 :]
        prev_size = np.prod(prev_shape)
        next_size = np.prod(next_shape)

        local_embeds = self.local_embed_block(local_features)

        # get and convert global embeds:
        if next_size > prev_size:
            global_embeds = F.interpolate(
                input=self.global_embed_block(global_features),
                size=next_shape,
            )
            coeff = F.interpolate(
                self.coeff_block(global_features),
                size=next_shape,
            )
        else:
            interpolated_global_features = F.interpolate(
                global_features,
                size=next_size,
            )
            global_embeds = self.global_embed_block(interpolated_global_features)
            coeff = self.coeff_block(interpolated_global_features)

        return local_embeds * coeff + global_embeds