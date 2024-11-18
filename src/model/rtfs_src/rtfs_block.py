import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_block import Attention2D
from .conv_blocks import ConvBlockWithActivation
from .reconstruction_block import ReconstructionBlock
from .rnn_block import DualPathRNN


class RTFSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int = 64,
        kernel_size: int = 5,
        stride: int = 2,
        upsampling_depth: int = 2,
        is_conv_2d: bool = True,
        is2d: bool = True,
        rnn_hidden=32,
    ) -> None:
        super(RTFSBlock, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.upsampling_depth = upsampling_depth
        self.is_conv_2d = is_conv_2d
        self.is2d = is2d

        self.normalization_class = nn.BatchNorm2d if is_conv_2d else nn.BatchNorm1d
        self.pool = F.adaptive_avg_pool2d if self.is2d else F.adaptive_avg_pool1d
        self.conv_class = nn.Conv2d if self.is2d else nn.Conv1d

        self.gateway = ConvBlockWithActivation(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            groups=self.in_channels,
            activation_function=nn.PReLU,
            is_conv_2d=self.is_conv_2d,
        )
        self.projection = ConvBlockWithActivation(
            in_channels=self.in_channels,
            out_channels=self.hid_channels,
            kernel_size=1,
            is_conv_2d=self.is_conv_2d,
        )

        self.downsample_layers = nn.ModuleList(
            [
                ConvBlockWithActivation(
                    in_channels=self.hid_channels,
                    out_channels=self.hid_channels,
                    kernel_size=self.kernel_size,
                    stride=1 if i == 0 else self.stride,
                    groups=self.hid_channels,
                    is_conv_2d=self.is_conv_2d,
                )
                for i in range(self.upsampling_depth)
            ]
        )

        self.layers = nn.Sequential(
            DualPathRNN(
                in_channels=self.hid_channels,
                hidden_channels=rnn_hidden,
                apply_to_time=False,
            ),
            DualPathRNN(
                in_channels=self.hid_channels,
                hidden_channels=rnn_hidden,
                apply_to_time=True,
            ),
            Attention2D(
                in_channels=self.hid_channels, hidden_channels=self.hid_channels
            ),
        )

        self.fusion_layers = nn.ModuleList(
            [
                ReconstructionBlock(
                    in_channels=self.hid_channels,
                    kernel_size=self.kernel_size,
                    is_conv_2d=self.is_conv_2d,
                )
                for _ in range(self.upsampling_depth)
            ]
        )
        self.concat_layers = nn.ModuleList(
            [
                ReconstructionBlock(
                    in_channels=self.hid_channels,
                    kernel_size=self.kernel_size,
                    is_conv_2d=self.is_conv_2d,
                )
                for _ in range(self.upsampling_depth - 1)
            ]
        )
        self.residual_conv = nn.Sequential(
            self.conv_class(
                in_channels=self.hid_channels,
                out_channels=self.hid_channels,
                kernel_size=1,
            ),
            self.normalization_class(self.in_channels),  # diff
            nn.ReLU(),  # diff
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, C, T, (F)
        print(f"RTFS INPUT shape {x.shape}")
        residual = self.gateway(x)
        projected_x = self.projection(residual)
        print(f"projected shape {projected_x.shape}")
        # bottom-up
        local_features = [self.downsample_layers[0](projected_x)]
        for i in range(1, self.upsampling_depth):
            local_features.append(self.downsample_layers[i](local_features[-1]))

        # AG
        downsampled_shape = local_features[-1].shape
        global_features = sum(
            self.pool(
                features,
                output_size=downsampled_shape[-(len(downsampled_shape) // 2) :],
            )
            for features in local_features
        )
        print(f"downsamples shape {global_features.shape}")
        # global attention module
        global_features = self.layers(global_features)  # B, N, T, (F)

        # add info from global attention
        united_features = [
            self.fusion_layers[i](
                local_features=local_features[i], global_features=global_features
            )
            for i in range(self.upsampling_depth)
        ]

        # reconstruction phase
        reconstructed_x = (
            self.concat_layers[-1](united_features[-2], united_features[-1])
            + local_features[-2]
        )
        for i in range(self.upsampling_depth - 3, -1, -1):
            reconstructed_x = (
                self.concat_layers[i](
                    local_features=united_features[i], global_features=reconstructed_x
                )
                + local_features[i]
            )

        return self.residual_conv(reconstructed_x) + residual
