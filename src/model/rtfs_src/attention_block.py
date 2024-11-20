import numpy as np
import torch
import torch.nn as nn

from .conv_blocks import AttentionConvBlockWithNormalization, DropPath, FeedForwardBlock


class Attention2D(nn.Module):
    """
    https://arxiv.org/pdf/2209.03952
    fig 2
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        features_dim: int,
        num_heads: int = 4,
        *args,
        **kwargs,  # E
    ):
        super(Attention2D, self).__init__()

        assert in_channels % num_heads == 0

        self.q_heads: nn.ModuleList = nn.ModuleList(
            AttentionConvBlockWithNormalization(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                is_conv_2d=True,
                activation_function=nn.PReLU,
                features_dim=features_dim,
            )
            for _ in range(num_heads)
        )
        self.k_heads: nn.ModuleList = nn.ModuleList(
            AttentionConvBlockWithNormalization(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                is_conv_2d=True,
                activation_function=nn.PReLU,
                features_dim=features_dim,
            )
            for _ in range(num_heads)
        )
        self.v_heads: nn.ModuleList = nn.ModuleList(
            AttentionConvBlockWithNormalization(
                in_channels=in_channels,
                out_channels=in_channels // num_heads,
                kernel_size=1,
                is_conv_2d=True,
                activation_function=nn.PReLU,
                features_dim=features_dim,
            )
            for _ in range(num_heads)
        )

        self.ffn: nn.Module = AttentionConvBlockWithNormalization(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            is_conv_2d=True,
            activation_function=nn.PReLU,
            features_dim=features_dim,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input shape: B x C x T x F
        output shape: B x C x T x F
        """
        # TODO: REWRITE
        input_residual = input
        # B x Heads x HidChannels x T x F
        q_values = torch.concat(
            [q_head(input).unsqueeze(0) for q_head in self.q_heads], dim=0
        ).transpose(0, 1)
        k_values = torch.concat(
            [k_head(input).unsqueeze(0) for k_head in self.k_heads], dim=0
        ).transpose(0, 1)
        v_values = torch.concat(
            [v_head(input).unsqueeze(0) for v_head in self.v_heads], dim=0
        ).transpose(0, 1)

        # B x Heads x T x HidChannels x F
        q_values = q_values.transpose(-2, -3)
        k_values = k_values.transpose(-2, -3)
        v_values = v_values.transpose(-2, -3)

        v_shape = v_values.shape

        q_values = q_values.flatten(start_dim=3)  # B x Heads x T x HidChannels * F
        k_values = k_values.flatten(start_dim=3)
        v_values = v_values.flatten(start_dim=3)

        d = q_values.shape[-1]

        # B x Heads x T x T
        attn = torch.matmul(q_values, k_values.transpose(-1, -2)) / (d**0.5)

        attn = torch.nn.functional.softmax(attn, dim=3)  # B x Heads x T x T

        v_values = torch.matmul(attn, v_values)  # B x Heads x T x F'

        # B x Heads x C x T x F
        v_values = v_values.reshape(v_shape).transpose(-2, -3)

        b, h, c, t, f = v_values.shape

        v_values = v_values.reshape(b, h * c, t, f)

        return self.ffn(v_values) + input_residual


class PositionalEncoder(nn.Module):
    # original sinusoid encoding from https://arxiv.org/abs/1706.03762
    def __init__(self, in_channels: int, max_len: int = 10000) -> None:
        super(PositionalEncoder, self).__init__()
        self.in_channels = in_channels
        self.max_len = max_len

        encoded_positions = torch.zeros(self.max_len, self.in_channels)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.in_channels, 2, dtype=torch.float)
            * -(np.log(self.max_len) / self.in_channels)
        )

        encoded_positions[:, 0::2] = torch.sin(position * div_term)
        encoded_positions[:, 1::2] = torch.cos(position * div_term)
        encoded_positions = encoded_positions.unsqueeze(0)
        self.register_buffer("encoded_positions", encoded_positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.encoded_positions[:, : x.size(1)]
        return x


class Attention1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super(Attention1D, self).__init__()

        assert in_channels % num_heads == 0

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm_1 = nn.LayerNorm(self.in_channels)
        self.positional_encoder = PositionalEncoder(in_channels=self.in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.in_channels,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.additional_dropout = nn.Dropout(self.dropout)
        self.norm_2 = nn.LayerNorm(self.in_channels)
        self.drop_path = DropPath(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = x.transpose(1, 2)  # B, C, T -> B, T, C

        x = self.positional_encoder(self.norm_1(x))
        residual = x

        x = self.attention(x, x, x)[0]  # self attention

        x = self.norm_2(self.additional_dropout(x) + residual)

        x = x.transpose(2, 1)  # B, T, C -> B, C, T

        return self.drop_path(x) + res


class GlobalAttention1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super(GlobalAttention1d, self).__init__()
        self.in_channels = in_channels
        self.hid_chan = 2 * self.in_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.Sequential(
            Attention1D(
                in_channels=self.in_channels,
                num_heads=self.num_heads,
                dropout=self.dropout,
            ),
            FeedForwardBlock(
                in_channels=self.in_channels,
                hidden_channels=self.hid_chan,
                kernel_size=kernel_size,
                dropout=self.dropout,
                is_conv_2d=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
