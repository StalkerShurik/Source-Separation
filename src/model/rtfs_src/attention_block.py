import torch
import torch.nn as nn

from .conv_blocks import ConvBlockWithActivation


class Attn(nn.Module):
    """
    https://arxiv.org/pdf/2209.03952
    fig 2
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_heads: int = 4,
        *args,
        **kwargs,  # E
    ):
        super().__init__(*args, **kwargs)

        assert in_channels % n_heads == 0

        self.q_heads: list[torch.Module] = [
            ConvBlockWithActivation(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                is_conv_2d=True,
            )
            for i in range(n_heads)
        ]
        self.k_heads: list[torch.Module] = [
            ConvBlockWithActivation(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                is_conv_2d=True,
            )
            for i in range(n_heads)
        ]
        self.v_heads: list[torch.Module] = [
            ConvBlockWithActivation(
                in_channels=in_channels,
                out_channels=in_channels // n_heads,
                kernel_size=1,
                is_conv_2d=True,
            )
            for i in range(n_heads)
        ]

        self.FFN: torch.Module = ConvBlockWithActivation(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            is_conv_2d=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input shape: B x C x T x F
        output shape: B x C x T x F
        """

        input_residual = input

        Q = [q_head(input).unsqueeze(0) for q_head in self.q_heads]
        K = [k_head(input).unsqueeze(0) for k_head in self.k_heads]
        V = [v_head(input).unsqueeze(0) for v_head in self.v_heads]

        Q = torch.concat(Q, dim=0).transpose(0, 1)  # B x Heads x HidChannels x T x F
        K = torch.concat(K, dim=0).transpose(0, 1)
        V = torch.concat(V, dim=0).transpose(0, 1)

        Q = Q.transpose(-2, -3)  # B x Heads x T x HidChannels x F
        K = K.transpose(-2, -3)
        V = V.transpose(-2, -3)

        V_shape = V.shape

        Q = Q.flatten(start_dim=3)  # B x Heads x T x HidChannels * F
        K = K.flatten(start_dim=3)
        V = V.flatten(start_dim=3)

        d = Q.shape[-1]

        attn = torch.matmul(Q, K.transpose(-1, -2)) / (d**0.5)  # B x Heads x T x T

        attn = torch.nn.functional.softmax(attn, dim=3)  # B x Heads x T x T

        V = torch.matmul(attn, V)  # B x Heads x T x F'

        V = V.reshape(V_shape).transpose(-2, -3)  # B x Heads x C x T x F

        B, H, C, T, F = V.shape

        V = V.reshape(B, H * C, T, F)

        V = self.FFN(V)

        V = V + input_residual

        return V
