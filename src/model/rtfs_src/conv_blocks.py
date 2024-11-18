import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape) <= keep_prob).to(x.dtype).to(x.device)
        output = x * mask / keep_prob
        return output


class ConvBlockWithActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        is_conv_2d: bool,
        bias: bool = False,
        groups: int = 1,
        padding: int | str | None = None,
        dilation: int = 1,
        stride: int = 1,
        activation_function: type = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv_class = nn.Conv2d if is_conv_2d else nn.Conv1d
        self.norm_class = nn.BatchNorm2d if is_conv_2d else nn.BatchNorm1d
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2 if stride > 1 else "same"
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.groups = groups

        self.layers = nn.Sequential(
            self.conv_class(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                groups=groups,
                padding=padding,
                stride=stride,
            ),
            self.norm_class(out_channels),
            activation_function(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dropout: float = 0,
        is_conv_2d: bool = False,
    ) -> None:
        super(FeedForwardBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.is_conv_2d = is_conv_2d

        self.encoder = ConvBlockWithActivation(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
            is_conv_2d=self.is_conv_2d,
        )
        self.refiner = ConvBlockWithActivation(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            is_conv_2d=self.is_conv_2d,
        )
        self.decoder = ConvBlockWithActivation(
            in_channels=self.hidden_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            is_conv_2d=self.is_conv_2d,
        )  # FC 2
        self.dropout_layer = DropPath(self.dropout)

        self.layers = nn.Sequential(
            self.encoder,
            self.refiner,
            self.dropout_layer,
            self.decoder,
            self.dropout_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saved_x = x
        return self.layers(x) + saved_x
