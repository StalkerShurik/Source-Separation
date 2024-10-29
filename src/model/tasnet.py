import torch
import torch.nn as nn

# segmentation в transforms нужно закинуть
# пока внутри модели делаем


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = output_dim
        self.U = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1, bias=False)
        self.V = nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        input: tensor B x K x L
        output: tensot B x K x N
        """

        # normilize x here or in transforms?
        # here, because we will denorm back

        B, K, L = x.shape
        normalization = x.norm(p=2, dim=-1, keepdim=True)
        x = (x / (normalization + 1e-9)).view(B * K, L).unsqueeze(-1)

        return normalization, (self.sigmoid(self.U(x)) * self.relu(self.V(x))).view(
            B, K, self.N
        )


class Separator(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=500,
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=4,
        dropout=0,
        bias=True,
        n_sources=2,
        rnn_layers_activation="Identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        rnn_output_size = hidden_size if not bidirectional else hidden_size * 2
        self.num_layers = num_layers
        self.rnn_layers = []
        for layer_index in range(num_layers):
            self.rnn_layers.append(
                getattr(nn, rnn_type)(
                    input_size if layer_index == 0 else rnn_output_size,
                    hidden_size,
                    bidirectional=bidirectional,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout,
                    bias=bias,
                )
            )
        self.rnn_layers = nn.ParameterList(self.rnn_layers)
        self.rnn_layers_activation = getattr(nn, rnn_layers_activation)()

        for name, param in self.rnn_layers.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal(param)
        # TODO many rnn blocks with skip-connections and activations.

        self.n_sources = n_sources
        self.head = nn.Linear(rnn_output_size, n_sources * input_size)

    def forward(self, x):
        """
        input: tensor B x K x N
        output: tensor B x K x 2N
        """

        B, K, N = x.shape
        x_processed = x
        skip_connection_data = None

        for layer_index in range(self.num_layers):
            skip_connection_new_data = x_processed
            x_processed = self.rnn_layers[layer_index](x_processed)[0]
            if layer_index - 3 >= 1:
                x_processed += skip_connection_data
            skip_connection_data = skip_connection_new_data

        concatted_output = self.head(x_processed)

        return nn.functional.softmax(
            concatted_output.view(B, K, self.n_sources, N), dim=-1
        )


class Decoder(nn.Module):
    """
    multiply encoded input on masks from Separator to achieve 2 samples from 1
    in some way recover segments
    concatenate segments
    """

    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basis_decoder = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, normalization, encoded, rnn_mask):
        encoded_masked = encoded.unsqueeze(2) * normalization.unsqueeze(2) * rnn_mask
        basis_decoded = self.basis_decoder(encoded_masked)
        return basis_decoded


class TasNet(nn.Module):
    def __init__(
        self,
        L: int = 40,
        N: int = 500,
        n_sources: int = 2,
        rnn_hidden: int = 500,
        rnn_bidirectional: bool = True,
        rnn_type: str = "LSTM",
        rnn_layers: int = 4,
        rnn_dropout: float = 0.0,
        rnn_bias: bool = True,
        rnn_layers_activation: str = "Identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.n_sources = n_sources
        self.L = L
        self.encoder = Encoder(L, N)
        self.separator = Separator(
            input_size=N,
            hidden_size=rnn_hidden,
            bidirectional=rnn_bidirectional,
            rnn_type=rnn_type,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            bias=rnn_bias,
            n_sources=n_sources,
            rnn_layers_activation=rnn_layers_activation,
        )
        self.decoder = Decoder(N, L)

    def forward(self, mix: torch.Tensor, **batch):
        normalization, weights = self.encoder(
            mix.view(mix.shape[0], mix.shape[1] // self.L, self.L)
        )
        rnn_masks = self.separator(weights)
        processed_sources = self.decoder(normalization, weights, rnn_masks).permute(
            (0, 2, 1, 3)
        )  # decoder returns tensor of shape B x K x 2 x L
        return {
            "predict": processed_sources.reshape((mix.shape[0], self.n_sources, -1))
        }
