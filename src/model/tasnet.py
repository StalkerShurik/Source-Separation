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
        hidden_size=128,
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        n_sources=2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # TODO many rnn blocks with skip-connections and activations.

        self.n_sources = n_sources
        self.head_input_size = hidden_size if not bidirectional else hidden_size * 2
        self.head = nn.Linear(self.head_input_size, n_sources * input_size)

    def forward(self, x):
        """
        input: tensor B x K x N
        output: tensor B x K x 2N
        """

        B, K, N = x.shape
        x_processed = self.rnn(x)[0]
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
        L=50,
        N=500,
        n_sources=2,
        rnn_hidden=512,
        rnn_bidirectional=True,
        rnn_layers=4,
        rnn_dropout=0,
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
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            n_sources=n_sources,
        )
        self.decoder = Decoder(N, L)

    def forward(self, mix, **batch):
        normalization, weights = self.encoder(
            mix.view(mix.shape[0], mix.shape[1] // self.L, self.L)
        )
        rnn_masks = self.separator(weights)
        processed_sources = self.decoder(normalization, weights, rnn_masks)
        output = {}
        for i in range(self.n_sources):
            output[f"predicted_source_{i + 1}"] = (
                processed_sources[:, :, i, :].contiguous().view(mix.shape)
            )
        return output
