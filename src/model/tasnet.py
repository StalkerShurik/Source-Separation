import torch.nn as nn

# segmentation в transforms нужно закинуть


class Encoder(nn.Module):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv1d(input_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_conv = self.conv(x)

        return self.sigmoid(x_conv) * self.relu(x_conv)


class Separator(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        biderectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            biderectional=biderectional,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # TODO many rnn blocks with skip-connections and activations.
        self.output_size = 228  # todo
        self.head = nn.Linear(
            self.output_size,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_processed = self.rnn(x)

        return self.sigmoid(self.head(x_processed))


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TasNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoder -> Separator -> Decoder
