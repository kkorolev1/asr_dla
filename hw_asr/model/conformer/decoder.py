import torch
import torch.nn as nn


class ConformerDecoder(nn.Module):
    def __init__(self, n_class, encoder_dim, decoder_layers, decoder_dim, decoder_dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=decoder_dim,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=decoder_dropout,
            proj_size=n_class
        )

    def forward(self, x):
        return self.lstm(x)[0]