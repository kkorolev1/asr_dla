import torch
import torch.nn as nn


class ConformerDecoder(nn.Module):
    def __init__(self, encoder_dim, n_class, decoder_layers=1, decoder_dim=320):
        super().__init__()
        self.lstm = nn.LSTM(input_size=encoder_dim, hidden_size=decoder_dim, num_layers=decoder_layers, batch_first=True)
        self.linear = nn.Linear(decoder_dim, n_class)

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.linear(x)
        return x