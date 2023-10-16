import torch
import torch.nn as nn


class ConformerDecoder(nn.Module):
    def __init__(self, encoder_dim, n_class, decoder_layers=1, decoder_dropout=0.1, decoder_dim=320):
        super().__init__()
        self.lstm = nn.LSTM(input_size=encoder_dim, hidden_size=decoder_dim, num_layers=decoder_layers, batch_first=True, dropout=decoder_dropout, proj_size=n_class)

    def forward(self, x):
        return self.lstm(x)[0]