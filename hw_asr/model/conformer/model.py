import torch
import torch.nn as nn

from hw_asr.base import BaseModel
from hw_asr.model.conformer.convolution import ConvolutionSubsampling, ConvolutionModule
from hw_asr.model.conformer.feed_forward import FeedForwardModule
from hw_asr.model.conformer.attention import MultiHeadAttentionModule


class ConformerBlock(nn.Module):
    def __init__(self, emb_dim, attention_heads, conv_kernel_size=31, feed_forward_dropout=0.1, attention_dropout=0.1, conv_dropout=0.1):
        assert emb_dim % 4 == 0
        super().__init__()
        self.feed_forward = FeedForwardModule(emb_dim, dropout=feed_forward_dropout)
        self.attention = MultiHeadAttentionModule(emb_dim, attention_heads=attention_heads,
                                                  dropout=attention_dropout)
        self.conv = ConvolutionModule(emb_dim, kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.feed_forward2 = FeedForwardModule(emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + 0.5 * self.feed_forward(x)
        x = x + self.attention(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward2(x)
        x = self.layer_norm(x)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, spec_dim, encoder_layers=16, encoder_dim=144, attention_heads=8, encoder_dropout=0.1,
                 conv_kernel_size=31, feed_forward_dropout=0.1, attention_dropout=0.1, conv_dropout=0.1):
        super().__init__()
        self.conv_subsampling = ConvolutionSubsampling(1, encoder_dim, kernel_size=3)
        self.linear = nn.Linear(encoder_dim * (((spec_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.dropout = nn.Dropout(encoder_dropout)
        self.blocks = nn.ModuleList([ConformerBlock(emb_dim=encoder_dim, attention_heads=attention_heads, 
                                                    conv_kernel_size=conv_kernel_size, feed_forward_dropout=feed_forward_dropout,
                                                    attention_dropout=attention_dropout, conv_dropout=conv_dropout) 
                                                    for _ in range(encoder_layers)])

        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.conv_subsampling(x)
        x = self.linear(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return x

class ConformerDecoder(nn.Module):
    def __init__(self, emb_dim, n_class, decoder_layers=1, decoder_dim=320):
        super().__init__()
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=decoder_dim, num_layers=decoder_layers, batch_first=True)
        self.linear = nn.Linear(decoder_dim, n_class)

        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.lstm(x)[0]
        x = self.linear(x)
        return x

class Conformer(BaseModel):
    def __init__(self, n_feats, n_class, encoder_layers=16, encoder_dim=144, encoder_dropout=0.1, decoder_layers=1,
                 decoder_dim=320, attention_heads=4, conv_kernel_size=31,
                 feed_forward_dropout=0.1, attention_dropout=0.1, conv_dropout=0.1):
        super().__init__(n_feats, n_class)
        self.encoder = ConformerEncoder(spec_dim=n_feats, encoder_layers=encoder_layers, encoder_dim=encoder_dim, 
                                        encoder_dropout=encoder_dropout, attention_heads=attention_heads,
                                        conv_kernel_size=conv_kernel_size, feed_forward_dropout=feed_forward_dropout,
                                        attention_dropout=attention_dropout, conv_dropout=conv_dropout)
        self.decoder = ConformerDecoder(emb_dim=encoder_dim, decoder_layers=decoder_layers,
                                        decoder_dim=decoder_dim, n_class=n_class)

    def forward(self, spectrogram, **batch):
        return self.decoder(self.encoder(spectrogram.transpose(1, 2)))

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - 3) // 4

if __name__ == "__main__":
    batch = torch.ones((1, 20, 80))
    cb = ConformerBlock(80, 8)
    print(cb(batch).shape)

    encoder = ConformerEncoder(80, 4, 144, 8)
    print(encoder(batch).shape)

    decoder = ConformerDecoder(144, 3, 64, 10)
    print(decoder(encoder(batch)).shape)
