import torch
import torch.nn as nn

from hw_asr.base import BaseModel
from hw_asr.model.conformer.encoder import ConformerEncoder
from hw_asr.model.conformer.decoder import ConformerDecoder


class Conformer(BaseModel):
    def __init__(self, n_feats, n_class, encoder_layers=16, encoder_dim=144,
                 encoder_dropout=0.1, decoder_layers=1,
                 decoder_dim=320, attention_heads=4, conv_kernel_size=31,
                 feed_forward_dropout=0.1, feed_forward_expansion=2, attention_dropout=0.1, conv_dropout=0.1):
        super().__init__(n_feats, n_class)
        self.encoder = ConformerEncoder(n_feats=n_feats, encoder_layers=encoder_layers, encoder_dim=encoder_dim, 
                                        encoder_dropout=encoder_dropout, attention_heads=attention_heads,
                                        conv_kernel_size=conv_kernel_size, feed_forward_dropout=feed_forward_dropout,
                                        feed_forward_expansion=feed_forward_expansion, attention_dropout=attention_dropout,
                                        conv_dropout=conv_dropout)
        self.decoder = ConformerDecoder(encoder_dim=encoder_dim, decoder_layers=decoder_layers,
                                        decoder_dim=decoder_dim, n_class=n_class)


    def forward(self, spectrogram, spectrogram_length, **batch):
        return self.decoder(self.encoder(spectrogram.transpose(1, 2), spectrogram_length))


    def transform_input_lengths(self, input_lengths):
        return input_lengths


if __name__ == "__main__":
    batch = torch.ones((1, 20, 80))
    cb = ConformerBlock(80, 8)
    print(cb(batch).shape)

    encoder = ConformerEncoder(80, 4, 144, 8)
    print(encoder(batch).shape)

    decoder = ConformerDecoder(144, 3, 64, 10)
    print(decoder(encoder(batch)).shape)
