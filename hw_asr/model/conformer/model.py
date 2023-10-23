from hw_asr.base import BaseModel
from hw_asr.model.conformer.encoder import ConformerEncoder
from hw_asr.model.conformer.decoder import ConformerDecoder


class Conformer(BaseModel):
    def __init__(self, n_feats, n_class,
                 encoder_layers, encoder_dim, encoder_dropout,
                 decoder_layers, decoder_dim, decoder_dropout,
                 attention_heads, attention_dropout,
                 conv_kernel_size, conv_dropout,
                 feed_forward_expansion, feed_forward_dropout):
        super().__init__(n_feats, n_class)
        self.encoder = ConformerEncoder(n_feats=n_feats, 
                                        
                                        encoder_layers=encoder_layers,
                                        encoder_dim=encoder_dim, 
                                        encoder_dropout=encoder_dropout,

                                        attention_heads=attention_heads, 
                                        attention_dropout=attention_dropout,

                                        conv_kernel_size=conv_kernel_size, 
                                        conv_dropout=conv_dropout,

                                        feed_forward_expansion=feed_forward_expansion, 
                                        feed_forward_dropout=feed_forward_dropout)
        self.decoder = ConformerDecoder(n_class=n_class, 
                                        encoder_dim=encoder_dim, 
                                        decoder_layers=decoder_layers,
                                        decoder_dim=decoder_dim, 
                                        decoder_dropout=decoder_dropout)


    def forward(self, spectrogram, spectrogram_length, **batch):
        return self.decoder(self.encoder(spectrogram.transpose(1, 2), spectrogram_length))


    def transform_input_lengths(self, input_lengths):
        return ((input_lengths - 1) // 2 - 1) // 2
