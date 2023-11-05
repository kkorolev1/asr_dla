import torch
import torch.nn as nn

from hw_asr.model.conformer.convolution import ConvSubsampling, ConvolutionModule
from hw_asr.model.conformer.feed_forward import FeedForwardModule
from hw_asr.model.conformer.attention import PositionalEncoder, MultiHeadAttentionModule, RelativeMultiHeadAttentionModule


class Residual(nn.Module):
    def __init__(self, module, input_factor=1.0, module_factor=1.0):
        super().__init__()
        self.module = module
        self.input_factor = input_factor
        self.module_factor = module_factor

    def forward(self, x, *args):
        return x * self.input_factor + self.module(x, *args) * self.module_factor


class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim=144, attention_heads=4, conv_kernel_size=31,
                 feed_forward_dropout=0.1, feed_forward_expansion=2,
                 attention_dropout=0.1, conv_dropout=0.1, positional_encoder=None):
        super().__init__()
        
        self.ff1 = Residual(
            FeedForwardModule(
                encoder_dim=encoder_dim, 
                feed_forward_expansion=feed_forward_expansion,
                dropout=feed_forward_dropout),
            module_factor=0.5)
        
        self.mha = Residual(
            RelativeMultiHeadAttentionModule(
                encoder_dim=encoder_dim,
                attention_heads=attention_heads,
                dropout=attention_dropout,
                positional_encoder=positional_encoder)
            )
        self.conv = Residual(
            ConvolutionModule(encoder_dim, kernel_size=conv_kernel_size, dropout=conv_dropout)
        )
        self.ff2 = Residual(
            FeedForwardModule(
                encoder_dim=encoder_dim, 
                feed_forward_expansion=feed_forward_expansion, dropout=feed_forward_dropout),
            module_factor=0.5)
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask):
        x = self.ff1(x)
        x = self.mha(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, n_feats, encoder_layers=16, encoder_dim=144,
                 attention_heads=8, encoder_dropout=0.1,
                 conv_kernel_size=31, feed_forward_dropout=0.1,
                 feed_forward_expansion=2, attention_dropout=0.1,
                 conv_dropout=0.1):
        super().__init__()
        self.conv_subsampling = ConvSubsampling(out_channels=encoder_dim)
        self.linear = nn.Linear(encoder_dim * (((n_feats - 1) // 2 - 1) // 2), encoder_dim)
        self.dropout = nn.Dropout(encoder_dropout)
        positional_encoder = PositionalEncoder(encoder_dim)
        self.blocks = nn.ModuleList([ConformerBlock(encoder_dim=encoder_dim, 
                                                    attention_heads=attention_heads, 
                                                    conv_kernel_size=conv_kernel_size,
                                                    feed_forward_dropout=feed_forward_dropout,
                                                    feed_forward_expansion=feed_forward_expansion,
                                                    attention_dropout=attention_dropout,
                                                    conv_dropout=conv_dropout,
                                                    positional_encoder=positional_encoder)
                                                    for _ in range(encoder_layers)])

    def forward(self, x, lengths):
        mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]), dtype=bool, device=lengths.device)
        for i, l in enumerate(lengths):
            mask[i, :, :l] = 0

        x = self.conv_subsampling(x)

        mask = mask[:, :-2:2, :-2:2] 
        mask = mask[:, :-2:2, :-2:2]

        x = self.linear(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return x