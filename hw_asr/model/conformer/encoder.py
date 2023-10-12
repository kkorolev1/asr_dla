import torch
import torch.nn as nn

from hw_asr.model.conformer.convolution import ConvolutionSubsampling, ConvolutionModule
from hw_asr.model.conformer.feed_forward import FeedForwardModule
from hw_asr.model.conformer.attention import MultiHeadAttentionModule


def lengths_to_padding_mask(lengths):
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class Residual(nn.Module):
    def __init__(self, module, input_factor=1.0, module_factor=1.0):
        super().__init__()
        self.module = module
        self.input_factor = input_factor
        self.module_factor = module_factor

    def forward(self, x, *args):
        return x * self.input_factor + self.module(x, *args) * self.module_factor


class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim, attention_heads=4, conv_kernel_size=31, feed_forward_dropout=0.1, feed_forward_expansion=2, attention_dropout=0.1, conv_dropout=0.1):
        super().__init__()
        self.feed_forward = Residual(FeedForwardModule(encoder_dim=encoder_dim, 
                                                       feed_forward_expansion=feed_forward_expansion,
                                                       dropout=feed_forward_dropout),
                                                       module_factor=0.5)
        
        self.attention = Residual(MultiHeadAttentionModule(encoder_dim=encoder_dim,
                                                           attention_heads=attention_heads,
                                                           dropout=attention_dropout))
        self.sequential = nn.Sequential(
            Residual(ConvolutionModule(encoder_dim, kernel_size=conv_kernel_size, dropout=conv_dropout)),
            Residual(FeedForwardModule(encoder_dim=encoder_dim, 
                                       feed_forward_expansion=feed_forward_expansion, dropout=feed_forward_dropout),
                                       module_factor=0.5),
            nn.LayerNorm(encoder_dim)
        )

    def forward(self, x, padding_mask):
        x = self.feed_forward(x)
        x = self.attention(x, padding_mask)
        return self.sequential(x)


class ConformerEncoder(nn.Module):
    def __init__(self, n_feats, encoder_layers=16, encoder_dim=144, attention_heads=8, encoder_dropout=0.1,
                 conv_kernel_size=31, feed_forward_dropout=0.1, feed_forward_expansion=2, 
                 attention_dropout=0.1, conv_dropout=0.1):
        super().__init__()
        #self.conv_subsampling = ConvolutionSubsampling(1, encoder_dim, kernel_size=3)
        #self.linear = nn.Linear(encoder_dim * (((spec_dim - 1) // 2 - 1) // 2), encoder_dim)
        self.linear = nn.Linear(n_feats, encoder_dim)
        self.dropout = nn.Dropout(encoder_dropout)
        self.blocks = nn.ModuleList([ConformerBlock(encoder_dim=encoder_dim, attention_heads=attention_heads, 
                                                    conv_kernel_size=conv_kernel_size, feed_forward_dropout=feed_forward_dropout,
                                                    feed_forward_expansion=feed_forward_expansion, attention_dropout=attention_dropout,
                                                    conv_dropout=conv_dropout) for _ in range(encoder_layers)])

    def forward(self, x, lengths):
        #x = self.conv_subsampling(x)
        padding_mask = lengths_to_padding_mask(lengths)

        x = self.linear(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, padding_mask)
        return x