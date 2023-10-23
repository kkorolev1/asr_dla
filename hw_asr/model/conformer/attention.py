import torch
import torch.nn as nn


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, encoder_dim, attention_heads, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.attention = nn.MultiheadAttention(embed_dim=encoder_dim,
                                               num_heads=attention_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        x = self.layer_norm(x)
        x = self.attention(query=x, key=x, value=x, 
                           key_padding_mask=padding_mask, need_weights=False)[0]
        x = self.dropout(x)
        return x

