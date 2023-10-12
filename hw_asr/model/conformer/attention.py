import torch
import torch.nn as nn


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, encoder_dim, attention_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.attention = nn.MultiheadAttention(encoder_dim, attention_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        x = self.layer_norm(x)
        x = self.attention(query=x, key=x, value=x, 
                           key_padding_mask=padding_mask, need_weights=False)[0]
        x = self.dropout(x)
        return x
    
if __name__ == "__main__":
    def lengths_to_padding_mask(lengths):
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
            batch_size, max_length
        ) >= lengths.unsqueeze(1)
        return padding_mask
    batch = torch.ones((32, 31, 512))
    mham = MultiHeadAttentionModule(512, 8)
    padding_mask = lengths_to_padding_mask(torch.arange(32))
    print(padding_mask.shape)
    print(mham(batch, padding_mask).shape)
