'''
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
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoder(nn.Module):
  def __init__(self, encoder_dim, max_len=10000):
    super().__init__()
    self.encoder_dim = encoder_dim
    encodings = torch.zeros(max_len, encoder_dim)
    pos = torch.arange(0, max_len, dtype=torch.float)
    inv_freq = 1 / (10000 ** (torch.arange(0.0, encoder_dim, 2.0) / encoder_dim))
    encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
    encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
    self.register_buffer('encodings', encodings)
    
  def forward(self, len):
    return self.encodings[:len, :]


class RelativeMultiHeadAttentionModule(nn.Module):
    def __init__(self, encoder_dim, attention_heads, dropout, positional_encoder):
        super().__init__()

        #dimensions
        assert encoder_dim % attention_heads == 0
        self.encoder_dim = encoder_dim
        self.d_head = encoder_dim // attention_heads
        self.attention_heads = attention_heads

        # Linear projection weights
        self.W_q = nn.Linear(encoder_dim, encoder_dim)
        self.W_k = nn.Linear(encoder_dim, encoder_dim)
        self.W_v = nn.Linear(encoder_dim, encoder_dim)
        self.W_pos = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.W_out = nn.Linear(encoder_dim, encoder_dim)

        # Trainable bias parameters
        self.u = nn.Parameter(torch.Tensor(self.attention_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.attention_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        # etc
        self.layer_norm = nn.LayerNorm(encoder_dim, eps=6.1e-5)
        self.positional_encoder = positional_encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        #layer norm and pos embeddings
        x = self.layer_norm(x)
        pos_emb = self.positional_encoder(seq_length)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        #Linear projections, split into heads
        q = self.W_q(x).view(batch_size, seq_length, self.attention_heads, self.d_head)
        k = self.W_k(x).view(batch_size, seq_length, self.attention_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, attention_heads, d_head, time)
        v = self.W_v(x).view(batch_size, seq_length, self.attention_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, attention_heads, d_head, time)
        pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.attention_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, attention_heads, d_head, time)

        #Compute attention scores with relative position embeddings
        AC = torch.matmul((q + self.u).transpose(1, 2), k)
        BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        BD = self.rel_shift(BD)
        attn = (AC + BD) / math.sqrt(self.encoder_dim)

        #Mask before softmax with large negative number
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
            attn.masked_fill_(mask, mask_value)

        #Softmax
        attn = F.softmax(attn, -1)

        #Construct outputs from values
        output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2) # (batch_size, time, attention_heads, d_head)
        output = output.contiguous().view(batch_size, -1, self.encoder_dim) # (batch_size, time, encoder_dim)

        #Output projections and dropout
        output = self.W_out(output)
        return self.dropout(output)


    def rel_shift(self, emb):
        batch_size, attention_heads, seq_length1, seq_length2 = emb.size()
        zeros = emb.new_zeros(batch_size, attention_heads, seq_length1, 1)
        padded_emb = torch.cat([zeros, emb], dim=-1)
        padded_emb = padded_emb.view(batch_size, attention_heads, seq_length2 + 1, seq_length1)
        shifted_emb = padded_emb[:, :, 1:].view_as(emb)
        return shifted_emb


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