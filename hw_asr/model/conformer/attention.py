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

        assert encoder_dim % attention_heads == 0
        self.encoder_dim = encoder_dim
        self.d_head = encoder_dim // attention_heads
        self.attention_heads = attention_heads

        self.W_q = nn.Linear(encoder_dim, encoder_dim)
        self.W_k = nn.Linear(encoder_dim, encoder_dim)
        self.W_v = nn.Linear(encoder_dim, encoder_dim)
        self.W_pos = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.W_out = nn.Linear(encoder_dim, encoder_dim)

        self.u = nn.Parameter(torch.Tensor(self.attention_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.attention_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

        self.layer_norm = nn.LayerNorm(encoder_dim, eps=6.1e-5)
        self.positional_encoder = positional_encoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, seq_length, _ = x.size()

        x = self.layer_norm(x)
        pos_emb = self.positional_encoder(seq_length)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        q = self.W_q(x).view(batch_size, seq_length, self.attention_heads, self.d_head)
        k = self.W_k(x).view(batch_size, seq_length, self.attention_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, attention_heads, d_head, time)
        v = self.W_v(x).view(batch_size, seq_length, self.attention_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, attention_heads, d_head, time)
        pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.attention_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, attention_heads, d_head, time)

        AC = torch.matmul((q + self.u).transpose(1, 2), k)
        BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
        BD = self.rel_shift(BD)
        attn = (AC + BD) / math.sqrt(self.encoder_dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
            attn.masked_fill_(mask, mask_value)

        attn = F.softmax(attn, -1)

        output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2)
        output = output.contiguous().view(batch_size, -1, self.encoder_dim)

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

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.attention(query=x, key=x, value=x, need_weights=False)[0]
        x = self.dropout(x)
        return x
