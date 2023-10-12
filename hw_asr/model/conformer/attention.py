import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, emb_dim):
        super().__init__()
        self.features = torch.zeros(max_length, emb_dim)

        pos = torch.arange(0, max_length, dtype=float)
        coord = torch.exp(torch.arange(0, emb_dim, 2, dtype=float) * (-math.log(10000) / emb_dim))
        arg = pos[:, None] * coord[None, :]

        self.features[:, 0::2] = torch.sin(arg)
        self.features[:, 1::2] = torch.cos(arg)
        self.features = nn.Parameter(self.features, requires_grad=False)
        

    def forward(self, lengths):
        return self.features[:lengths, :]


class Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()

        assert emb_dim % num_heads == 0
        attention_dim = emb_dim // num_heads

        self.query_proj = nn.Linear(emb_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, attention_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.query_proj.weight)
        torch.nn.init.xavier_uniform_(self.key_proj.weight)
        torch.nn.init.xavier_uniform_(self.value_proj.weight)
    
    def forward(self, query, key, value, mask=None):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        dot_products = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        
        if mask is not None:
            dot_products = dot_products.where(mask, -1e9)

        attention_scores = torch.softmax(dot_products, dim=-1)
        attention = torch.bmm(self.dropout(attention_scores), value)

        return attention, attention_scores

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0

        self.heads = nn.ModuleList([Attention(emb_dim, num_heads, dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(emb_dim, emb_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, query, key, value, mask=None):
        attentions, attentions_scores = [], []

        for head in self.heads:
            attention, attention_scores = head(query, key, value, mask)
            attentions.append(attention)
            attentions_scores.append(attention_scores)
        
        attentions = torch.cat(attentions, dim=-1)
        attentions_scores = torch.stack(attentions_scores, dim=-1)

        attentions = self.linear(attentions)
        return attentions, attentions_scores
    
class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))

        self.out_proj = nn.Linear(d_model, d_model)

        torch.nn.init.xavier_uniform_(self.query_proj.weight)
        torch.nn.init.zeros_(self.query_proj.bias)

        torch.nn.init.xavier_uniform_(self.key_proj.weight)
        torch.nn.init.zeros_(self.key_proj.bias)

        torch.nn.init.xavier_uniform_(self.value_proj.weight)
        torch.nn.init.zeros_(self.value_proj.bias)

        torch.nn.init.xavier_uniform_(self.pos_proj.weight)

        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)

        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

    def forward(self, query, key, value, pos_embedding, mask=None):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, emb_dim, attention_heads, pos_encoding_max_length=1500, dropout=0.1):
        super().__init__()
        assert emb_dim % attention_heads == 0

        self.pos_encoding = PositionalEncoding(pos_encoding_max_length, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        
        self.mha = MultiHeadAttention(emb_dim, attention_heads, dropout)
        #self.mha = RelativeMultiHeadAttention(emb_dim, attention_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #pos_enc = self.pos_encoding(x.shape[1]).repeat(x.shape[0], 1, 1)
        x = self.layer_norm(x)
        x = self.mha(x, x, x)[0]
        #x = self.mha(x, x, x, pos_enc)
        x = self.dropout(x)
        return x
    
if __name__ == "__main__":
    batch = torch.ones((32, 100, 512))
    mham = MultiHeadAttentionModule(512, 8)
    print(mham(batch).shape)
