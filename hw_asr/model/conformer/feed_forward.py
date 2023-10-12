import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    def __init__(self, emb_dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, expansion_factor * emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion_factor * emb_dim, emb_dim)
        self.dropout2 = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
    

if __name__ == "__main__":
    batch = torch.ones((32, 20, 32))
    ffm = FeedForwardModule(32)
    print(ffm(batch).shape)
