import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    def __init__(self, encoder_dim, feed_forward_expansion=2, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, feed_forward_expansion * encoder_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_expansion * encoder_dim, encoder_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.sequential(x)
    

if __name__ == "__main__":
    batch = torch.ones((32, 20, 32))
    ffm = FeedForwardModule(32)
    print(ffm(batch).shape)
