import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampling(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.sequential(x.unsqueeze(1))
        batch_size, encoder_dim, subsampled_time, subsampled_freq = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, subsampled_time, encoder_dim * subsampled_freq)


class ConvolutionModule(nn.Module):
    def __init__(self, encoder_dim, kernel_size, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(encoder_dim, 2 * encoder_dim, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=kernel_size, padding="same", groups=encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        return self.sequential(x.transpose(1, 2)).transpose(1, 2)
