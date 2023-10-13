import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampling(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.sequential(x.unsqueeze(1))
        batch_size, encoder_dim, subsampled_time, subsampled_freq = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, subsampled_time, encoder_dim * subsampled_freq)


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvolutionModule(nn.Module):
    def __init__(self, encoder_dim, kernel_size, dropout=0.1, bias=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.sequential = nn.Sequential(
            PointwiseConv(encoder_dim, 2 * encoder_dim, bias=bias),
            nn.GLU(dim=1),
            DepthwiseConv(encoder_dim, encoder_dim, kernel_size=kernel_size, padding="same", bias=bias),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            PointwiseConv(encoder_dim, encoder_dim, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        return self.sequential(x.transpose(1, 2)).transpose(1, 2)
    
if __name__ == "__main__":
    batch = torch.ones((1, 863, 32))
    cm = ConvolutionModule(32, 3)
    print(cm(batch).shape)

    for len in range(862, 868):
        batch = torch.ones((1, len, 32))
        subsampler = ConvSubsampling(1, 32, 3)
        print(len, subsampler(batch).shape)
        break