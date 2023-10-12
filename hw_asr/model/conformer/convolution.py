import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        batch_size, channels, subsampled_lengths, sumsampled_dim = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        return x

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
    def __init__(self, emb_dim, kernel_size, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.pointwise_conv = PointwiseConv(emb_dim, expansion_factor * emb_dim)
        self.depthwise_conv = DepthwiseConv(emb_dim, emb_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.batch_norm = nn.BatchNorm1d(emb_dim)
        self.pointwise_conv2 = PointwiseConv(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pointwise_conv(x.transpose(1, 2))
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
    
if __name__ == "__main__":
    batch = torch.ones((1, 863, 32))
    cm = ConvolutionModule(32, 3)
    print(cm(batch).shape)

    for len in range(862, 868):
        batch = torch.ones((1, len, 32))
        subsampler = ConvolutionSubsampling(1, 32, 3)
        print(len, subsampler(batch).shape)
        break