import torch
import torch.nn as nn
from pytorch3Dlearning.networks.layers.graph_layers import EdgeGraphConvBlock


class DGCNN(nn.Module):
    def __init__(self, k=32, emb_dims=1024, in_channels=3, output_channels=12):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.in_channels = in_channels
        self.output_channels = output_channels

        self.conv1 = EdgeGraphConvBlock(self.in_channels, 64, 64, "local_global", self.k)
        self.conv2 = EdgeGraphConvBlock(64, 64, 128, "local_global", self.k)
        self.conv3 = EdgeGraphConvBlock(128, 128, self.emb_dims, "local_global", self.k)

        self.final = nn.Sequential(
            nn.Conv1d(self.emb_dims, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, self.output_channels, 1),
        )

    def forward(self, x, pos):
        x, idx1 = self.conv1(x, pos=pos)
        x, idx2 = self.conv2(x, pos=pos)
        x, idx3 = self.conv3(x, pos=pos)
        x = x.transpose(2, 1).contiguous()
        x = self.final(x)
        x = x.transpose(2, 1).contiguous()
        return x

if __name__ == "__main__":
    model = DGCNN(in_channels=6, output_channels=3)
    print(model)
    x = torch.randn(10, 100, 6).float()
    pos = torch.randn(10, 100, 3).float()
    out = model(x, pos)

