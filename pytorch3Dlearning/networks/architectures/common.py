import torch
import torch.nn as nn

from pytorch3Dlearning.networks.layers.folding_layers import FoldingDecoder
from pytorch3Dlearning.networks.layers.graph_layers import EdgeGraphConvBlock, GraphGroupSelfAttention, BasicPointLayer


class DGCNN(nn.Module):
    def __init__(self, k=32, emb_dims=512, in_channels=3, output_channels=12):
        super(DGCNN, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.in_channels = in_channels
        self.output_channels = output_channels

        self.conv1 = EdgeGraphConvBlock(self.in_channels, 64, 64, "local_global", self.k)
        self.conv2 = EdgeGraphConvBlock(64, 64, 128, "local_global", self.k)
        self.conv3 = EdgeGraphConvBlock(128, 128, self.emb_dims, "local_global", self.k)

        self.final = nn.Sequential(
            BasicPointLayer(self.emb_dims + 128 + 64, 512),
            BasicPointLayer(512, 256),
            BasicPointLayer(256, self.output_channels, dropout=0, is_out=True)
        )

    def forward(self, x, pos):
        x1, idx1 = self.conv1(x, pos=pos)
        x2, idx2 = self.conv2(x1, pos=pos)
        x3, idx3 = self.conv3(x2, pos=pos)
        x = torch.cat([x1, x2, x3], dim=2)
        x = self.final(x)
        return x


class GroupAttentionDGCNN(nn.Module):
    def __init__(self, k=32, attention_groups=256, emb_dims=1024, in_channels=3, output_channels=12):
        super(GroupAttentionDGCNN, self).__init__()
        self.k = k
        self.attention_groups = attention_groups
        self.emb_dims = emb_dims
        self.in_channels = in_channels
        self.output_channels = output_channels

        self.conv1 = EdgeGraphConvBlock(self.in_channels, 64, 64, "local_global", self.k)
        self.conv2 = EdgeGraphConvBlock(64, 64, 128, "local_global", self.k)
        self.conv3 = EdgeGraphConvBlock(128, 128, 256, "local_global", self.k)

        self.hidden = BasicPointLayer(256 + 128 + 64, self.emb_dims)

        self.attention = GraphGroupSelfAttention(self.emb_dims, group_k=attention_groups, num_heads=self.emb_dims // 32)

        self.final = nn.Sequential(
            BasicPointLayer(self.emb_dims, 512),
            BasicPointLayer(512, 256),
            BasicPointLayer(256, self.output_channels, is_out=True)
        )

    def forward(self, x, pos):
        x1, idx1 = self.conv1(x, pos=pos)
        x2, idx2 = self.conv2(x1, pos=x)
        x3, idx3 = self.conv3(x2, pos=x)
        x = torch.cat([x1, x2, x3], dim=2)
        x = self.hidden(x)
        x = self.attention(x, pos)
        x = self.final(x)
        return x


class FoldingNet(nn.Module):
    """
    FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation
    """

    def __init__(self, in_channels=3, k=32, encoder_dim=512):
        super(FoldingNet, self).__init__()
        self.k = k
        self.encoder_dim = encoder_dim
        self.in_channels = in_channels

        # encoder
        self.conv1 = EdgeGraphConvBlock(self.in_channels, 64, 64, "local_global", self.k)
        self.conv2 = EdgeGraphConvBlock(64, 64, 128, "local_global", self.k)
        self.conv3 = EdgeGraphConvBlock(128, 128, 256, "local_global", self.k)

        self.hidden = BasicPointLayer(256 + 128 + 64, self.encoder_dim)

        self.decoder = FoldingDecoder(in_channel=self.encoder_dim)

    def forward(self, x, pos):
        x1, idx1 = self.conv1(x, pos=pos)
        x2, idx2 = self.conv2(x1, pos=pos)
        x3, idx3 = self.conv3(x2, pos=pos)
        x = torch.cat([x1, x2, x3], dim=2)
        x = self.hidden(x)

        x = x.transpose(2, 1)
        x = torch.max(x, dim=-1)[0]

        # decoder
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = FoldingNet(in_channels=6)
    print(model)
    x = torch.randn(10, 100, 6).float()
    pos = torch.randn(10, 100, 3).float()
    out = model(x, pos)
