import torch
from torch import nn
from utils.sampling import knn, fps


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def get_graph_feature(x, k=20, idx=None, pos=None, edge_function='global'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if pos is None:
            idx = knn(x, k=k)
        else:
            idx = knn(pos, k=k)
    device = x.device

    idx_org = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if edge_function == 'global':
        feature = feature.permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local':
        feature = (feature - x).permute(0, 3, 1, 2).contiguous()
    elif edge_function == 'local_global':
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_org  # (batch_size, 2*num_dims, num_points, k)

class EdgeGraphConvBlock(nn.Module):
    """
    EdgeGraphConvBlock is a module that performs edge graph convolution on input features.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (str): Type of edge function to use. Can be "global", "local", or "local_global".
        k (int): Number of nearest neighbors to consider for local edge function. Default is 32.

    Raises:
        ValueError: If edge_function is not one of "global", "local", or "local_global".

    Attributes:
        edge_function (str): Type of edge function used.
        in_channels (int): Number of input channels.
        k (int): Number of nearest neighbors considered for local edge function.
        conv (nn.Sequential): Sequential module consisting of convolutional layers.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, k=32):
        super(EdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function in ["local_global"]:
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, idx=None, pos=None):
        """
        Forward pass of the EdgeGraphConvBlock.

        Args:
            x (torch.Tensor): Input features.
            idx (torch.Tensor, optional): Index tensor for graph construction of shape (B, N, K), where B is the batch size. Defaults to None.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions. Default is None.

        Returns:
            torch.Tensor: Output features after edge graph convolution.
            torch.Tensor: Updated index tensor.

        """
        x_t = x.transpose(2, 1)
        pos_t = pos.transpose(2, 1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx, pos=pos_t)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx
    
class DilatedEdgeGraphConvBlock(nn.Module):
    """
    A block implementing a dilated edge graph convolution operation.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        edge_function (str): Type of edge function to use. Must be one of "global", "local", or "local_global".
        dilation_k (int): Number of nearest neighbors to consider for the dilation operation.
        k (int): Number of nearest neighbors to consider for the graph convolution operation.

    Raises:
        ValueError: If `dilation_k` is smaller than `k` or if `edge_function` is not one of the allowed values.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_function, dilation_k=128, k=32):
        super(DilatedEdgeGraphConvBlock, self).__init__()
        self.edge_function = edge_function
        self.in_channels = in_channels
        if dilation_k < k:
            raise ValueError(f'Dilation k {dilation_k} must be larger than k {k}')
        self.dilation_k = dilation_k
        self.k = k
        if edge_function not in ["global", "local", "local_global"]:
            raise ValueError(
                f'Edge Function {edge_function} is not allowed. Only "global", "local" or "local_global" are valid')
        if edge_function in ["local_global"]:
            self.in_channels = self.in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, idx=None, pos=None):
        """
        Forward pass of the dilated edge graph convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size, N is the number of nodes,
                and C is the number of input channels.
            idx (torch.Tensor, optional): Index tensor of shape (B, N, K), where K is the number of nearest neighbors
                to consider for the graph convolution operation. Defaults to None.
            pos (torch.Tensor, optional): Position tensor of shape (B, N, D), where D is the number of dimensions.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_channels), where out_channels is the number of output channels.
            torch.Tensor: Index tensor of shape (B, N, K), representing the indices of the nearest neighbors.

        """
        x_t = x.transpose(2, 1)
        if pos is None:
            pos = x
        B, N, C = x.shape
        cd = torch.cdist(pos, pos, p=2)
        idx_l = torch.topk(cd, self.dilation_k, largest=False)[1].reshape(B * N, -1)
        idx_fps = fps(pos.reshape(B * N, -1)[idx_l], self.k)
        idx_fps = batched_index_select(idx_l, 1, idx_fps).reshape(B, N, -1)
        out, idx = get_graph_feature(x_t, edge_function=self.edge_function, k=self.k, idx=idx_fps)
        out = self.conv(out)
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.transpose(2, 1)
        return out, idx
    

class GraphGroupSelfAttention(nn.Module):
    """
    Graph Group Self-Attention module.

    Args:
        in_channels (int): Number of input channels.
        group_k (int, optional): Number of groups to divide the input into. Default is 32.
        num_heads (int, optional): Number of attention heads. Default is 3.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """

    def __init__(self, in_channels, group_k=32, num_heads=3, dropout=0.1):
        super(GraphGroupSelfAttention, self).__init__()
        self.group_k = group_k
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        group_idx = fps(x, self.group_k)
        groups = batched_index_select(x, 1, group_idx)  # (B, N, C) -> (B, group_k, C)
        attn_output, attn_output_weights = self.multihead_attn(x, groups, groups)
        return attn_output



