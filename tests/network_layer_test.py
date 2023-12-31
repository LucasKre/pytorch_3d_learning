from pytorch3Dlearning.networks.layers.graph_layers import DilatedEdgeGraphConvBlock, EdgeGraphConvBlock, GraphGroupSelfAttention
import torch
import torch.nn as nn
import unittest

class TestDilatedEdgeGraphConvBlock(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.hidden_channels = 64
        self.out_channels = 128
        self.edge_function = "global"
        self.dilation_k = 64
        self.k = 32

    def test_initialization(self):
        block = DilatedEdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, self.edge_function, self.dilation_k, self.k)
        self.assertIsInstance(block, nn.Module)
        self.assertEqual(block.in_channels, self.in_channels)
        self.assertEqual(block.dilation_k, self.dilation_k)
        self.assertEqual(block.k, self.k)
        self.assertEqual(block.edge_function, self.edge_function)

    def test_forward(self):
        block = DilatedEdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, self.edge_function, self.dilation_k, self.k)
        x = torch.randn(10, 100, self.in_channels)
        pos = torch.randn(10, 100, 3)
        out, idx = block(x, pos=pos)
        self.assertEqual(out.shape, (10, 100, self.out_channels))
        self.assertEqual(idx.shape, (10, 100, self.k))

    def test_invalid_dilation_k(self):
        with self.assertRaises(ValueError):
            DilatedEdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, self.edge_function, 10, self.k)

    def test_invalid_edge_function(self):
        with self.assertRaises(ValueError):
            DilatedEdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, "invalid", self.dilation_k, self.k)

class TestEdgeGraphConvBlock(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.hidden_channels = 64
        self.out_channels = 128
        self.edge_function = "global"
        self.k = 32

    def test_initialization(self):
        block = EdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, self.edge_function, self.k)
        self.assertIsInstance(block, nn.Module)
        self.assertEqual(block.in_channels, self.in_channels)
        self.assertEqual(block.k, self.k)
        self.assertEqual(block.edge_function, self.edge_function)

    def test_forward(self):
        block = EdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, self.edge_function, self.k)
        x = torch.randn(10, 100, self.in_channels)
        pos = torch.randn(10, 100, 3)
        out, idx = block(x, pos=pos)
        self.assertEqual(out.shape, (10, 100, self.out_channels))
        self.assertEqual(idx.shape, (10, 100, self.k))

    def test_invalid_edge_function(self):
        with self.assertRaises(ValueError):
            EdgeGraphConvBlock(self.in_channels, self.hidden_channels, self.out_channels, "invalid", self.k)


class TestGraphGroupSelfAttention(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.group_k = 32
        self.num_heads = 3
        self.dropout = 0.1

    def test_initialization(self):
        block = GraphGroupSelfAttention(self.in_channels, self.group_k, self.num_heads, self.dropout)
        self.assertIsInstance(block, nn.Module)
        self.assertEqual(block.in_channels, self.in_channels)
        self.assertEqual(block.group_k, self.group_k)
        self.assertEqual(block.num_heads, self.num_heads)
        self.assertIsInstance(block.multihead_attn, nn.MultiheadAttention)

    def test_forward(self):
        block = GraphGroupSelfAttention(self.in_channels, self.group_k, self.num_heads, self.dropout)
        x = torch.randn(10, 100, self.in_channels)
        out = block(x)
        self.assertEqual(out.shape, (10, 100, self.in_channels))



if __name__ == "__main__":
    unittest.main()