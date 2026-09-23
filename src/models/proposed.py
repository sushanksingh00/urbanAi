import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .stgcn import TemporalConvLayer

class AdaptiveGraphConvLayer(nn.Module):
    """
    Graph Convolution Layer using an Adaptive Adjacency Matrix.
    """
    def __init__(self, in_channels, out_channels):
        super(AdaptiveGraphConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, x, adaptive_adj):
        # x: (batch, in_channels, nodes, time)
        # adaptive_adj: (nodes, nodes)
        b, c, n, t = x.size()
        
        # Reshape x to (batch * time, nodes, in_channels)
        x = x.permute(0, 3, 2, 1).contiguous().view(-1, n, c)
        
        # A * X : (nodes, nodes) x (batch*time, nodes, in_channels)
        x = torch.matmul(adaptive_adj, x) 
        
        # (A*X) * W
        output = torch.matmul(x, self.weight) + self.bias
        
        # Reshape back to (batch, out_channels, nodes, time)
        output = output.view(b, t, n, self.out_channels).permute(0, 3, 2, 1).contiguous()
        return F.relu(output)

class ASTGCNLiteBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size):
        super(ASTGCNLiteBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(in_channels, hidden_channels, kernel_size=kernel_size)
        self.gconv = AdaptiveGraphConvLayer(hidden_channels, hidden_channels)
        self.tconv2 = TemporalConvLayer(hidden_channels, out_channels, kernel_size=kernel_size)
        self.ln = nn.LayerNorm([out_channels])

    def forward(self, x, adaptive_adj):
        res = x
        x = self.tconv1(x)
        x = self.gconv(x, adaptive_adj)
        x = self.tconv2(x)
        if res.size(1) == x.size(1) and res.size(3) == x.size(3):
            x = x + res
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ASTGCNLite(nn.Module):
    """
    Proposed Architecture: ASTGCNLite (Adaptive Spatio-Temporal Graph Convolutional Network)
    It learns an adaptive adjacency matrix to capture hidden spatial dependencies.
    """
    def __init__(self, blocks, n_nodes, history_steps, horizon_steps, features=1, embed_dim=10):
        super(ASTGCNLite, self).__init__()
        self.n_nodes = n_nodes
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        
        # Learnable node embeddings for adaptive graph
        self.node_emb1 = nn.Parameter(torch.randn(n_nodes, embed_dim))
        self.node_emb2 = nn.Parameter(torch.randn(n_nodes, embed_dim))
        
        modules = []
        in_channels = features
        for block in blocks:
            modules.append(ASTGCNLiteBlock(in_channels, block[0], block[1], block[2]))
            in_channels = block[1]
            
        self.st_blocks = nn.ModuleList(modules)
        
        temporal_size = history_steps
        for block in blocks:
            temporal_size -= 2 * (block[2] - 1)
            
        if temporal_size <= 0:
            raise ValueError("Too many blocks or too large kernel sizes.")
            
        self.fc1 = nn.Linear(temporal_size * in_channels, 64)
        self.fc2 = nn.Linear(64, horizon_steps * features)

    def forward(self, x):
        # Adaptive graph generation
        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_emb1, self.node_emb2.T)), dim=1)
        
        # x: (batch, time, nodes, features) -> (batch, features, nodes, time)
        x = x.permute(0, 3, 2, 1).contiguous()
        
        for block in self.st_blocks:
            x = block(x, adaptive_adj)
            
        b, c, n, t = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b * n, -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = x.view(b, n, self.horizon_steps, -1).permute(0, 2, 1, 3).contiguous()
        return x
