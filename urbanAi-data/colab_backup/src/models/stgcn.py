import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvLayer(nn.Module):
    """
    Temporal Gated Convolution Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        # 1D conv over time dimension
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size), padding=(0, 0))
    
    def forward(self, x):
        # x: (batch, in_channels, nodes, time)
        x = self.conv(x)
        # GLU
        P, Q = torch.chunk(x, 2, dim=1)
        x = P * torch.sigmoid(Q)
        return x


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer using Chebyshev Polynomials.
    """
    def __init__(self, in_channels, out_channels, Ks):
        super(GraphConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, x, cheb_polynomials):
        # x: (batch, in_channels, nodes, time)
        # cheb_polynomials: (Ks, nodes, nodes)
        b, c, n, t = x.size()
        
        # Reshape x to (batch * time, nodes, in_channels)
        x = x.permute(0, 3, 2, 1).contiguous().view(-1, n, c)
        
        output = torch.zeros(b * t, n, self.out_channels).to(x.device)
        
        for k in range(self.Ks):
            # T_k: (nodes, nodes)
            T_k = cheb_polynomials[k]
            # T_k x: (batch*time, nodes, in_channels)
            T_k_x = torch.matmul(T_k, x)
            # T_k_x W: (batch*time, nodes, out_channels)
            output += torch.matmul(T_k_x, self.weight[k])
            
        output += self.bias
        # Reshape back to (batch, out_channels, nodes, time)
        output = output.view(b, t, n, self.out_channels).permute(0, 3, 2, 1).contiguous()
        return F.relu(output)


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Block.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, Ks):
        super(STGCNBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(in_channels, hidden_channels, kernel_size=kernel_size)
        self.gconv = GraphConvLayer(hidden_channels, hidden_channels, Ks=Ks)
        self.tconv2 = TemporalConvLayer(hidden_channels, out_channels, kernel_size=kernel_size)
        
        # Layer norm
        self.ln = nn.LayerNorm([out_channels])

    def forward(self, x, cheb_polynomials):
        # x: (batch, in_channels, nodes, time)
        res = x
        x = self.tconv1(x)
        x = self.gconv(x, cheb_polynomials)
        x = self.tconv2(x)
        
        # If input and output channels are same, add residual
        if res.size(1) == x.size(1) and res.size(3) == x.size(3):
            x = x + res
            
        # Apply layer norm on the channels dimension. LayerNorm expects last dims.
        # x: (batch, out_channels, nodes, time) -> (batch, nodes, time, out_channels)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class STGCN(nn.Module):
    """
    STGCN Model.
    """
    def __init__(self, blocks, n_nodes, history_steps, horizon_steps, features=1, Ks=3):
        super(STGCN, self).__init__()
        self.n_nodes = n_nodes
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.Ks = Ks
        
        modules = []
        in_channels = features
        for block in blocks:
            # block = (hidden_channels, out_channels, kernel_size)
            modules.append(STGCNBlock(in_channels, block[0], block[1], block[2], Ks))
            in_channels = block[1]
            
        self.st_blocks = nn.ModuleList(modules)
        
        # We need to figure out the temporal size after the STGCN blocks.
        # Each temporal conv reduces time dimension by (kernel_size - 1).
        # A block has 2 temporal convs, reducing time by 2 * (kernel_size - 1).
        temporal_size = history_steps
        for block in blocks:
            temporal_size -= 2 * (block[2] - 1)
            
        if temporal_size <= 0:
            raise ValueError("Too many STGCN blocks or too large kernel sizes for the given history_steps.")
            
        # Output layer
        self.fc1 = nn.Linear(temporal_size * in_channels, 64)
        self.fc2 = nn.Linear(64, horizon_steps * features)

    def forward(self, x, cheb_polynomials):
        # Input x shape from dataloader: (batch, time, nodes, features)
        # Expected shape for STGCN: (batch, features, nodes, time)
        x = x.permute(0, 3, 2, 1).contiguous()
        
        for block in self.st_blocks:
            x = block(x, cheb_polynomials)
            
        # x: (batch, out_channels, nodes, remaining_time)
        b, c, n, t = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b * n, -1) # (batch * nodes, out_channels * remaining_time)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # (batch * nodes, horizon_steps * features)
        
        # Reshape to (batch, horizon_steps, nodes, features)
        x = x.view(b, n, self.horizon_steps, -1) # (batch, nodes, horizon_steps, features)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x
