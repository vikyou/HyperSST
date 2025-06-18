from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class ProposedConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0,
                 act: Callable = nn.PReLU(), bias: bool = False, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        x = self.lin_n2e(x)
        e = x
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)
        n = x

        return n, e
