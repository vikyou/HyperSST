from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from TriCL.layers import ProposedConv
from TriCL.layers2 import ProposedDeconv

class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x, e


class HyperDecoder(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperDecoder, self).__init__()
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.act = act

        self.deconvs = nn.ModuleList()
        if num_layers == 1:
            self.deconvs.append(ProposedDeconv(self.in_dim, self.edge_dim, self.out_dim, act=act))
        else:
            self.deconvs.append(ProposedDeconv(self.in_dim, self.edge_dim, self.out_dim, act=act))
            for _ in range(self.num_layers - 2):
                self.deconvs.append(ProposedDeconv(self.out_dim, self.edge_dim, self.out_dim, act=act))
            self.deconvs.append(ProposedDeconv(self.out_dim, self.edge_dim, self.out_dim, act=act))

    def reset_parameters(self):
        for conv in self.deconvs:
            conv.reset_parameters()

    def forward(self, x: Tensor, e: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x = self.deconvs[i](x, e, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x


class TriCL(nn.Module):
    def __init__(self, encoder: HyperEncoder, encoder1: HyperEncoder, decoder: HyperDecoder, decoder1: HyperDecoder, proj_dim: int):
        super(TriCL, self).__init__()
        self.encoder = encoder
        self.encoder1 = encoder1
        self.decoder = decoder
        self.decoder1 = decoder1

        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)
        self.fc1_x = nn.Linear(self.decoder.out_dim, proj_dim)
        self.fc2_x = nn.Linear(proj_dim, self.decoder.out_dim)
        self.fc11_x = nn.Linear(self.decoder1.out_dim, proj_dim)
        self.fc22_x = nn.Linear(proj_dim, self.decoder1.out_dim)

        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.encoder1.reset_parameters()
        self.decoder.reset_parameters()
        self.decoder1.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.fc1_x.reset_parameters()
        self.fc2_x.reset_parameters()
        self.disc.reset_parameters()

    def forward(self, x: Tensor, xx: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)

        n1, e1 = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)
        n2, e2 = self.encoder1(xx, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)

        x11 = self.decoder(n1, e1, self_loop_hyperedge_index, num_nodes, num_edges)
        x21 = self.decoder1(n2, e2, self_loop_hyperedge_index, num_nodes, num_edges)
        x12 = self.decoder(n2, e2, self_loop_hyperedge_index, num_nodes, num_edges)
        x22 = self.decoder1(n1, e1, self_loop_hyperedge_index, num_nodes, num_edges)

        nn1 = torch.cat((n1, n2), dim=1)
        return nn1, n1, e1[:num_edges], n2, e2[:num_edges], x11, x21, x12, x22

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                         num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)

    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))

    def feature_projection(self, z: Tensor):
        return self.fc2_x(F.elu(self.fc1_x(z)))

    def feature_projection1(self, z: Tensor):
        return self.fc22_x(F.elu(self.fc11_x(z)))

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)
            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2)
        return loss.mean() if mean else loss.sum()