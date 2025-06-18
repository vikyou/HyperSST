import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph_Attention_Union(nn.Module):
    def __init__(self, input_chs, output_chs):
        super(Graph_Attention_Union, self).__init__()
        self.support = nn.Linear(input_chs, input_chs)
        self.query = nn.Linear(input_chs, input_chs)
        self.g = nn.Sequential(
            nn.Linear(input_chs, input_chs),
            nn.BatchNorm1d(input_chs),
            nn.ReLU()
        )
        self.fi = nn.Sequential(
            nn.Linear(input_chs*2, output_chs),
            nn.BatchNorm1d(output_chs),
            nn.ReLU()
        )

    def forward(self, zf, xf):
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        similar = torch.matmul(xf_trans.unsqueeze(1), zf_trans.unsqueeze(2))
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g.unsqueeze(1))
        embedding = embedding.squeeze(1)

        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output


