import argparse
import random
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from TriCL.loader import DatasetLoader
from TriCL.models1 import HyperEncoder, HyperDecoder, TriCL
from TriCL.utils import valid_node_edge_mask
from TriCL.evaluation import linear_evaluation
from TriCL.GAM import Graph_Attention_Union


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(model_type, num_negs):
    features, hyperedge_index = data.features, data.hyperedge_index
    features2, hyperedge_index2 = data.features2, data.hyperedge_index2
    x1, x2 = features, features2
    num_nodes, num_edges = data.num_nodes, data.num_edges

    model.train()
    optimizer.zero_grad(set_to_none=True)

    node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index, num_nodes, num_edges)
    node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = node_mask1.to(args.device) & node_mask2.to(args.device)
    edge_mask = edge_mask1.to(args.device) & edge_mask2.to(args.device)

    nn1, n1, e1, n2, e2, x11, x21, x12, x22 = model(x1, x2, hyperedge_index, num_nodes, num_edges)

    n1, n2 = model.node_projection(n1), model.node_projection(n2)
    e1, e2 = model.edge_projection(e1), model.edge_projection(e2)
    x11, x12, x21, x22 = model.feature_projection(x11), model.feature_projection(x12), model.feature_projection(x21), model.feature_projection(x22)

    loss_n = model.node_level_loss(n1, n2, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs) if model_type in ['tricl_ng', 'tricl'] else 0

    wc = 4
    lossc = loss_n + wc * loss_g

    criterion = nn.MSELoss()
    loss1 = criterion(x1, x11)
    loss2 = criterion(x2, x21)

    GAM = Graph_Attention_Union(input_chs=n1.shape[-1], output_chs=x11.shape[-1]).to(args.device)
    x_gam = GAM(n1, n2)
    y_gam = GAM(n2, n1)

    wg = 10
    loss3 = wg * criterion(x22, x_gam)
    loss4 = wg * criterion(x12, y_gam)

    l2_lambda = 0.01
    param_list = list(model.parameters())
    params1 = torch.cat([p.view(-1) for p in param_list])
    l2_loss = torch.norm(params1, p=2)
    lossg = (loss1 + loss2 + loss3 + loss4) / 4 + l2_lambda * l2_loss
    lossg = lossg.mean()

    w = 1
    loss = lossc + w * lossg

    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor',
                                 'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40', 'test'])
    parser.add_argument('--model_type', type=str, default='tricl', choices=['tricl_n', 'tricl_ng', 'tricl'])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    data = DatasetLoader().load(args.dataset).to(args.device)

    accs = []
    kss = []

    for seed in range(args.num_seeds):
        fix_seed(seed)
        encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        encoder1 = HyperEncoder(data.features2.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        decoder = HyperDecoder(params['hid_dim'], params['hid_dim'], data.features.shape[1], params['num_layers'])
        decoder1 = HyperDecoder(params['hid_dim'], params['hid_dim'], data.features2.shape[1], params['num_layers'])
        model = TriCL(encoder, encoder1, decoder, decoder1, params['proj_dim']).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for epoch in tqdm(range(1, params['epochs'] + 1)):
            train(args.model_type, num_negs=None)

        acc, ks = linear_evaluation(model.encoder, data, args.device)
        accs.append(acc)
        kss.append(ks)

    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}, Accuracy: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')

    kss = np.array(kss).reshape(-1, 3)
    kss_mean = list(np.mean(kss, axis=0))
    kss_std = list(np.std(kss, axis=0))
    print(f'[Final] dataset: {args.dataset}, kappa coefficient: {kss_mean[2]:.2f}+-{kss_std[2]:.2f}')