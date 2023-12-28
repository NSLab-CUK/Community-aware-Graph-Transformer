import argparse
import copy
import logging
import math
import time
from pathlib import Path
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import os
import random
import dgl
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='scipy._lib.messagestream.MessageStream')
from gnnutils import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
    load_film, load_airports, load_amazon, load_coauthor, load_WikiCS, load_crocodile, load_Cora_ML

from util import get_B_sim_phi, getM_logM, load_dgl, get_A_D

from models import Transformer

MODEl_DICT = {"Transformer": Transformer}
from script_classification import run_node_classification, run_node_clustering, update_evaluation_value


def filter_rels(data, r):
    data = copy.deepcopy(data)
    mask = data.edge_color <= r
    data.edge_index = data.edge_index[:, mask]
    data.edge_weight = data.edge_weight[mask]
    data.edge_color = data.edge_color[mask]
    return data


def run(i, data, num_features, num_classes, g, M, adj_org, adj, logM, B, sim, phi, degree, n_edges, aug_check,
        sim_check, phi_check, pretrain_check,test_node_degree):
    if args.model in MODEl_DICT:
        model = MODEl_DICT[args.model](num_features,
                                       args.out_size,
                                       num_classes,
                                       hidden_dim=args.dims,
                                       num_layers=args.num_layers,
                                       num_heads=args.num_heads,
                                       k_transition=args.k_transition,
                                       aug_check=aug_check,
                                       sim_check=sim_check,
                                       phi_check=phi_check,
                                       alfa=args.alfa, beta=args.beta,
                                       )
    dataset_1 = ['cora', 'citeseer', 'Photo','WikiCS']

    if args.dataset in dataset_1:
        model.to(device)
    else:
        if torch.cuda.device_count() > 1:

            id_2 = int ((str(args.device_2)).split(":")[1])
            print(f'Running multi-GPUs {id_2}')
            model = torch.nn.DataParallel(model,  device_ids=[id_2])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    if args.custom_masks:
        # create random mask
        data = make_masks(data, val_test_ratio = 0.0)
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    best_val_acc = 0

    pat = 20
    best_model = model
    graph_name = args.dataset
    best_loss = 100000000

    best_epoch = 0

    num_epochs = args.epochs + 1

    if args.pretrain_check == 0:
        print('pretrain_check: False, fine tunning')
        epoch = 1
        torch.save(model,
                   '{}{}_{}_{}_{}_{}_{}_{}.pt'.format(args.output_path, args.dataset, args.lr, args.dims,
                                                      args.num_layers, args.k_transition, args.alfa, args.beta))
    else: # pre-training 
        print(f'pre-training')
        for epoch in range(1, num_epochs):

            train_loss = train(model, data, train_mask, optimizer, device,device_2, g=g, adj_org=adj_org,
                               trans_logM=logM, sim=sim, phi=phi, B=B, k_transition=args.k_transition, current_epoch=epoch,
                               alpha_1=args.alpha_1, alpha_2=args.alpha_2, alpha_3=args.alpha_3)

            if best_loss >= train_loss:
                best_model = model
                best_epoch = epoch
                best_loss = train_loss

            if epoch - best_epoch > 200:
                break
            if epoch % 1 == 0:
                print('Epoch: {:02d}, Best Epoch {:02d}, Train Loss: {:0.4f}'.format(epoch, best_epoch, train_loss))

        print(' saving model and embeddings')
        torch.save(best_model,
                   '{}{}_{}_{}_{}_{}_{}_{}.pt'.format(args.output_path, args.dataset, args.lr, args.dims,
                                                      args.num_layers, args.k_transition, args.alfa, args.beta))
        time.sleep(1)
    run_node_classification(args, args.index_excel, args.dataset, args.output_path, args.file_name, data, num_features,
                            args.out_size,
                            num_classes, g, adj_org, M, logM, sim, phi, B, degree, args.k_transition, device,device_2,
                            args.run_times_fine, current_epoch=epoch, aug_check=aug_check,sim_check=sim_check,phi_check=phi_check,test_node_degree = test_node_degree)

    time.sleep(1)
    
    run_node_clustering(args, args.index_excel, args.dataset, args.output_path, args.file_name, data, num_features,
                        args.out_size,
                        num_classes, g, M, logM, sim, phi, B, args.k_transition, device,device_2, args.run_times_fine, adj,
                        degree, n_edges, current_epoch=epoch, aug_check=aug_check,sim_check=sim_check,phi_check=phi_check)

    time.sleep(1)


import collections
from collections import defaultdict


def main():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = args.dataset + "-" + timestr + ".log"
    Path("./exp_logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename="exp_logs/" + log_file, filemode="w", level=logging.INFO)
    logging.info("Starting on device: %s", device)
    logging.info("Config: %s ", args)
    isbgp = False
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        og_data, _ = load_planetoid(args.dataset)

    elif args.dataset in ['Computers', 'Photo']:
        print('computers and photo processing...')
        assert args.custom_masks == True
        og_data,  _ = load_amazon(args.dataset)
    elif args.dataset in ['WikiCS']:
        print('WikiCS processing...')
        assert args.custom_masks == True
        og_data,  _ = load_WikiCS(args.dataset) 
    else:
        raise NotImplementedError

    data = og_data

    num_classes = len(data.y.unique())

    num_features = data.x.shape[1]
    nx_g = to_networkx(data, to_undirected=True)

    print(f'Loading dataset: {args.dataset}, data zize: {data.x.size()}')


    num_of_nodes = nx_g.number_of_nodes()
    print(f'number of nodes: {num_of_nodes}')

    adj, degree, n_edges = get_A_D(nx_g, num_of_nodes)

    ### load information

    path = "pts/" + args.dataset + "_kstep_" + str(args.k_transition) + ".pt"
    if not os.path.exists(path):
        M, logM, B, sim, phi = load_bias(nx_g, num_of_nodes, num_classes, data.x)

        M = torch.from_numpy(np.array(M)).float()
        logM = torch.from_numpy(np.array(logM)).float()
        B = torch.from_numpy(B).float()
        sim = torch.from_numpy(np.array(sim)).float()
        phi = torch.from_numpy(phi).float()

        print('saving M, logM,B, sim, phi')
        save_info(M, logM, B, sim, phi)

    else:
        print('file exist, loading M, logM, B, sim, phi')
        M, logM, B, sim, phi = load_info(path)

    sim[torch.isnan(sim)] = 0

    g = load_dgl(nx_g, data.x, sim, phi)

    adj_org = torch.from_numpy(adj)

    runs_acc = []

    for i in tqdm(range(args.run_times)):
        acc = run(i, data, num_features, num_classes, g, M, adj_org, adj, logM, B, sim, phi, degree, n_edges,
                  args.aug_check, args.sim_check, args.phi_check, args.pretrain_check, args.test_node_degree)
        runs_acc.append(acc)


def save_info(M, logM, B, sim, phi):
    path = "pts/"
    torch.save({"M": M, "logM": logM, "B": B, "sim": sim, "phi": phi},
               path + args.dataset + "_kstep_" + str(args.k_transition) + '.pt')
    print('save_info done.')


def load_info(path):
    dic = torch.load(path)
    M = dic['M']
    logM = dic['logM']
    B = dic['B']
    sim = dic['sim']

    phi = dic['phi']
    print('load_info done.')
    return M, logM, B, sim, phi


def load_bias(nx_g, num_nodes, n_class, X):
    M, logM = getM_logM(nx_g, num_nodes, kstep=args.k_transition)
    B, sim, phi = get_B_sim_phi(nx_g, M, num_nodes, n_class, X, kstep=args.k_transition)

  
    return M, logM, B, sim, phi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments")
    #
    parser.add_argument("--dataset", default="citeseer", help="Dataset")
    parser.add_argument("--model", default="Transformer", help="GNN Model")

    parser.add_argument("--run_times", type=int, default=1)

    parser.add_argument("--drop", type=float, default=0.5, help="dropout")
    parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

    # adding args
    parser.add_argument("--device", default="cuda:0", help="GPU ids")
    parser.add_argument("--device_2", default="cuda:0", help="GPU ids")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--dims", type=int, default=64, help="hidden dims")
    parser.add_argument("--out_size", type=int, default=64, help="outsize dims")

    parser.add_argument("--k_transition", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--output_path", default="outputs/", help="outputs model")

    parser.add_argument("--pretrain_check", type=int, default=0)
    parser.add_argument("--aug_check", type=int, default=1)
    parser.add_argument("--sim_check", type=int, default=1)
    parser.add_argument("--phi_check", type=int, default=1)

    parser.add_argument("--alfa", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)

    parser.add_argument("--run_times_fine", type=int, default=200)
    parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
    parser.add_argument("--file_name", default="outputs_excels/cora.xlsx", help="file_name dataset")

    parser.add_argument("--alpha_1", type=float, default=1)
    parser.add_argument("--alpha_2", type=float, default=1)
    parser.add_argument("--alpha_3", type=float, default=1)
    parser.add_argument("--test_node_degree", type=int, default= 0)


    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    device_2 = torch.device(args.device_2)
    print(f"Using device:{device}, device_2: {device_2}")
    main()