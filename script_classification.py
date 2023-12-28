import argparse
import copy
import logging
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.sparse as sp
import os
import os.path
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset
import dgl.sparse as dglsp
import numpy as np
import torch
from tqdm import tqdm

import dgl
from dgl import LaplacianPE
# from dgl.nn import LaplacianPosEnc
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from collections import defaultdict
import itertools

from collections import deque

from gnnutils import make_masks, test, add_original_graph, load_webkb, load_planetoid, \
    load_wiki, load_bgp, load_film, load_airports, train_finetuning_class, train_finetuning_cluster, test_cluster

from collections import defaultdict
from collections import deque
from torch.utils.data import DataLoader, ConcatDataset
import random as rnd
import warnings

warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")

from models import Transformer_class, Transformer_cluster

MODEl_DICT = {"Transformer_class": Transformer_class}

db_name = 0


def update_evaluation_value(file_path, colume, row, value):
    try:
        df = pd.read_excel(file_path)

        df[colume][row] = value

        df.to_excel(file_path, sheet_name='data', index=False)

        return
    except:
        print("Error when saving results! Save again!")
        time.sleep(3)


def run_node_classification(args, index_excel, ds_name, output_path, file_name, data_all, num_features, out_size,
                            num_classes, g, adj_org, M, trans_logM, sim, phi, B, degree, k_transition, device,device_2,
                            num_epochs, current_epoch, aug_check,sim_check,phi_check,test_node_degree):
    print("running run_node_classification")

    index_excel = index_excel

    if True:
        dataset = args.dataset
        lr = args.lr
        dims =args.dims
        out_size = args.out_size
        num_layers = args.num_layers
        k_transition = args.k_transition
        alfa =args.alfa
        beta = args.beta
    
        print(f"Node class process - {index_excel}")

        cp_filename = output_path + f'{dataset}_{lr}_{dims}_{num_layers}_{k_transition}_{alfa}_{beta}.pt'

        if not os.path.exists(cp_filename):
            print(f"run_node_classification: no file: {cp_filename}")
            print(f"run_node_classification: no file: {cp_filename}")
            return None

        runs_acc = []
        for i in tqdm(range(1)):
            print(f'run_node_classification, run time: {i}')
            acc = run_epoch_node_classification(i, data_all, num_features, out_size, num_classes,
                                                g, adj_org, M, trans_logM, sim, phi, B, degree, k_transition,
                                                cp_filename, dims,
                                                num_layers, lr, device, device_2,num_epochs, current_epoch, aug_check,sim_check,phi_check,test_node_degree)
            runs_acc.append(acc)

        runs_acc = np.array(runs_acc) * 100


        final_msg = "Node Classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
        print(final_msg)


def run_epoch_node_classification(i, data, num_features, out_size, num_classes,
                                  g, adj_org, M, trans_logM, sim, phi, B, degree, k_transition, cp_filename, dims,
                                  num_layers, lr, device,device_2, num_epochs, current_epoch,aug_check,sim_check,phi_check,test_node_degree):
    graph_name = ""
    best_val_acc = 0
    best_model = None
    pat = 20
    best_epoch = 0

    # fine tuning & testing
    print('fine tuning ...')

    model = Transformer_class(num_features, out_size, num_classes, hidden_dim=dims,
                              num_layers=num_layers, num_heads=4, graph_name=graph_name,
                              cp_filename=cp_filename, aug_check = aug_check,sim_check = sim_check,phi_check = sim_check) #.to(device)

    dataset_1 = ['cora', 'citeseer', 'Photo','WikiCS']
    for ds in dataset_1:
        if ds in cp_filename:
            model.to(device)
    else:
        if torch.cuda.device_count() > 1:

            id_2 = int(str(device_2).split(":")[1])
            model = torch.nn.DataParallel(model, device_ids=[id_2])

    best_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print("creating  random mask")
    data = make_masks(data, val_test_ratio=0.2)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # save dataload
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask

    test_check = 0
    for epoch in range(1, num_epochs):
        current_epoch = epoch
        train_loss, train_acc = train_finetuning_class(model, data, train_mask, optimizer, device,device_2,  g=g, adj_org=adj_org,
                                                       M=M, trans_logM=trans_logM, sim=sim, phi=phi, B=B,
                                                       k_transition=k_transition,
                                                       pre_train=0, current_epoch=current_epoch)

        if epoch % 1 == 0:
            valid_acc, valid_f1 = test(model, data, val_mask, device,device_2, g=g, adj_org=adj_org, trans_logM=trans_logM,
                                       sim=sim, phi=phi, B=B, degree=degree,
                                       k_transition=k_transition, current_epoch=current_epoch, test_node_degree=0)

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_model = model
                best_epoch = epoch
                pat = (pat + 1) if (pat < 5) else pat
            else:
                pat -= 1

            if epoch % 1 == 0:
                print(
                    'Epoch: {:02d}, best_epoch: {:02d}, Train Loss: {:0.4f}, Train Acc: {:0.4f}, Val Acc: {:0.4f} '.format(
                        epoch, best_epoch, train_loss, train_acc, valid_acc))

            if epoch - best_epoch > 100:
                print("1 validation patience reached ... finish training")
                break

    # Testing
    test_check = 1
    test_acc, test_f1 = test(best_model, data, test_mask, device,device_2, g=g, adj_org=adj_org, trans_logM=trans_logM, sim=sim,
                             phi=phi, B=B, degree=degree,
                             k_transition=k_transition, current_epoch=current_epoch, test_node_degree=test_node_degree)
    print('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f}, F1_test: {:0.4f}'.format(
        best_epoch, best_val_acc, test_acc, test_f1))
    return test_acc, test_f1
def run_node_clustering(args, index_excel, ds_name, output_path, file_name, data_all, num_features, out_size,
                        num_classes, g, M, logM, sim, phi, B, k_transition, device,device_2, num_epochs, adj, d, n_edges,
                        current_epoch, aug_check,sim_check,phi_check):
    index_excel = index_excel

    if True:
        dataset = args.dataset
        lr = args.lr
        dims =args.dims
        out_size = args.out_size
        num_layers = args.num_layers
        k_transition = args.k_transition
        alfa =args.alfa
        beta = args.beta

        print(f"Node clustering process - {index_excel}")

        cp_filename = output_path + f'{dataset}_{lr}_{dims}_{num_layers}_{k_transition}_{alfa}_{beta}.pt'

        if os.path.isfile(cp_filename) == False:
            print(f"run_node_clustering: no file {cp_filename}")
            return None

        runs_acc = []
        for i in tqdm(range(1)):
            print(f'run time: {i}')
            _, _, _, _, q, c = run_epoch_node_clustering(i, data_all, num_features, out_size,
                                                                          num_classes,
                                                                          g, M, logM, sim, phi, B, k_transition,
                                                                          cp_filename, dims, num_layers, lr, device,device_2,
                                                                          num_epochs, adj, d, n_edges, current_epoch, aug_check,sim_check,phi_check)
            #runs_acc.append(acc)
            time.sleep(1)
        print('Node clustering results: \n Q: {:0.4f}, , C: {:0.4f}'.format(q,c))

def run_epoch_node_clustering(i, data, num_features, out_size, num_classes, g, M, logM, sim, phi, B, k_transition,
                              cp_filename, dims, num_layers, lr, device,device_2, num_epochs, adj, d, n_edges, current_epoch,aug_check,sim_check,phi_check):
    graph_name = ""
    best_val_acc = 0
    best_model = None
    pat = 20
    best_epoch = 0

    # fine tuning & testing
    print('fine tuning run_epoch_node_clustering...')

    model = Transformer_cluster(num_features, out_size, num_classes, hidden_dim=dims,
                                num_layers=num_layers, num_heads=4, graph_name=graph_name,
                                cp_filename=cp_filename, aug_check=aug_check,sim_check=sim_check,phi_check=phi_check) #.to(device)

    dataset_1 = ['cora', 'citeseer', 'Photo','WikiCS']
    for ds in dataset_1:
        if ds in cp_filename:
            model.to(device)
    else:
        if torch.cuda.device_count() > 1:
            id_2 = int(str(device_2).split(":")[1])
            model = torch.nn.DataParallel(model, device_ids=[id_2])

    best_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print("creating  random mask")
    data = make_masks(data, val_test_ratio=0.0)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # save dataload.
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask
    adj = torch.FloatTensor(adj).to(device)
    for epoch in range(1, num_epochs):
        current_epoch = epoch
        train_loss = train_finetuning_cluster(model, data, train_mask, optimizer, device, device_2,g=g,
                                              M=M, trans_logM=logM, sim=sim, phi=phi, B=B, k_transition=k_transition,
                                              pre_train=0, adj=adj, d=d, n_edges=n_edges, current_epoch=current_epoch)

        if epoch % 1 == 0:
            print('Epoch: {:02d}, Train Loss: {:0.4f}'.format(epoch, train_loss))

    # Testing

    acc, precision, recall, nmi, q, c = test_cluster(best_model, data, train_mask, optimizer, device,device_2, g=g,
                                                     M=M, trans_logM=logM, sim=sim, phi=phi, B=B,
                                                     k_transition=k_transition, pre_train=0, adj=adj, d=d,
                                                     n_edges=n_edges, current_epoch=current_epoch)

    
    return acc, precision, recall, nmi, q, c
