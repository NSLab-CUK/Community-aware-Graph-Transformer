#%%
from typing import Optional, Callable

import os.path as osp

import torch
import numpy as np

from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.datasets import Planetoid,  Amazon, WikipediaNetwork, Coauthor, WikiCS, SNAPDataset



def mask_init(self, num_train_per_class=20, num_val_per_class=30, seed=12345):
    num_nodes = self.data.y.size(0)
    self.train_mask = torch.zeros([num_nodes], dtype=torch.bool)
    self.val_mask = torch.zeros([num_nodes], dtype=torch.bool)
    self.test_mask = torch.ones([num_nodes], dtype=torch.bool)
    np.random.seed(seed)
    for c in range(self.num_classes):
        samples_idx = (self.data.y == c).nonzero().squeeze()
        perm = list(range(samples_idx.size(0)))
        np.random.shuffle(perm)
        perm = torch.as_tensor(perm).long()
        self.train_mask[samples_idx[perm][:num_train_per_class]] = True
        self.val_mask[samples_idx[perm][num_train_per_class:num_train_per_class + num_val_per_class]] = True
    self.test_mask[self.train_mask] = False
    self.test_mask[self.val_mask] = False


def mask_getitem(self, datum):
    datum.__setitem__("train_mask", self.train_mask)
    datum.__setitem__("val_mask", self.val_mask)
    datum.__setitem__("test_mask", self.test_mask)
    return datum


class DigitizeY(object):

    def __init__(self, bins, transform_y=None):
        self.bins = np.asarray(bins)
        self.transform_y = transform_y

    def __call__(self, data):
        y = self.transform_y(data.y).numpy()
        digitized_y = np.digitize(y, self.bins)
        data.y = torch.from_numpy(digitized_y)
        return data

    def __repr__(self):
        return '{}(bins={})'.format(self.__class__.__name__, self.bins.tolist())



class WikipediaNetwork_crocodile(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processing data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/master')

    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            print('test')
            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            y = torch.from_numpy(data['label']).to(torch.float)
            train_mask = torch.from_numpy(data['train_mask']).to(torch.bool)
            test_mask = torch.from_numpy(data['test_mask']).to(torch.bool)
            val_mask = torch.from_numpy(data['val_mask']).to(torch.bool)
            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

#%%
from ogb.nodeproppred import NodePropPredDataset

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def load_arxiv_year_dataset(root):
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv',root=root)
    graph = ogb_dataset.graph
    graph['edge_index'] = torch.as_tensor(graph['edge_index'])
    graph['node_feat'] = torch.as_tensor(graph['node_feat'])

    label = even_quantile_labels(graph['node_year'].flatten(), 5, verbose=False)
    label = torch.as_tensor(label).reshape(-1, 1)
    import os
    split_idx_lst = load_fixed_splits("arxiv-year",os.path.join(root,"splits"))

    train_mask = torch.stack([split["train"] for split in split_idx_lst],dim=1)
    val_mask = torch.stack([split["valid"] for split in split_idx_lst],dim=1)
    test_mask = torch.stack([split["test"] for split in split_idx_lst],dim=1)
    data = Data(x=graph["node_feat"],y=torch.squeeze(label.long()),edge_index=graph["edge_index"],\
                train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
    return data



def load_fixed_splits(dataset,split_dir):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    import os
    splits_lst = np.load(os.path.join(split_dir,"{}-splits.npy".format(name)), allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst
# %%