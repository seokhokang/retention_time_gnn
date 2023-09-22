import os, sys
import numpy as np
import torch
from dgl import graph
import pickle as pkl
from sklearn.model_selection import KFold


class GraphDataset():

    def __init__(self, name = 'SMRT', cv_id = 0, split = 'trn', seed = 134):

        self.n_splits = 10
        
        assert cv_id in list(range(self.n_splits))
        assert split in ['trn', 'tst']

        self.name = name
        self.cv_id = cv_id
        self.split = split
        self.seed = seed
        
        self.load()


    def load(self):

        [mol_dict] = np.load('./data/dataset_graph_%s.npz'%self.name, allow_pickle=True)['data']
        kf = KFold(n_splits = self.n_splits, random_state = 134, shuffle = True)
        cv_splits = [split for split in kf.split(range(len(mol_dict['label'])))]
        cv_splits = cv_splits[self.cv_id]
        
        if self.split == 'trn':
            mol_indices = np.array([i in cv_splits[0] for i in range(len(mol_dict['label']))], dtype = bool)
        elif self.split == 'tst':
            mol_indices = np.array([i in cv_splits[1] for i in range(len(mol_dict['label']))], dtype = bool)

        node_indices = np.repeat(mol_indices, mol_dict['n_node'])
        self.label = mol_dict['label'][mol_indices].reshape(-1,1)

        edge_indices = np.repeat(mol_indices, mol_dict['n_edge'])
        self.n_node = mol_dict['n_node'][mol_indices]
        self.n_edge = mol_dict['n_edge'][mol_indices]
        self.node_attr = mol_dict['node_attr'][node_indices]
        self.edge_attr = mol_dict['edge_attr'][edge_indices]
        self.src = mol_dict['src'][edge_indices]
        self.dst = mol_dict['dst'][edge_indices]

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
            
        assert len(self.n_node) == len(self.label)
            

    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()
        label = self.label[idx]

        return g, label
        
        
    def __len__(self):

        return len(self.label)