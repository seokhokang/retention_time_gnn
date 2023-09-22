import numpy as np
import os
import pickle as pkl
import torch

from torch.utils.data import DataLoader
from dataset import GraphDataset
from util import collate_2D_graphs

from gnn import MPNNPredictor, GINPredictor, GTNPredictor, GATPredictor

from model import Trainer
from argparse import ArgumentParser
from sklearn.metrics import mean_absolute_error, r2_score


parser = ArgumentParser()
parser.add_argument('--target', '-t', type=str)
parser.add_argument('--cvid', '-k', type=int, choices=list(range(10)), default=0)
parser.add_argument('--gnn', '-g', type=str, choices=['MPNN','GAT','GTN','GIN','GIN3'], default='GIN')
parser.add_argument('--method', '-m', type=str, choices=['scratch', 'feature', 'finetune'], default='finetune')
parser.add_argument('--opt', '-o', type=str, choices=['adam', 'lbfgs'], default='lbfgs')
parser.add_argument('--seed', '-s', type=int, default=134)

args = parser.parse_args()


# configurations
target = args.target
model_type = args.gnn
method = args.method
opt = args.opt
cv_id = args.cvid
random_state = args.seed
cuda = torch.device('cuda:0')

source_path = './model/model_SMRT.pt'
target_path = './model/model_%s_cv_%d.pt'%(target, cv_id)
if not os.path.exists('./model/'): os.makedirs('./model/')
print('-- source_path: %s'%source_path)
print('-- target_path: %s'%target_path)


# dataset
trndata = GraphDataset(target, cv_id = cv_id, split = 'trn', seed = random_state)
tstdata = GraphDataset(target, cv_id = cv_id, split = 'tst', seed = random_state)
print('-- trn/tst: %d/%d'%(len(trndata), len(tstdata)))


# model 
if model_type == 'GIN':
    net = GINPredictor(
        node_in_feats=trndata.node_attr.shape[1],
        edge_in_feats=trndata.edge_attr.shape[1]
    )
elif model_type == 'GIN3':
    net = GINPredictor(
        node_in_feats=trndata.node_attr.shape[1],
        edge_in_feats=trndata.edge_attr.shape[1],
        num_layers=3
    )
elif model_type == 'MPNN':
    net = MPNNPredictor(
        node_in_feats=trndata.node_attr.shape[1],
        edge_in_feats=trndata.edge_attr.shape[1]
    )
elif model_type == 'GAT':
    net = GATPredictor(
        node_in_feats=trndata.node_attr.shape[1],
        edge_in_feats=trndata.edge_attr.shape[1]
    )  
elif model_type == 'GTN':
    net = GTNPredictor(
        node_in_feats=trndata.node_attr.shape[1],
        edge_in_feats=trndata.edge_attr.shape[1]
    )  

collate_fn = collate_2D_graphs


# training
print('-- TRAINING')
trn_y = trndata.label
trn_y_mean, trn_y_std = np.mean(trn_y), np.std(trn_y)

if opt == 'adam':
    batch_size = 32
    val_size = int(np.round(1/9 * len(trndata)))
    trnsubset, valsubset = torch.utils.data.random_split(trndata, [len(trndata) - val_size, val_size], torch.Generator().manual_seed(random_state))
    trn_loader = DataLoader(dataset=trnsubset, batch_size=min([len(trnsubset), batch_size]), shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(dataset=valsubset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    trainer = Trainer(net, source_path, cuda)
    trainer.target_path = target_path
    trainer.training(trn_loader, val_loader, trn_y_mean, trn_y_std, method = method, opt = opt)
    
elif opt == 'lbfgs':
    trn_loader = DataLoader(dataset=trndata, batch_size=len(trndata), shuffle=False, collate_fn=collate_fn)

    trainer = Trainer(net, source_path, cuda)
    trainer.target_path = target_path
    trainer.training(trn_loader, None, trn_y_mean, trn_y_std, method = method, opt = opt)


# inference
tst_loader = DataLoader(dataset=tstdata, batch_size=len(tstdata), shuffle=False, collate_fn=collate_fn)

tst_y = tstdata.label
tst_y_preds = trainer.inference(tst_loader, trn_y_mean, trn_y_std)

tst_mae = mean_absolute_error(tst_y, tst_y_preds)
tst_medae = np.median(np.abs(tst_y - tst_y_preds))
tst_r2 = r2_score(tst_y, tst_y_preds)

print('test MAE', tst_mae)
print('test MedAE', tst_medae)
print('test R2', tst_r2)