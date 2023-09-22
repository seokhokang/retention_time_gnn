import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from dataset import GraphDataset
from util import collate_2D_graphs

from gnn import MPNNPredictor, GINPredictor, GTNPredictor, GATPredictor

from model import Trainer
from argparse import ArgumentParser
from sklearn.metrics import mean_absolute_error, r2_score


parser = ArgumentParser()
parser.add_argument('--gnn', '-g', type=str, choices=['MPNN','GAT','GTN','GIN','GIN3'], default='GIN')
parser.add_argument('--seed', '-s', type=int, default=134)

args = parser.parse_args()


# configurations
model_type = args.gnn
random_state = args.seed
cuda = torch.device('cuda:0')

model_path = './model/model_SMRT.pt'
if not os.path.exists('./model/'): os.makedirs('./model/')
print('-- model_path: %s'%model_path)


# dataset
trndata = GraphDataset('SMRT', split = 'trn', seed = random_state)
tstdata = GraphDataset('SMRT', split = 'tst', seed = random_state)
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
batch_size = 128
val_size = int(np.round(1/9 * len(trndata)))
trnsubset, valsubset = torch.utils.data.random_split(trndata, [len(trndata) - val_size, val_size], torch.Generator().manual_seed(random_state))
trn_loader = DataLoader(dataset=trnsubset, batch_size=min([len(trnsubset), batch_size]), shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(dataset=valsubset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

trn_y = trndata.label
trn_y_mean, trn_y_std = np.mean(trn_y), np.std(trn_y)

trainer = Trainer(net, model_path, cuda)
trainer.target_path = model_path
trainer.training(trn_loader, val_loader, trn_y_mean, trn_y_std, method = 'scratch', opt = 'adam', verbose = True)
trainer.load(model_path)


# inference
tst_loader = DataLoader(dataset=tstdata, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

tst_y = tstdata.label
tst_y_preds = trainer.inference(tst_loader, trn_y_mean, trn_y_std)

tst_mae = mean_absolute_error(tst_y, tst_y_preds)
tst_medae = np.median(np.abs(tst_y - tst_y_preds))
tst_r2 = r2_score(tst_y, tst_y_preds)

print('test MAE', tst_mae)
print('test MedAE', tst_medae)
print('test R2', tst_r2)