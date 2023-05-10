import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from dataset import GraphDataset
from util import collate_2D_graphs

from gnn.gin_predictor import GINPredictor

from model import Trainer
from sklearn.metrics import mean_absolute_error


# configurations
random_state = 134
cuda = torch.device('cuda:0')

model_path = './model/model_SMRT.pt'
if not os.path.exists('./model/'): os.makedirs('./model/')
print('-- model_path: %s'%model_path)


# dataset
trndata = GraphDataset('SMRT', split = 'trn', seed = random_state)
valdata = GraphDataset('SMRT', split = 'tst', seed = random_state)
print('-- trn/val: %d/%d'%(len(trndata), len(valdata)))


# model 
net = GINPredictor(
    node_in_feats=trndata.node_attr.shape[1],
    edge_in_feats=trndata.edge_attr.shape[1]
)
collate_fn = collate_2D_graphs


# training
print('-- TRAINING')
train_y = trndata.label
train_y_mean, train_y_std = np.mean(train_y), np.std(train_y)

batch_size = 128
trn_loader = DataLoader(dataset=trndata, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

trainer = Trainer(net, batch_size, model_path, cuda)
trainer.target_path = model_path
trainer.training(trn_loader, val_loader, train_y_mean, train_y_std, method = 'scratch', opt = 'adam', verbose = True)