# -*- coding: utf-8 -*-
#
# Modified version of gin_predictor.py to accomodate continuous node and edge features
# Original version available at https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/gin_predictor.py
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GIN-based model for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set

from .gin import GIN

__all__ = ['GINPredictor']

# pylint: disable=W0221
class GINPredictor(nn.Module):
    """GIN-based model for regression and classification on graphs.

    GIN was first introduced in `How Powerful Are Graph Neural Networks
    <https://arxiv.org/abs/1810.00826>`__ for general graph property
    prediction problems. It was further extended in `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    for pre-training and semi-supervised learning on large-scale datasets.

    For classification tasks, the output will be logits, i.e. values before
    sigmoid or softmax.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    num_layers : int
        Number of GIN layers to use. Default to 5.
    emb_dim : int
        The size of each embedding vector. Default to 300.
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5.
    readout : str
        Readout for computing graph representations out of node representations, which
        can be ``'sum'``, ``'mean'``, ``'max'``, ``'attention'``, or ``'set2set'``. Default
        to 'mean'.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=5,
                 emb_dim=300,
                 dropout=0.1,
                 readout='mean',
                 n_tasks=1,
                 predictor_hidden_feats=256):
        super(GINPredictor, self).__init__()

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn = GIN(node_in_feats=node_in_feats, 
                       edge_in_feats=edge_in_feats,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       dropout=dropout)

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            self.readout = GlobalAttentionPooling(
                gate_nn=nn.Linear(emb_dim, 1))
        elif readout == 'set2set':
            self.readout = Set2Set()
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max', 'attention' or 'set2set', got {}".format(readout))

        self.predict = nn.Sequential(
            nn.Linear(emb_dim, predictor_hidden_feats), nn.ReLU(),
            nn.Linear(predictor_hidden_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(g, node_feats, edge_feats)        
        graph_feats = self.readout(g, node_feats)
        output = self.predict(graph_feats)

        return output