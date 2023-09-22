# -*- coding: utf-8 -*-
#
# Modified version of pagtn_predictor.py
# Original version available at https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/pagtn_predictor.py
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Path-Augmented Graph Transformer Network
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set

from .gtn import GTN

__all__ = ['GTNPredictor']

class GTNPredictor(nn.Module):
    """PAGTN model for regression and classification on graphs.

    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node features in PAGTN layers.
    node_hid_feats : int
        Size for the hidden node features in PAGTN layers.
    depth : int
        Number of PAGTN layers to be applied.
    nheads : int
        Number of attention heads.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
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
                 node_out_feats=256,
                 node_hid_feats=256,
                 depth=5,
                 nheads=1,
                 dropout=0.1,
                 activation=nn.LeakyReLU(0.2),
                 readout='mean',
                 n_tasks=1,
                 predictor_hidden_feats=256):
        super(GTNPredictor, self).__init__()
        self.gnn = GTN(node_in_feats, node_out_feats,
                       node_hid_feats, edge_in_feats,
                       depth, nheads, dropout, activation)
                              
        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            self.readout = GlobalAttentionPooling(gate_nn=nn.Linear(node_out_feats, 1))
        elif readout == 'set2set':
            self.readout = Set2Set()
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max', 'attention' or 'set2set', got {}".format(readout))
                             
        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, predictor_hidden_feats), nn.ReLU(),
            nn.Linear(predictor_hidden_feats, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        float32 tensor of shape (V, node_out_feats)
            Updated node features.
        """

        node_feats = self.gnn(g, node_feats, edge_feats)  
        graph_feats = self.readout(g, node_feats)
        output = self.predict(graph_feats)

        return output