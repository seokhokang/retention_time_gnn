# -*- coding: utf-8 -*-
#
# Modified version of gin.py to accomodate continuous node and edge features
# Original version available at https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/gin.py
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Isomorphism Networks.
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from  dgl.nn import GINEConv

__all__ = ['GIN']

# pylint: disable=W0221, C0103
class GINLayer(nn.Module):
    r"""Single Layer GIN from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    Parameters
    ----------
    emb_dim : int
        The size of each embedding vector.
    batch_norm : bool
        Whether to apply batch normalization to the output of message passing.
        Default to True.
    activation : None or callable
        Activation function to apply to the output node representations.
        Default to None.
    """
    def __init__(self, emb_dim, batch_norm=True, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )  
        self.conv = GINEConv()
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : FloatTensor of shape (N, emb_dim)
            * Input node features
            * N is the total number of nodes in the batch of graphs
            * node_in_feats is the input node feature size
        edge_feats : list of LongTensor of shape (E, emb_dim)
            * Input edge features
            * E is the total number of edges in the batch of graphs
            * edge_in_feats is the input edge feature size
        Returns
        -------
        node_feats : float32 tensor of shape (N, emb_dim)
            Output node representations
        """
        node_feats = self.conv(g, node_feats, edge_feats)
        node_feats = self.mlp(node_feats)
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats

class GIN(nn.Module):
    r"""Graph Isomorphism Network from `Strategies for
    Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__
    This module is for updating node representations only.
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
    JK : str
        JK for jumping knowledge as in `Representation Learning on Graphs with
        Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__. It decides
        how we are going to combine the all-layer node representations for the final output.
        There can be four options for this argument, ``concat``, ``last``, ``max`` and ``sum``.
        Default to 'last'.
        * ``'concat'``: concatenate the output node representations from all GIN layers
        * ``'last'``: use the node representations from the last GIN layer
        * ``'max'``: apply max pooling to the node representations across all GIN layers
        * ``'sum'``: sum the output node representations from all GIN layers
    dropout : float
        Dropout to apply to the output of each GIN layer. Default to 0.5
    """
    def __init__(self, node_in_feats, edge_in_feats,
                 num_layers=5, emb_dim=300, dropout=0.5):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, emb_dim),
            nn.ReLU()
        )
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayer(emb_dim))
            else:
                self.gnn_layers.append(GINLayer(emb_dim, activation=F.relu))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.project_edge_feats[0].reset_parameters()
        self.project_edge_feats[-1].reset_parameters()

        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations
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
        final_node_feats : float32 tensor of shape (N, M)
            Output node representations, N for the number of nodes and
            M for output size.
        """
        node_embeds = self.project_node_feats(node_feats)
        edge_embeds = self.project_edge_feats(edge_feats)
        
        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_embeds = self.gnn_layers[layer](g, all_layer_node_feats[layer], edge_embeds)
            node_embeds = self.dropout(node_embeds)
            all_layer_node_feats.append(node_embeds)

        final_node_feats = all_layer_node_feats[-1]

        return final_node_feats