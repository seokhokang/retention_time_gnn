# -*- coding: utf-8 -*-
#
# Modified version of gat.py to accomodate continuous node and edge features
# Original version available at https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/gat.py
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Graph Attention Networks
#
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import EdgeGATConv, WeightAndSum

__all__ = ['GAT', 'WeightedSumAndMax']

# pylint: disable=W0221
class GATLayer(nn.Module):
    r"""Single GAT layer from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    in_node_feats : int
        Number of input node features
    in_edge_feats : int
        Number of input edge features
    out_node_feats : int
        Number of output node features
    out_edge_feats : int
        Number of output edge features
    num_heads : int
        Number of attention heads
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all-head results or 'mean' for averaging
        all head results.
    bias : bool
        Whether to use bias in the GAT layer.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 num_heads,
                 agg_mode="flatten",
                 activation=None,
                 dropout=0.1,
                 bias=True):
        super(GATLayer, self).__init__()

        self.gat_conv = EdgeGATConv(
            in_feats=node_in_feats,
            edge_feats=edge_in_feats,
            out_feats=node_out_feats,
            num_heads=num_heads,
            activation=activation,
            feat_drop=dropout,
            bias=bias
        )
        assert agg_mode in ["flatten", "mean"]
        self.agg_mode = agg_mode

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gat_conv.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Update node representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        node_feats/edge_feats : float32 tensors of shapes (V, out_feats) and (V, out_feats)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              out_feats in initialization if self.agg_mode == 'mean' and
              out_feats * num_heads in initialization otherwise.
        """
        node_feats = self.gat_conv(g, node_feats, edge_feats)
        if self.agg_mode == "flatten":
            node_feats = node_feats.flatten(1)
        else:
            node_feats = node_feats.mean(1)

        return node_feats


class GAT(nn.Module):
    r"""GAT from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : list of int
        ``hidden_feats`` gives the output size of an attention head in the GAT layer.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    agg_mode : str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        By default, we flatten all-head results for each GAT layer.
    activation : list of activation function or None
        ``activation`` gives the activation function applied to the aggregated multi-head
        results for the GAT layer. By default, no activation is applied for each GAT layer.
    bias : list of bool
        ``bias`` gives whether to use bias for the GAT layer. By default, we use bias for all GAT layers.
    """

    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        n_layers=5,
        hidden_feats=32,
        num_heads=4,
        agg_mode='flatten',
        activation=nn.functional.elu,
        dropout=0.1,
        bias=True,
    ):
        super(GAT, self).__init__()
                
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats),
            nn.ReLU()
        )
        
        node_in_feats = hidden_feats        
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == n_layers - 1:
                agg_mode = 'mean'
                activation = None
            
            self.gnn_layers.append(
                GATLayer(
                    node_in_feats,
                    edge_in_feats,
                    hidden_feats,
                    num_heads,
                    agg_mode,
                    activation,
                    dropout,
                    bias,
                )
            )
            if agg_mode == "flatten":
                node_in_feats = hidden_feats * num_heads
            else:
                node_in_feats = hidden_feats
        
    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

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
        node_feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        """
        node_feats = self.project_node_feats(node_feats)
        
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats, edge_feats)
            
        return node_feats
        
        
class WeightedSumAndMax(nn.Module):
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.

    Parameters
    ----------
    in_feats : int
        Input node feature size
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g