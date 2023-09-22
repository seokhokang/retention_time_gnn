# -*- coding: utf-8 -*-
#
# Modified version of gat_predictor.py to accomodate continuous node and edge features
# Original version available at https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/model_zoo/gat_predictor.py
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GIN-based model for regression and classification on graphs.
# pylint: disable= no-member, arguments-differ, invalid-name

import torch
import torch.nn as nn

from .gat import GAT, WeightedSumAndMax

__all__ = ['GATPredictor']

class GATPredictor(nn.Module):
    r"""GAT-based model for regression and classification on graphs.

    GAT is introduced in `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__.
    This model is based on GAT and can be used for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        multi-head results for intermediate GAT layers and compute mean of multi-head results
        for the last GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, ELU is applied for intermediate GAT layers and no activation is applied
        for the last GAT layer.
    biases : list of bool
        ``biases[i]`` gives whether to add bias for the i-th GAT layer. ``len(activations)``
        equals the number of GAT layers. By default, bias is added for all GAT layers.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    predictor_hidden_feats : int
        Size for hidden representations in the output MLP predictor. Default to 128.
    predictor_dropout : float
        The probability for dropout in the output MLP predictor. Default to 0.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 n_layers=5,
                 hidden_feats=300,
                 num_heads=1,
                 dropout=0.1,
                 agg_mode='flatten',
                 activation=nn.functional.elu,
                 bias=True,
                 n_tasks=1,
                 predictor_hidden_feats=256):
        super(GATPredictor, self).__init__()

        self.gnn = GAT(node_in_feats=node_in_feats,
                       edge_in_feats=edge_in_feats,
                       n_layers=n_layers,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       agg_mode=agg_mode,
                       activation=activation,
                       dropout=dropout,
                       bias=bias)

        gnn_out_feats = hidden_feats
        
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        self.predict = nn.Sequential(
            nn.Linear(2 * gnn_out_feats, predictor_hidden_feats), nn.ReLU(),
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