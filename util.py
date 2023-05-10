import torch
import dgl
import numpy as np


def collate_2D_graphs(batch):

    g_list, label_list = map(list, zip(*batch))
    
    g_list = dgl.batch(g_list)
    label_list = torch.FloatTensor(np.vstack(label_list))
    
    return g_list, g_list.ndata['node_attr'], g_list.edata['edge_attr'], label_list