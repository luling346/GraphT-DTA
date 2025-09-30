import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_transformer_layer import GraphTransformerLayer

import dgl

"""
    Graph Transformer with edge features
    
"""


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers=10, node_dim=44, edge_dim=10, hidden_dim=128, out_dim=128, n_heads=8,
                 in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8):
        super(GraphTransformer, self).__init__()
        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.linear_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

    def forward(self, g):
        # input embedding
        g = g.to(self.device)
        h = g.ndata['atom'].float().to(self.device)
        h_lap_pos_enc = g.ndata['lap_pos_enc'].to(self.device)
        # e = g.edata['bond'].float().to(self.device)
        sign_flip = torch.rand(h_lap_pos_enc.size(1), device=h.device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        h_lap_pos_enc = h_lap_pos_enc * sign_flip.unsqueeze(0)

        h = self.linear_h(h)
        h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        h = h + h_lap_pos_enc
        h = self.in_feat_dropout(h)

        # e = self.linear_e(e)

        # convnets
        for conv in self.layers:
            h = conv(g, h)

        g.ndata['h'] = h

        h = dgl.sum_nodes(g, 'h')

        return h
