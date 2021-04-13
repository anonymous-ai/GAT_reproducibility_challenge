import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import math

from models.layers import *


class NodeModel(nn.Module):
    def __init__(self,config):
        super(NodeModel,self).__init__()
        self.network = config.network
        self.dp_rate = config.dp_rate

        if config.network=='GAT':
            self.layer1 = MultiHeadGATLayer(d_input=config.d_input,
                                            d_output=config.d_hidden,
                                            n_heads=config.n_heads,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='concat')
            self.layer2 = MultiHeadGATLayer(d_input=config.d_hidden*config.n_heads,
                                            d_output=config.nclasses,
                                            n_heads=1,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='mean')
            self.GAT_param_init()
        elif config.network=='GAT_dot':
            self.layer1 = MultiHeadDotAttLayer(d_input=config.d_input,
                                            d_output=config.d_hidden,
                                            n_heads=config.n_heads,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='concat')
            self.layer2 = MultiHeadDotAttLayer(d_input=config.d_hidden*config.n_heads,
                                            d_output=config.nclasses,
                                            n_heads=1,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='mean')
            self.GAT_param_init()
        elif config.network=='GAT_const':
            self.layer1 = ConstMultiHeadGATLayer(d_input=config.d_input,
                                            d_output=config.d_hidden,
                                            n_heads=config.n_heads,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='concat')
            self.layer2 = ConstMultiHeadGATLayer(d_input=config.d_hidden*config.n_heads,
                                            d_output=config.nclasses,
                                            n_heads=1,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='mean')
            self.GAT_param_init()
        elif config.network=='GCN':
            self.layer1 = GCNConv(
                                in_channels=config.d_input,
                                out_channels=config.d_hidden* config.n_heads)
            self.layer2 = GCNConv(
                                in_channels=config.d_hidden* config.n_heads,
                                out_channels=config.nclasses)


    def GAT_param_init(self):
        INIT = 0.01
        if self.network == 'GAT':
            self.layer1.W.data.uniform_(-INIT,INIT)
            self.layer1.a.data.uniform_(-INIT,INIT)
            self.layer2.W.data.uniform_(-INIT,INIT)
            self.layer2.a.data.uniform_(-INIT,INIT)

        elif self.network == 'GAT_dot':
            self.layer1.WK.data.uniform_(-INIT,INIT)
            self.layer1.WQ.data.uniform_(-INIT,INIT)
            self.layer1.WV.data.uniform_(-INIT,INIT)
            self.layer2.WK.data.uniform_(-INIT,INIT)
            self.layer2.WQ.data.uniform_(-INIT,INIT)
            self.layer2.WV.data.uniform_(-INIT,INIT)

        elif self.network == 'GAT_const':
            self.layer1.W.data.uniform_(-INIT,INIT)
            self.layer2.W.data.uniform_(-INIT,INIT)

    def forward(self, x, edge_idx):
        x = F.dropout(x, self.dp_rate, training=self.training)
        h1 = self.layer1(x,edge_idx)
        h1 = F.elu(h1)
        h11 = F.dropout(h1, self.dp_rate, training=self.training)
        h2 = self.layer2(h11,edge_idx)
        return h2

class GraphModel(nn.Module):
    def __init__(self,config):
        super(GraphModel,self).__init__()
        self.network = config.network
        self.dp_rate = config.dp_rate

        if config.network=='GAT':
            self.layer1 = MultiHeadGATLayer(d_input=config.d_input,
                                            d_output=config.d_hidden,
                                            n_heads=config.n_heads,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='concat')
            self.layer2 = MultiHeadGATLayer(d_input=config.d_hidden*config.n_heads,
                                            d_output=config.nclasses,
                                            n_heads=1,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='mean')
            self.GAT_param_init()
        elif config.network=='GAT_dot':
            self.layer1 = MultiHeadDotAttLayer(d_input=config.d_input,
                                            d_output=config.d_hidden,
                                            n_heads=config.n_heads,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='concat')
            self.layer2 = MultiHeadDotAttLayer(d_input=config.d_hidden*config.n_heads,
                                            d_output=config.nclasses,
                                            n_heads=1,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='mean')
            self.GAT_param_init()
        elif config.network=='GAT_const':
            self.layer1 = ConstMultiHeadGATLayer(d_input=config.d_input,
                                            d_output=config.d_hidden,
                                            n_heads=config.n_heads,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='concat')
            self.layer2 = ConstMultiHeadGATLayer(d_input=config.d_hidden*config.n_heads,
                                            d_output=config.nclasses,
                                            n_heads=1,
                                            att_dp_rate=config.dp_rate,
                                            alpha=config.alpha,
                                            aggregate='mean')
            self.GAT_param_init()
        elif config.network=='GCN':
            self.layer1 = GCNConv(
                                in_channels=config.d_input,
                                out_channels=config.d_hidden* config.n_heads)
            self.layer2 = GCNConv(
                                in_channels=config.d_hidden* config.n_heads,
                                out_channels=config.nclasses)


    def GAT_param_init(self):
        INIT = 0.01
        if self.network == 'GAT':
            self.layer1.W.data.uniform_(-INIT,INIT)
            self.layer1.a.data.uniform_(-INIT,INIT)
            self.layer2.W.data.uniform_(-INIT,INIT)
            self.layer2.a.data.uniform_(-INIT,INIT)

        elif self.network == 'GAT_dot':
            self.layer1.WK.data.uniform_(-INIT,INIT)
            self.layer1.WQ.data.uniform_(-INIT,INIT)
            self.layer1.WV.data.uniform_(-INIT,INIT)
            self.layer2.WK.data.uniform_(-INIT,INIT)
            self.layer2.WQ.data.uniform_(-INIT,INIT)
            self.layer2.WV.data.uniform_(-INIT,INIT)

        elif self.network == 'GAT_const':
            self.layer1.W.data.uniform_(-INIT,INIT)
            self.layer2.W.data.uniform_(-INIT,INIT)

    def forward(self, x, edge_idx, batch):
        x = F.dropout(x, self.dp_rate, training=self.training)
        h1 = self.layer1(x,edge_idx)
        h1 = F.elu(h1)
        h11 = F.dropout(h1, self.dp_rate, training=self.training)
        h2 = self.layer2(h11,edge_idx)

        h3 = global_mean_pool(h2,batch)
        return h3
