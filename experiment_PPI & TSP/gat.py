import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

sys.path.append('..')

from models import MultiHeadGATLayer, ConstMultiHeadGATLayer, MultiHeadDotAttLayer

INIT = 0.1


class PPIGAT(nn.Module):
	def __init__(self,layer_type,d_input,d_hidden:list,nheads:list,dp_rate,alpha,skip=True):
		super(PPIGAT, self).__init__()

		if skip:
			assert d_hidden[0]*nheads[0]==d_hidden[1]*nheads[1]

		if layer_type=='gat':
			self.att_layer1 = MultiHeadGATLayer(d_input,d_hidden[0],nheads[0],att_dp_rate=dp_rate,alpha=alpha,aggregate='concat')
			self.att_layer2 = MultiHeadGATLayer(d_hidden[0]*nheads[0],d_hidden[1],nheads[1],att_dp_rate=dp_rate,alpha=alpha,aggregate='concat')
			self.att_layer3 = MultiHeadGATLayer(d_hidden[1]*nheads[1],d_hidden[2],nheads[2],att_dp_rate=dp_rate,alpha=alpha,aggregate='mean')
		elif layer_type=='const':
			self.att_layer1 = ConstMultiHeadGATLayer(d_input,d_hidden[0],nheads[0],att_dp_rate=dp_rate,alpha=alpha,aggregate='concat')
			self.att_layer2 = ConstMultiHeadGATLayer(d_hidden[0]*nheads[0],d_hidden[1],nheads[1],att_dp_rate=dp_rate,alpha=alpha,aggregate='concat')
			self.att_layer3 = ConstMultiHeadGATLayer(d_hidden[1]*nheads[1],d_hidden[2],nheads[2],att_dp_rate=dp_rate,alpha=alpha,aggregate='mean')
		elif layer_type=='transformer':
			self.att_layer1 = MultiHeadDotAttLayer(d_input,d_hidden[0],nheads[0],att_dp_rate=dp_rate,alpha=alpha,aggregate='concat')
			self.att_layer2 = MultiHeadDotAttLayer(d_hidden[0]*nheads[0],d_hidden[1],nheads[1],att_dp_rate=dp_rate,alpha=alpha,aggregate='concat')
			self.att_layer3 = MultiHeadDotAttLayer(d_hidden[1]*nheads[1],d_hidden[2],nheads[2],att_dp_rate=dp_rate,alpha=alpha,aggregate='mean')
		elif layer_type=='gcn':
			self.att_layer1 = GCNConv(d_input,d_hidden[0]*nheads[0],improved=False,cached=False,add_self_loops=True,normalize=True,bias=True)
			self.att_layer2 = GCNConv(d_hidden[0]*nheads[0],d_hidden[1]*nheads[1],improved=False,cached=False,add_self_loops=True,normalize=True,bias=True)
			self.att_layer3 = GCNConv(d_hidden[1]*nheads[1],d_hidden[2],improved=False,cached=False,add_self_loops=True,normalize=True,bias=True)
		else:
			print('Invalid att layer type')

		self.layer_type = layer_type
		self.nonlinearity = F.elu if layer_type in ['gat','transformer','const'] else F.relu
		self.dp_rate = dp_rate
		self.skip = skip

	def param_init(self):
		
		if self.layer_type=='gat':
			nn.init.xavier_uniform_(self.att_layer1.W.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer1.a.data,gain=1.0) #1.414)
			# attention.W.data.uniform_(-INIT,INIT)
			# attention.a.data.uniform_(-INIT,INIT)
			
			nn.init.xavier_uniform_(self.att_layer2.W.data, gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.a.data, gain=1.0) #1.414)
			# attention.W.data.uniform_(-INIT,INIT)
			# attention.a.data.uniform_(-INIT,INIT)

			nn.init.xavier_uniform_(self.att_layer3.W.data, gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.a.data, gain=1.0) #1.414)
		
		elif self.layer_type=='const':
			nn.init.xavier_uniform_(self.att_layer1.W.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.W.data, gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.W.data, gain=1.0) #1.414)
		
		elif self.layer_type=='transformer':
			nn.init.xavier_uniform_(self.att_layer1.WK.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer1.WQ.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer1.WV.data,gain=1.0) #1.414)
			
			nn.init.xavier_uniform_(self.att_layer2.WK.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.WQ.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.WV.data,gain=1.0) #1.414)

			nn.init.xavier_uniform_(self.att_layer3.WK.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.WQ.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.WV.data,gain=1.0) #1.414)
		
		elif self.layer_type=='gcn':
			self.att_layer1.reset_parameters()
			self.att_layer2.reset_parameters()
			self.att_layer3.reset_parameters()
		else:
			print('Invalid att layer type')

	def forward(self, x, edge_idx):
    	# x~(n_node,d_input), edge_idx~(2,E)
		if self.dp_rate:
			x = F.dropout(x, self.dp_rate, training=self.training)
		h1 = self.nonlinearity(self.att_layer1(x,edge_idx)) # (n_node,nheads[0]*d_hidden[0])
		if self.dp_rate:
			h1 = F.dropout(h1, self.dp_rate, training=self.training)
		h2 = self.nonlinearity(self.att_layer2(h1,edge_idx)) # (n_node,nheads[1]*d_hidden[1])
		if self.dp_rate:
			h2 = F.dropout(h2, self.dp_rate, training=self.training)
		if self.skip:
			h3_input = h1+h2
		else:
			h3_input = h2
		return self.att_layer3(h3_input,edge_idx) # (n_node,d_hidden[2])



class TSPBlock(nn.Module):
	def __init__(self,layer_type,d_input,d_hidden:list,nheads:list,alpha,skip=True,aggr='concat'):
		super(TSPBlock, self).__init__()

		if skip:
			assert d_hidden[0]*nheads[0]==d_hidden[1]*nheads[1]

		if layer_type=='gat':
			self.att_layer1 = MultiHeadGATLayer(d_input,d_hidden[0],nheads[0],att_dp_rate=None,alpha=alpha,aggregate='concat')
			self.att_layer2 = MultiHeadGATLayer(d_hidden[0]*nheads[0],d_hidden[1],nheads[1],att_dp_rate=None,alpha=alpha,aggregate='concat')
			self.att_layer3 = MultiHeadGATLayer(d_hidden[1]*nheads[1],d_hidden[2],nheads[2],att_dp_rate=None,alpha=alpha,aggregate=aggr)
		elif layer_type=='const':
			self.att_layer1 = ConstMultiHeadGATLayer(d_input,d_hidden[0],nheads[0],att_dp_rate=None,alpha=alpha,aggregate='concat')
			self.att_layer2 = ConstMultiHeadGATLayer(d_hidden[0]*nheads[0],d_hidden[1],nheads[1],att_dp_rate=None,alpha=alpha,aggregate='concat')
			self.att_layer3 = ConstMultiHeadGATLayer(d_hidden[1]*nheads[1],d_hidden[2],nheads[2],att_dp_rate=None,alpha=alpha,aggregate=aggr)
		elif layer_type=='transformer':
			self.att_layer1 = MultiHeadDotAttLayer(d_input,d_hidden[0],nheads[0],att_dp_rate=None,alpha=alpha,aggregate='concat')
			self.att_layer2 = MultiHeadDotAttLayer(d_hidden[0]*nheads[0],d_hidden[1],nheads[1],att_dp_rate=None,alpha=alpha,aggregate='concat')
			self.att_layer3 = MultiHeadDotAttLayer(d_hidden[1]*nheads[1],d_hidden[2],nheads[2],att_dp_rate=None,alpha=alpha,aggregate=aggr)
		elif layer_type=='gcn':
			self.att_layer1 = GCNConv(d_input,d_hidden[0]*nheads[0],improved=False,cached=False,add_self_loops=True,normalize=True,bias=False)
			self.att_layer2 = GCNConv(d_hidden[0]*nheads[0],d_hidden[1]*nheads[1],improved=False,cached=False,add_self_loops=True,normalize=True,bias=False)
			self.att_layer3 = GCNConv(d_hidden[1]*nheads[1],d_hidden[2]*nheads[2],improved=False,cached=False,add_self_loops=True,normalize=True,bias=False)
		else:
			print('Invalid att layer type')

		self.bn1 = nn.BatchNorm1d(d_hidden[0]*nheads[0],affine=True,track_running_stats=True)
		self.bn2 = nn.BatchNorm1d(d_hidden[1]*nheads[1],affine=True,track_running_stats=True)
		self.bn3 = nn.BatchNorm1d(d_hidden[2]*nheads[2],affine=True,track_running_stats=True)

		self.layer_type = layer_type
		self.nonlinearity = F.elu if layer_type in ['gat','transformer','const'] else F.relu
		# self.dp_rate = dp_rate
		self.skip = skip

	def param_init(self):
		
		if self.layer_type=='gat':
			nn.init.xavier_uniform_(self.att_layer1.W.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer1.a.data,gain=1.0) #1.414)
			# attention.W.data.uniform_(-INIT,INIT)
			# attention.a.data.uniform_(-INIT,INIT)
			
			nn.init.xavier_uniform_(self.att_layer2.W.data, gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.a.data, gain=1.0) #1.414)
			# attention.W.data.uniform_(-INIT,INIT)
			# attention.a.data.uniform_(-INIT,INIT)

			nn.init.xavier_uniform_(self.att_layer3.W.data, gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.a.data, gain=1.0) #1.414)
		
		elif self.layer_type=='const':
			nn.init.xavier_uniform_(self.att_layer1.W.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.W.data, gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.W.data, gain=1.0) #1.414)
		
		elif self.layer_type=='transformer':
			nn.init.xavier_uniform_(self.att_layer1.WK.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer1.WQ.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer1.WV.data,gain=1.0) #1.414)
			
			nn.init.xavier_uniform_(self.att_layer2.WK.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.WQ.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer2.WV.data,gain=1.0) #1.414)

			nn.init.xavier_uniform_(self.att_layer3.WK.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.WQ.data,gain=1.0) #1.414)
			nn.init.xavier_uniform_(self.att_layer3.WV.data,gain=1.0) #1.414)
		elif self.layer_type=='gcn':
			self.att_layer1.reset_parameters()
			self.att_layer2.reset_parameters()
			self.att_layer3.reset_parameters()
		else:
			print('Invalid att layer type')

	def forward(self, x, edge_idx):
    	# x~(n_node,d_input), edge_idx~(2,E)
		h1 = self.nonlinearity(self.bn1(self.att_layer1(x,edge_idx))) # (n_node,nheads[0]*d_hidden[0])
		
		h2 = self.nonlinearity(self.bn2(self.att_layer2(h1,edge_idx))) # (n_node,nheads[1]*d_hidden[1])
		
		if self.skip:
			h3_input = h1+h2
		else:
			h3_input = h2
		return self.bn3(self.att_layer3(h3_input,edge_idx)) # (n_node,nheads[2]*d_hidden[2])

class TSPGAT(nn.Module):
	def __init__(self,layer_type,nblocks,d_input,d_hidden:list,nheads:list,dp_rate,alpha,skip=True):
		super(TSPGAT, self).__init__()

		if skip:
			assert d_hidden[0]*nheads[0]==d_hidden[1]*nheads[1]

		self.layers = nn.ModuleList()
		self.layers.append(TSPBlock(layer_type,d_input,d_hidden,nheads,alpha,skip,'concat'))
		for _ in range(nblocks-1):
			self.layers.append(TSPBlock(layer_type,nheads[-1]*d_hidden[-1],d_hidden,nheads,alpha,skip,'concat'))
		
		self.mlp = nn.Sequential(nn.Linear(2*nheads[-1]*d_hidden[-1],d_hidden[-1],bias=False),
			nn.BatchNorm1d(d_hidden[-1],affine=True,track_running_stats=True), # BN over the edge dimension
			nn.ReLU(),
			nn.Linear(d_hidden[-1],1,bias=False),
			nn.BatchNorm1d(1,affine=True,track_running_stats=True))

		self.nonlinearity = F.elu if layer_type in ['gat','transformer','const'] else F.relu

		self.layer_type = layer_type
		self.nblocks = nblocks
		self.dp_rate = dp_rate
		self.skip = skip

	def param_init(self):
		
		for block in self.layers:
			block.param_init()

		self.mlp[0].weight.data.uniform_(-INIT,INIT)
		# self.mlp[0].bias.data.zero_()
		self.mlp[3].weight.data.uniform_(-INIT,INIT)
		# self.mlp[2].bias.data.zero_()

	def forward(self, x, edge_idx):
    	# x~(n_node,d_input), edge_idx~(2,E)
		self_loop = torch.tensor([range(x.size(0)),range(x.size(0))],device=edge_idx.device.type) # (2,n_node)
		edge_idx_prime = torch.cat([edge_idx,self_loop],dim=1) # (2,E')
		h = x
		for block in self.layers:
			h = self.nonlinearity(block(h,edge_idx_prime)) # (n_node,nheads[2]*d_hidden[2])
		return self.mlp(torch.cat([h[edge_idx[0,:],:],h[edge_idx[1,:],:]],dim=1)).squeeze(1) # (E)