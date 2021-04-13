import time

import torch
import torch.nn.functional as F

from utils import logging, show_time
from utils import train_F1, train_loss, dev_F1, dev_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class micro_F1(object):
	def __init__(self):
		super(micro_F1,self).__init__()
		self.TP = 0
		self.TN = 0
		self.FP = 0
		self.FN = 0
	
	def add(self,input_preds,input_targets):
		preds = input_preds.int()
		targets = input_targets.int()
		self.TP += (preds*targets).sum().item()
		self.TN += ((1-preds)*(1-targets)).sum().item()
		self.FP += (preds*(1-targets)).sum().item()
		self.FN += ((1-preds)*targets).sum().item()

	def get(self):
		precision = 0 if self.TP==0 else self.TP/(self.TP+self.FP)
		recall = 0 if self.TP==0 else self.TP/(self.TP+self.FN)
		f1 = 0 if (precision+recall)==0 else (2*precision*recall)/(precision+recall)
		return f1

	def reset(self):
		pass


def train(batchloader,devloader,model,optimizer,cur_epoch,name:str):
	
	# train through the whole dataset for one time
	f1 = micro_F1()
	epoch_loss = 0
	total_n_nodes = 0

	# In each epoch, shuffle the whole dataset to get different mini-batch splits.
	# batchloader.reshuffle()

	model.train()
	optimizer.zero_grad()

	for batch_idx,batch in enumerate(batchloader):
		
		start = time.time()
		update_loss = torch.FloatTensor(1).zero_().to(device)
		
		x = batch.pos if batch.x is None else batch.x # (n_nodes,d_input)
		edge_index = batch.edge_index # (2,E)
		# b_info = batch.batch # (n_nodes)
		y = batch.y # (n_nodes,121) or (E)
		
		n_nodes = y.size(0) # for TSP, is n_edges

		scores = model(x.to(device),edge_index.to(device)) # PPI: (n_nodes,121), TSP: (E)

		update_loss = F.binary_cross_entropy_with_logits(scores,y.to(device).float(),reduction='mean')

		optimizer.zero_grad()
		update_loss.backward()

		grad_gat = torch.cat([param.grad.view(-1) for param in model.parameters()])
			
		grad_avg = grad_gat.sum().item()/len(grad_gat)

		optimizer.step()

		logging('{:05d}, loss(e-4):{:.6f}, '.format(batch_idx,update_loss.item()*1e4)+show_time(time.time()-start),name)
		logging('gradient(e-6) gat:{:.4f}'.format(grad_avg*1e6),name)
		

		preds = torch.sigmoid(scores).cpu()>=0.5
		f1.add(preds,y.cpu())
		epoch_loss += update_loss.cpu().item()*n_nodes
		total_n_nodes += n_nodes		

	F1 = f1.get()
	epoch_loss /= total_n_nodes 
	
	train_F1.append(F1)
	train_loss.append(epoch_loss)

	logging('epoch {:2d}'.format(cur_epoch),name)
	test(devloader,model,name,is_dev=True)
	logging('Train F1 '+str(train_F1),name)
	logging('Dev F1 '+str(dev_F1),name)
	logging('Train loss '+str(train_loss),name)
	logging('Dev loss '+str(dev_loss),name)
	logging('',name)


def test(batchloader,model,name:str,is_dev:bool):
	
	# train through the whole dataset for one time
	f1 = micro_F1()
	epoch_loss = 0
	total_n_nodes = 0

	# In each epoch, shuffle the whole dataset to get different mini-batch splits.
	# batchloader.reshuffle()

	model.eval()

	with torch.no_grad():
		for batch_idx,batch in enumerate(batchloader):
			
			start = time.time()
			update_loss = torch.FloatTensor(1).zero_().to(device)
			
			x = batch.pos if batch.x is None else batch.x # (n_nodes,d_input)
			edge_index = batch.edge_index # (2,E)
			# b_info = batch.batch # (n_nodes)
			y = batch.y # (n_nodes,121)
		
			n_nodes = y.size(0)

			scores = model(x.to(device),edge_index.to(device)) # (n_nodes,121)

			update_loss = F.binary_cross_entropy_with_logits(scores,y.to(device).float(),reduction='mean')
			

			preds = torch.sigmoid(scores).cpu()>=0.5
			f1.add(preds,y.cpu())
			epoch_loss += update_loss.cpu().item()*n_nodes
			total_n_nodes += n_nodes	

		F1 = f1.get()
		epoch_loss /= total_n_nodes 
	
	if is_dev:
		dev_F1.append(F1)
		dev_loss.append(epoch_loss)
	else:
		logging('Test F1: {:.8f}'.format(F1),name)
		logging('Test loss: {:.8f}'.format(epoch_loss),name)
