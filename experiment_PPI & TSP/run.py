import os

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI, GNNBenchmarkDataset
import torch_geometric.transforms as T

from utils import get_args, logging, train_F1, train_loss, dev_F1, dev_loss
# from data import
from gat import PPIGAT, TSPGAT
from train import train, test



args=get_args()

# random.seed(args.seed)
# np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)
	# torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset=='PPI':
	trainset = PPI(root='PPI/',split='train',transform=T.AddSelfLoops())
	devset = PPI(root='PPI/',split='val',transform=T.AddSelfLoops())
	testset = PPI(root='PPI/',split='test',transform=T.AddSelfLoops())

	# trainset[0].contains_self_loops() # False
	# trainset[0].is_undirected() # True

	model = PPIGAT(args.att,d_input=50,d_hidden=(256,256,121),nheads=(4,4,6),dp_rate=args.dp,alpha=0.2,skip=args.residual)


elif args.dataset=='TSP':
	trainset = GNNBenchmarkDataset(root='TSP/',name='TSP',split='train',transform=None)
	devset = GNNBenchmarkDataset(root='TSP/',name='TSP',split='val',transform=None)
	testset = GNNBenchmarkDataset(root='TSP/',name='TSP',split='test',transform=None)

	# trainset[0].contains_self_loops() # False, T.AddSelfLoops() needs edge_attr
	# trainset[0].is_undirected() # False, T.ToUndirected() needs adj_t

	model = TSPGAT(args.att,nblocks=args.nblocks,d_input=2,d_hidden=(32,32,32),nheads=(4,4,4),dp_rate=args.dp,alpha=0.2,skip=args.residual)


trainloader = DataLoader(trainset,batch_size=args.bsz,shuffle=True)
devloader = DataLoader(devset,batch_size=args.bsz,shuffle=False)
testloader = DataLoader(testset,batch_size=args.bsz,shuffle=False)

if args.load_model=='':
	model.param_init()
else:
	model.load_state_dict(torch.load(args.load_model))
model.to(device)

if args.mode=='train':
	optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=(args.mom,0.999),weight_decay=args.wd)

	max_dev_F1 = 0
	min_dev_loss = 1e6
	save_epoch = -1
	save_dev_F1 = 0
	save_dev_loss = 0
	patience_epoch = 0
	for epoch in range(args.epoch):
		train(trainloader,devloader,model,optimizer,epoch,args.name)
		if dev_F1[-1]>max_dev_F1 or dev_loss[-1]<min_dev_loss:
			if dev_F1[-1]>max_dev_F1 and dev_loss[-1]<min_dev_loss:
				save_dev_F1 = dev_F1[-1]
				save_dev_loss = dev_loss[-1]
				torch.save(model.state_dict(),os.path.join('Log',args.name+'.dict'))
				save_epoch = epoch
			max_dev_F1 = max(max_dev_F1,dev_F1[-1])
			min_dev_loss = min(min_dev_loss,dev_loss[-1])
			patience_epoch = 0
		else:
			patience_epoch += 1
			if patience_epoch==args.patience:
				logging('Early Stop! max_dev_F1={:.4f}, min_dev_loss={:.6f}'.format(max_dev_F1,min_dev_loss),args.name)
				logging('Early Stop! save_epoch={:d}, save_dev_F1={:.4f}, save_dev_loss={:.6f}'.format(save_epoch,save_dev_F1,save_dev_loss),args.name)
				break

	torch.save((train_F1,dev_F1,train_loss,dev_loss),os.path.join('Log','train_'+args.name+'.list'))

elif args.mode=='test':
	test(testloader,model,args.name,is_dev=False)
else:
	print('Invalid arg: --mode')