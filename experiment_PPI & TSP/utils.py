
from argparse import ArgumentParser
import time
import os.path

train_F1 = []
train_loss = []
dev_F1 = []
dev_loss = []

def get_args():
	# CUDA_VISIBLE_DEVICES=0,1,2,3 to select GPU

	parser=ArgumentParser(description='GAT')
	
	# data arguments
	parser.add_argument('--dataset',type=str,default='PPI') # {'PPI','TSP'}

	# model arguments
	parser.add_argument('--load-model',type=str,default='') # pre-trained model path 
	parser.add_argument('--att',type=str,default='gat') # {'gat','transformer','const'}
	parser.add_argument('--residual',action='store_true',default=False) # whether to adopt skip connections
	parser.add_argument('--nblocks',type=int,default=100) # num of blocks for TSPGAT

	# optimization arguments
	parser.add_argument('--bsz',type=int,default=32) # 2
	parser.add_argument('--epoch',type=int,default=1000) # max training epochs
	parser.add_argument('--patience',type=int,default=100) # eraly stopping patience
	parser.add_argument('--lr',type=float,default=0.03) # initial learning rate
	parser.add_argument('--mom',type=float,default=0.9)
	parser.add_argument('--wd',type=float,default=0)
	parser.add_argument('--dp',type=float,default=0.1) # dropout rate {0.0, 0.2, 0.4, 0.5}
	
	# general arguments
	parser.add_argument('--seed',type=int,default=100)
	parser.add_argument('--mode',type=str,default='train')
	parser.add_argument('--name',type=str,default='') # used in saved filenames to distinct different configurations
	
	args=parser.parse_args()
	return args

def show_time(seconds,show_hour=False):
	m=seconds//60 # m is an integer
	s=seconds%60   # s is a real number
	if show_hour:
		h=m//60  # h is an integer
		m=m%60    # m is an integer
		return '%02d:%02d:%05.2f'%(h,m,s)
	else:
		return '%02d:%05.2f'%(m,s)

def logging(s:str,model_name='',log_=True,print_=True):
	if print_:
		print(s)
	if log_:
		with open(os.path.join('Log','log_'+model_name+'.txt'), 'a+') as f_log:
			f_log.write(s + '\n')