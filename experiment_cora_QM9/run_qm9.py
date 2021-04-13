
# Some code is borrowed from experiment_citeseer/run.py

import torch
import torch_geometric
import os
import sys
import argparse
import torch.nn.functional as F
import torch_geometric.datasets as datasets
from torch_geometric.data import DataLoader
import typing
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.layers import *
from experiment_cora_QM9.networks import *


parser = argparse.ArgumentParser()

# model params
parser.add_argument('--network', type=str, default='GAT',
                    help='network type: GAT, GAT_dot, GAT_const, GCN')
parser.add_argument('--d_hidden', type=int, default=8,
                    help='hidden dimension of each layer')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--dp_rate', type=float, default=0.0,
                    help='dropout ratio')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='alpha value for leacky Relu')

# training params
parser.add_argument('--repeat', type=int, default=10,
                    help='Number of repetetion of experiment')
parser.add_argument('--epochs', type=int, default=300,
                    help='Training epochs')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--l2_weight', type=float, default=0.0005,
                    help='weight on l2 loss')
parser.add_argument('--patience', type=int, default=100,
                    help='number of patience for early stopping')

# data params
parser.add_argument('--task', type=int, default=0,
                    help='task, from 0 to 18')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('--num_val', type=int, default=500,
                    help='number of validation data')
parser.add_argument('--num_test', type=int, default=1000,
                    help='number of training test')

# path
parser.add_argument('--data_root_dir', type=str, default='../data/',
                    help='directory path to save data')

args = parser.parse_args()

device = 'cpu' # if torch.cuda.is_available() else 'cpu'

def train(dataloader,model,optim):
    model.train()
    epoch_loss = 0.0
    for k in tqdm(dataloader,ncols=75,leave=False):
        k.to(device)
        logits = model(k.x,k.edge_index,k.batch)
        loss = F.mse_loss(logits, k.y[:,args.task:args.task+1], reduction='none')

        optim.zero_grad()
        loss.sum(1).mean().backward()
        optim.step()

        epoch_loss = loss.mean(0).to('cpu').data.numpy()
    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss

def test(dataloader,model):
    model.eval()
    epoch_loss = 0.0
    for k in tqdm(dataloader,ncols=75,leave=False):
        k.to(device)
        logits = model(k.x,k.edge_index,k.batch)
        loss = F.mse_loss(logits, k.y[:,args.task:args.task+1], reduction='none')

        epoch_loss = loss.mean(0).to('cpu').data.numpy()
    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss

def run_experiment():
    dataset = datasets.QM9(
        root=args.data_root_dir)

    np.random.seed(0)
    indices = np.random.choice(len(dataset),2500)
    train_indices = list(indices[:1000])
    val_indices = list(indices[1000:1500])
    test_indices = list(indices[1500:2500])

    trainloader = DataLoader(dataset[train_indices],batch_size=args.batch_size,shuffle=True)
    valloader = DataLoader(dataset[val_indices],batch_size=args.batch_size,shuffle=False)
    testloader = DataLoader(dataset[test_indices],batch_size=args.batch_size,shuffle=False)


    args.nclasses = 1 #dataset.num_classes
    args.d_input = dataset.num_features

    model = GraphModel(args)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)


    best_loss_val = float('inf')
    best_acc_val = 0.0
    reported_test_loss = 0.0
    patience = 0
    for epoch in range(args.epochs):
        print('Epoch {}, Patience {}, Report test loss'.format(epoch, patience), reported_test_loss)
        # train
        loss = train(trainloader,model,optim)
        print('>>> Train: Loss ', loss)

        loss_val = test(valloader,model)
        loss_test = test(testloader,model)
        print('>>> Val: Loss ', loss_val)
        print('>>> Test: Loss ', loss_test)
        
        if loss_val.sum().item() <= best_loss_val:
            reported_test_loss = loss_test
            best_loss_val = min(best_loss_val,loss_val.sum().item())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    
    return reported_test_loss


if __name__=='__main__':
    test_loss = []
    for _ in range(args.repeat):
        test_loss.append(run_experiment())
    print(test_loss)
    mean_loss = np.mean(test_loss)
    std_loss = np.std(test_loss)

    print('Mean loss: {:.4f}, Std loss: {:.4f}'.format(mean_loss,std_loss))
