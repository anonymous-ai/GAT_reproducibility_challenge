
# Some code is borrowed from experiment_citeseer/run.py

import torch
import torch_geometric
import os
import sys
import argparse
import torch.nn.functional as F
import torch_geometric.datasets as datasets
import typing
import tqdm
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
parser.add_argument('--dp_rate', type=float, default=0.6,
                    help='dropout ratio')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='alpha value for leacky Relu')

# training params
parser.add_argument('--repeat', type=int, default=100,
                    help='Number of repetetion of experiment')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Training epochs')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--l2_weight', type=float, default=0.0005,
                    help='weight on l2 loss')
parser.add_argument('--patience', type=int, default=100,
                    help='number of patience for early stopping')

# data params
parser.add_argument('--split', type=str, default='public')
parser.add_argument('--num_train_per_class', type=int, default=20,
                    help='number of training data per class')
parser.add_argument('--num_val', type=int, default=500,
                    help='number of validation data')
parser.add_argument('--num_test', type=int, default=1000,
                    help='number of training test')

# path
parser.add_argument('--data_root_dir', type=str, default='../data/',
                    help='directory path to save data')

args = parser.parse_args()

device = 'cpu' # if torch.cuda.is_available() else 'cpu'



def run_experiment():
    dataset = datasets.Planetoid(
        root=args.data_root_dir,
        name='Cora',
        split=args.split,
        num_train_per_class=args.num_train_per_class,
        num_val=args.num_val,
        num_test=args.num_test,
        transform=torch_geometric.transforms.GCNNorm())

    args.nclasses = dataset.num_classes
    args.d_input = dataset.num_features

    model = NodeModel(args)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)


    data = dataset.data.to(device)

    best_loss_val = float('inf')
    best_acc_val = 0.0
    reported_test_acc = 0.0
    patience = 0
    for epoch in range(args.epochs):
        print('Epoch {}, Reported test acc {:.3f}, Patience {}'.format(epoch, reported_test_acc, patience))
        # train
        model.train()
        logits = model(data.x,data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        acc = (logits.argmax(1)==data.y)[data.train_mask].sum().float() / data.train_mask.sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        print('>>> Train: Loss {:.3f}, Acc {:.3f}'.format(loss,acc))


        # validation and test
        model.eval()
        logits = model(data.x,data.edge_index)

        loss_val = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask])
        acc_val = (logits.argmax(1)==data.y)[data.val_mask].sum().float() / data.val_mask.sum()
        print('>>> Val: Loss {:.3f}, Acc {:.3f}'.format(loss_val,acc_val))


        loss_test = F.cross_entropy(logits[data.test_mask], data.y[data.test_mask])
        acc_test = (logits.argmax(1)==data.y)[data.test_mask].sum().float() / data.test_mask.sum()
        print('>>> Test: Loss {:.3f}, Acc {:.3f}'.format(loss_test,acc_test))

        print()

        
        if loss_val.item() <= best_loss_val or acc_val.item() >= best_acc_val:
            if loss_val.item() <= best_loss_val and acc_val.item() >= best_acc_val:
                reported_test_acc = acc_test.item()
            best_loss_val = min(best_loss_val,loss_val.item())
            best_acc_val = max(best_acc_val,acc_val.item())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    
    return reported_test_acc


if __name__=='__main__':
    test_acc = []
    for _ in range(args.repeat):
        test_acc.append(run_experiment())
    mean_acc = np.mean(test_acc)
    std_acc = np.std(test_acc)

    print('Mean Acc: {:.3f}, Std Acc: {:.3f}'.format(mean_acc,std_acc))
