# -*- coding: utf-8 -*-

__version__ = "2021.1"
__date__ = "10 04 2021"
__status__ = "Development"

import sys
import torch
import typing
import os
import torch.nn.functional as F
import argparse
import numpy as np
import torch_geometric
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import models.layers as layers


def return_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # just for sanity check, would be deprecated later
    parser.add_argument('--use_ours', action='store_true',
                        help='use our model or GAT layer in pytorch')
    parser.add_argument('--use_l2_loss', action='store_true',
                        help='use l2 regularization or not')
    # model params
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='hidden dimension of each layer')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='dropout ratio')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='alpha value for leacky Relu')
    parser.add_argument('--layer_type', type=str, default='GAT', choices=['GAT', 'GCN', 'DotGAT', 'ConstGAT', 'GATTorch'],
                        help='type of layer')

    # training params
    parser.add_argument('--dataset', type=str, default='PROTEINS', choices=['PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY'],
                        help='Name of dataset for training')
    parser.add_argument('--repeat', type=int, default=100,
                        help='Number of repetetion of experiment')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--l2_weight', type=float, default=0.0005,
                        help='weight on l2 loss')
    parser.add_argument('--patience', type=int, default=10,
                        help='number of patience for early stopping')

    # data params
    parser.add_argument('--num_val', type=int, default=100,
                        help='number of validation data')
    parser.add_argument('--num_test', type=int, default=200,
                        help='number of validation data')

    # path
    parser.add_argument('--data_root_dir', type=str, default='../data/',
                        help='directory path to save data')
    return parser.parse_args()


class GATNet(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 num_class: int,
                 dropout: float,
                 alpha: float,
                 layer_type: str = 'GAT'
                 ) -> None:
        super().__init__()

        assert layer_type in ['GAT', 'DotGAT', 'ConstGAT']
        if layer_type == 'GAT':
            self.att_layer1 = layers.MultiHeadGATLayer(
                d_input=input_dim,
                d_output=hidden_dim,
                n_heads=n_heads,
                att_dp_rate=dropout,
                alpha=alpha,
                aggregate='concat'
            )

            self.att_layer2 = layers.MultiHeadGATLayer(
                d_input=hidden_dim * n_heads,
                d_output=num_class,
                n_heads=1,
                att_dp_rate=dropout,
                alpha=alpha,
                aggregate='concat'
            )
        elif layer_type == 'DotGAT':
            self.att_layer1 = layers.MultiHeadDotAttLayer(
                d_input=input_dim,
                d_output=hidden_dim,
                n_heads=n_heads,
                att_dp_rate=dropout,
                alpha=alpha,
                aggregate='concat'
            )

            self.att_layer2 = layers.MultiHeadDotAttLayer(
                d_input=hidden_dim * n_heads,
                d_output=num_class,
                n_heads=1,
                att_dp_rate=dropout,
                alpha=alpha,
                aggregate='concat'
            )
        else:
            self.att_layer1 = layers.ConstMultiHeadGATLayer(
                d_input=input_dim,
                d_output=hidden_dim,
                n_heads=n_heads,
                att_dp_rate=dropout,
                alpha=alpha,
                aggregate='concat'
            )

            self.att_layer2 = layers.ConstMultiHeadGATLayer(
                d_input=hidden_dim * n_heads,
                d_output=num_class,
                n_heads=1,
                att_dp_rate=dropout,
                alpha=alpha,
                aggregate='concat'
            )

        self.layer_type = layer_type
        self.dp_rate = dropout

    def param_init(self):
        INIT = 0.01

        if self.layer_type == 'GAT':
            self.att_layer1.W.data.uniform_(-INIT, INIT)
            self.att_layer1.a.data.uniform_(-INIT, INIT)

            self.att_layer2.W.data.uniform_(-INIT, INIT)
            self.att_layer2.a.data.uniform_(-INIT, INIT)

        elif self.layer_type == 'DotGAT':
            self.att_layer1.WK.data.uniform_(-INIT,INIT)
            self.att_layer1.WQ.data.uniform_(-INIT,INIT)
            self.att_layer1.WV.data.uniform_(-INIT,INIT)

            self.att_layer2.WK.data.uniform_(-INIT,INIT)
            self.att_layer2.WQ.data.uniform_(-INIT,INIT)
            self.att_layer2.WV.data.uniform_(-INIT,INIT)

        else:
            self.att_layer1.W.data.uniform_(-INIT,INIT)
            self.att_layer2.W.data.uniform_(-INIT,INIT)

    def forward(self, data):
        # x~(n_node,d_input), edge_idx~(2,E)
        x = F.dropout(data.x, self.dp_rate, training=self.training)
        h1 = self.att_layer1(x, data.edge_index)  # (n_node,nheads*d_hidden)
        h1 = F.elu(h1)

        h11 = F.dropout(h1, self.dp_rate, training=self.training)
        h2 = self.att_layer2(h11, data.edge_index)  # (n_node,nclass)

        # aggregate by mean
        x = global_mean_pool(h2, data.batch)
        return F.log_softmax(x, dim=-1)


class GANNet(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 num_class: int,
                 dropout: float,
                 alpha: float,
                 ) -> None:
        super().__init__()

        self.layer1 = GCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim * n_heads,
        )

        self.layer2 = GCNConv(
            in_channels=hidden_dim * n_heads,
            out_channels=num_class,
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.layer1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layer2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # aggregate by mean
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=-1)

    def param_init(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()


def load_dataset(
        args: argparse.Namespace
) -> typing.Tuple[torch_geometric.data.Dataset, torch_geometric.data.Dataset,
                  torch_geometric.data.Dataset, argparse.Namespace]:
    """

    Args:
        args: arguments

    Returns: Tuple of dataset and related hparams

    """
    dataset = TUDataset(
        root=args.data_root_dir,
        name=args.dataset,
        transform=torch_geometric.transforms.AddSelfLoops()
    )
    # return dataset and updated args
    args.num_classes = dataset.num_classes
    args.input_dim = dataset.num_features

    np.random.seed(1234)
    idx = np.array([i for i in range(len(dataset))])
    idx = np.random.permutation(idx)

    val_idx = idx[:args.num_val].tolist()
    test_idx = idx[args.num_val:args.num_val + args.num_test].tolist()
    tr_idx = idx[args.num_val + args.num_test:].tolist()

    return dataset[tr_idx], dataset[val_idx], dataset[test_idx], args


def load_model(
        args: argparse.Namespace,
) -> torch.nn.Module:
    if 'GAT' in args.layer_type:
        model = GATNet(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_class=args.num_classes,
            dropout=args.dropout,
            alpha=args.alpha,
            n_heads=args.n_heads,
            layer_type=args.layer_type
        )
        model.param_init()
    else:
        model = GANNet(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            num_class=args.num_classes,
            dropout=args.dropout,
            alpha=args.alpha,
        )
        model.param_init()
    return model


def train(
        args: argparse.Namespace,
        device
) -> torch.nn.Module:
    train, val,  test, args = load_dataset(args)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=True)
    # test_loader = DataLoader(train, batch_size=32, shuffle=True)

    # Set components
    model = load_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.l2_weight)

    val_loss_list, val_acc_list = [], []
    min_loss, max_acc = 1e5, 0
    patience = 0
    for epoch in range(args.epochs):
        # train
        model.train()
        optimizer.zero_grad()

        for data in train_loader:
            data.to(device)

            logits = model(data)
            y_true = data.y

            loss = F.nll_loss(logits, y_true)
            loss.backward()
            optimizer.step()

        # validate
        val_loss, acc = evaluate(val_loader, model, device=device)

        print(f"Validation loss: {round(val_loss, 2)} | "
              f"Validation Acc: {round(acc, 3) * 100}")

        val_loss_list.append(val_loss)
        val_acc_list.append(acc)

        if val_loss <= min_loss or acc >= max_acc:
            if val_loss <= min_loss and acc >= max_acc:
                # save checkpoint (save to data folder)
                save_name = f'{args.layer_type}-{args.dataset}-checkpoint.ckpt'
                filename = os.path.join(args.data_root_dir, save_name)
                torch.save(model.state_dict(), filename)

            min_loss = min(val_loss, min_loss)
            max_acc = max(acc, max_acc)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print("Early Stop!!!")
                break

    # load best model from checkpoint
    load_name = f'{args.layer_type}-{args.dataset}-checkpoint.ckpt'
    filename = os.path.join(args.data_root_dir, load_name)
    model.load_state_dict(torch.load(filename))
    return model


def evaluate(
        loader: torch_geometric.data.Data,
        model: torch.nn.Module,
        device
) -> typing.Tuple[float, float]:

    model.eval()

    loss = 0
    pred_list, true_list = [], []
    for data in loader:
        data.to(device)
        logits = model(data)
        y_true = data.y

        loss_ = F.nll_loss(logits, y_true)
        loss += loss_.item()

        # accuracy
        pred = logits.max(dim=-1)[1]
        pred_list += pred.tolist()
        true_list += y_true.tolist()

    acc = [1 if p == t else 0 for (p,t) in zip(pred_list, true_list)]
    acc = sum(acc) / len(acc)
    return loss / len(loader), acc


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test, args = load_dataset(args)
    test_loader = DataLoader(test, batch_size=32, shuffle=True)

    test_loss_list, test_acc_list = [], []
    for _ in tqdm(range(args.repeat)):
        # Train model
        model = train(args, device=device)
        # Evaluate
        test_loss, test_acc = evaluate(test_loader, model, device=device)
        print(f"Test loss: {test_loss} | Test Acc: {test_acc * 100}")
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    mean_acc = np.mean(test_acc_list)
    std_acc = np.std(test_acc_list)

    mean_loss = np.mean(test_loss_list)
    std_loss = np.std(test_loss_list)

    outp = f"Exp repeat: {args.repeat} \nMean test Acc: {mean_acc} | Std test Acc: {std_acc} \n" \
           f"Mean test loss: {mean_loss} | Std test loss: {std_loss}"

    file_name = f'report_{args.layer_type}_{args.dataset}'
    file_name += '.txt'

    os.makedirs('report', exist_ok=True)
    with open(os.path.join('report', file_name), 'w', encoding='utf-8') as saveFile:
        saveFile.write(outp)


if __name__ == '__main__':
    args = return_args()
    main(args)





