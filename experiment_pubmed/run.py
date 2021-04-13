# Author: Zehua Cheng
# Mail: zehua.cheng@cs.ox.ac.uk

# Code borrowed from MJ's experiment_citeseer/run.py
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import typing

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.datasets as datasets
import tqdm
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.transforms import GCNNorm

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
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='hidden dimension of each layer')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='dropout ratio')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='alpha value for leacky Relu')
    parser.add_argument('--layer_type', type=str, default='GAT', choices=['GAT', 'GCN', 'DotGAT', 'ConstGAT'],
                        help='type of layer')
    # training params
    parser.add_argument('--dataset', type=str, default='CiteSeer', choices=['CiteSeer', 'CS', 'PubMed'],
                        help='Name of dataset for training')
    parser.add_argument('--repeat', type=int, default=100,
                        help='Number of repetetion of experiment')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--l2_weight', type=float, default=0.0005,
                        help='weight on l2 loss')
    parser.add_argument('--patience', type=int, default=10,
                        help='number of patience for early stopping')

    # data params
    parser.add_argument('--num_train_per_class', type=int, default=20,
                        help='number of training data per class')
    parser.add_argument('--num_val', type=int, default=500,
                        help='number of validation data')
    parser.add_argument('--num_test', type=int, default=1000,
                        help='number of training test')

    # path
    parser.add_argument('--data_root_dir', type=str, default='./data',
                        help='directory path to save data')
    return parser.parse_args()


class PytorchGeoNet(torch.nn.Module):
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

        assert layer_type in ['GAT', 'GCN']

        if layer_type == 'GAT':
            self.layer1 = GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=n_heads,
                dropout=dropout,
                concat=True
            )

            self.layer2 = GATConv(
                in_channels=hidden_dim * n_heads,
                out_channels=num_class,
                heads=1,
                dropout=dropout,
                concat=True
            )
        else:
            self.layer1 = GCNConv(
                in_channels=input_dim,
                out_channels=hidden_dim * n_heads,
            )

            self.layer2 = GCNConv(
                in_channels=hidden_dim * n_heads,
                out_channels=num_class
            )

        self.LeakyRelu = torch.nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.layer_type = layer_type

    def forward(self, x, edge_index):
        if self.layer1 == 'GAT':
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layer1(x, edge_index)
            x = self.LeakyRelu(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layer2(x, edge_index)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layer1(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.LeakyRelu(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layer2(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def param_init(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()


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
        self.LeakyRelu = torch.nn.LeakyReLU(alpha)

    def param_init(self):
        if self.layer_type == 'GAT':
            torch.nn.init.xavier_normal_(self.att_layer1.W.data)
            torch.nn.init.xavier_normal_(self.att_layer1.a.data)

            torch.nn.init.xavier_normal_(self.att_layer2.W.data)
            torch.nn.init.xavier_normal_(self.att_layer2.a.data)
        elif self.layer_type == 'DotGAT':
            torch.nn.init.xavier_normal_(self.att_layer1.WK.data)
            torch.nn.init.xavier_normal_(self.att_layer1.WQ.data)
            torch.nn.init.xavier_normal_(self.att_layer1.WV.data)

            torch.nn.init.xavier_normal_(self.att_layer2.WK.data)
            torch.nn.init.xavier_normal_(self.att_layer2.WQ.data)
            torch.nn.init.xavier_normal_(self.att_layer2.WV.data)
        else:
            torch.nn.init.xavier_normal_(self.att_layer1.W.data)

            torch.nn.init.xavier_normal_(self.att_layer2.W.data)

    def forward(self, x, edge_idx):
        # x~(n_node,d_input), edge_idx~(2,E)
        x = F.dropout(x, self.dp_rate, training=self.training)
        h1 = self.att_layer1(x, edge_idx)  # (n_node,nheads*d_hidden)
        h1 = self.LeakyRelu(h1)

        h11 = F.dropout(h1, self.dp_rate, training=self.training)
        h2 = self.att_layer2(h11, edge_idx)  # (n_node,nclass)
        return F.log_softmax(h2, dim=1)


def load_dataset(
        args: argparse.Namespace
) -> typing.Tuple[torch_geometric.data.Dataset, argparse.Namespace]:
    """

    Args:
        args: arguments

    Returns: Tuple of dataset and related hparams

    """
    if args.dataset in ['CiteSeer', 'PubMed']:
        dataset = datasets.Planetoid(
            root=args.data_root_dir,
            name=args.dataset,
            num_train_per_class=args.num_train_per_class,
            num_val=args.num_val,
            num_test=args.num_test
        )
        data = dataset[0]
    elif args.dataset in ['CS', 'Physics']:
        dataset = datasets.Coauthor(
            root=args.data_root_dir,
            name=args.dataset,
        )
        data = split_dataset(
            dataset=dataset,
            num_train_per_class=args.num_train_per_class,
            num_val=args.num_val,
            num_test=None
        )
    else:
        raise ValueError("Not proper dataset")

    # return dataset and updated args
    args.num_classes = dataset.num_classes
    args.input_dim = dataset.num_features
    return data


def split_dataset(dataset, num_train_per_class, num_val, num_test=None):
    data = dataset[0]

    n_nodes, y = data.x.size(0), data.y
    idx_all = [i for i in range(n_nodes)]

    unique_y = list(set(y.tolist()))
    y_ = y.numpy()

    # extract train_idx
    train_idx = []
    for uy in unique_y:
        idx = np.where(y_ == uy)[0]
        train_idx += idx[np.random.choice(len(idx), size=num_train_per_class, replace=False)].tolist()

    # remove train_idx from all index
    for i in train_idx:
        idx_all.remove(i)

    val_idx = np.array(idx_all)[np.random.choice(len(idx_all), size=num_val, replace=False)].tolist()
    # remove val_idx from all index
    for i in val_idx:
        idx_all.remove(i)

    if num_test:
        test_idx = np.array(idx_all)[np.random.choice(len(idx_all), size=num_test, replace=False)].tolist()
        # remove val_idx from all index
    else:
        test_idx = idx_all

    train_mask = torch.BoolTensor([True if i in train_idx else False for i in range(n_nodes)])
    val_mask = torch.BoolTensor([True if i in val_idx else False for i in range(n_nodes)])
    test_mask = torch.BoolTensor([True if i in test_idx else False for i in range(n_nodes)])

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


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
        model = PytorchGeoNet(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            num_class=args.num_classes,
            dropout=args.dropout,
            alpha=args.alpha,
            layer_type=args.layer_type
        )
        model.param_init()
    return model


def compute_l2_loss(
        model: torch.nn.Module,
        device
) -> torch.Tensor:
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_reg


def train(
        args: argparse.Namespace,
        device
) -> torch.nn.Module:
    args = return_args()
    data = load_dataset(args)
    data = data.to(device)

    # Set components
    model = load_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # preprocess data
    prepro = GCNNorm()
    data = prepro(data)

    val_loss_list, val_acc_list = [], []
    min_loss, max_acc = 1e5, 0
    patience = 0
    for epoch in range(args.epochs):
        # train
        model.train()
        optimizer.zero_grad()
        x, edge_index = data.x, data.edge_index

        # this part would be deprecated later
        if args.use_ours:
            logits = model(x, edge_index)
        else:
            logits = model(x, edge_index)

        # calculate loss
        ce_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        if args.use_l2_loss:
            l2_loss = compute_l2_loss(model, device)
            loss = ce_loss + args.l2_weight * l2_loss
        else:
            loss = ce_loss
        loss.backward()
        optimizer.step()

        # validate
        val_loss, acc = evaluate(args, data, model, data_type='val')
        val_loss = val_loss.item()
        print(f"Training loss: {round(loss.item(), 2)} | "
              f"Validation loss: {round(val_loss, 2)} | "
              f"Validation Acc: {round(acc, 3) * 100}")

        val_loss_list.append(val_loss)
        val_acc_list.append(acc)

        if val_loss <= min_loss or acc >= max_acc:
            if val_loss <= min_loss and acc >= max_acc:
                # save checkpoint (save to data folder)
                save_name = f'{args.layer_type}-{args.dataset}-num_per_class{args.num_train_per_class}-checkpoint.ckpt'
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
    load_name = f'{args.layer_type}-{args.dataset}-num_per_class{args.num_train_per_class}-checkpoint.ckpt'
    filename = os.path.join(args.data_root_dir, load_name)
    model.load_state_dict(torch.load(filename))
    return model


def evaluate(
        args: argparse.Namespace,
        data: torch_geometric.data.Data,
        model: torch.nn.Module,
        data_type: typing.Text = 'val'
) -> typing.Tuple[torch.Tensor, float]:
    assert data_type in ['val', 'test']
    if data_type == 'val':
        mask = data.val_mask
    else:
        mask = data.test_mask

    model.eval()
    x, edge_index = data.x, data.edge_index

    # this part would be deprecated later
    if args.use_ours:
        logits = model(x, edge_index)
    else:
        logits = model(x, edge_index)

    # compute loss
    ce_loss = F.nll_loss(logits[mask], data.y[mask])
    if args.use_l2_loss:
        l2_loss = compute_l2_loss(model, x.device)
        loss = ce_loss + args.l2_weight * l2_loss
    else:
        loss = ce_loss

    # accuracy
    pred = logits.max(dim=-1)[1]
    correct = float(pred[mask].eq(data.y[mask]).sum().item())
    acc = correct / mask.sum().item()
    return loss, acc


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_dataset(args)
    data = data.to(device)

    test_loss_list, test_acc_list = [], []
    for _ in tqdm.tqdm(range(args.repeat)):
        # Train model
        model = train(args, device=device)
        # Evaluate
        test_loss, test_acc = evaluate(args, data, model, data_type='test')
        print(f"Test loss: {test_loss} | Test Acc: {test_acc * 100}")
        test_loss_list.append(test_loss.item())
        test_acc_list.append(test_acc)

    mean_acc = np.mean(test_acc_list)
    std_acc = np.std(test_acc_list)

    mean_loss = np.mean(test_loss_list)
    std_loss = np.std(test_loss_list)

    outp = f"Exp repeat: {args.repeat} \nMean test Acc: {mean_acc} | Std test Acc: {std_acc} \n" \
           f"Mean test loss: {mean_loss} | Std test loss: {std_loss}"

    file_name = f'report_{args.layer_type}_{args.dataset}_num_per_class{args.num_train_per_class}'
    if args.use_ours:
        file_name += "_ours"
    file_name += '.txt'

    os.makedirs('report', exist_ok=True)
    with open(os.path.join('report', file_name), 'w', encoding='utf-8') as saveFile:
        saveFile.write(outp)


if __name__ == '__main__':
    args = return_args()
    main(args)
