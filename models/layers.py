import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGATLayer(nn.Module):
    '''sparse gradient, multi-head'''

    def __init__(self, d_input, d_output, n_heads, att_dp_rate, alpha, aggregate):
        super(MultiHeadGATLayer, self).__init__()

        self.W = nn.Parameter(torch.empty(n_heads, d_input, d_output))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * d_output, 1))

        self.n_heads = n_heads
        self.aggregate = aggregate
        self.att_dp_rate = att_dp_rate
        self.LeakyReLU = nn.LeakyReLU(alpha)

    def forward(self, h, edge_idx):
        # h~(n_node,d_input), edge_idx~(2,E)
        n_node = h.size(0)

        Wh = torch.matmul(h, self.W)  # (n_heads,n_node,d_output), h is broadcasted
        a_input = torch.cat([Wh[:, edge_idx[0, :], :], Wh[:, edge_idx[1, :], :]], dim=2)  # (n_heads,E,2*d_output)
        e = self.LeakyReLU(torch.matmul(a_input, self.a).squeeze(2))  # (n_heads,E)

        h_output = []
        # torch.sparse.tensor support 3D tensors, but torch.sparse.mm() only support 2D matrices
        for i in range(self.n_heads):
            att = torch.sparse.softmax(torch.sparse.FloatTensor(edge_idx, e[i], [n_node, n_node]),
                                       dim=1)  # (n_node,n_node) sparse
            # torch.sparse.FloatTensor will be requires_grad and on cuda if e is requires_grad and on cuda
            # the input and output of torch.sparse.softmax() are all sparse tensors, non specified elements in input are interpreted as -inf, not 0.

            if self.att_dp_rate:  # dropout on "the normalized attention coefficients"
                att = torch.sparse.FloatTensor(edge_idx, F.dropout(att.coalesce().values(), self.att_dp_rate,
                                                                   training=self.training), [n_node, n_node])

            h_prime = torch.sparse.mm(att, Wh[i])  # (n_node,d_output)
            # torch.sparse.mm: sparse*sparse->sparse; sparse*dense->dense
            # so h_prime is dense
            h_output.append(h_prime)

        if self.aggregate == 'concat':
            return torch.cat(h_output, dim=1)  # (n_node,n_heads*d_output), concat
        else:
            return sum(h_output) / self.n_heads  # (n_node,d_output), mean


class MultiHeadDotAttLayer(nn.Module):
    '''scaled dot-product attention'''

    def __init__(self, d_input, d_output, n_heads, att_dp_rate, alpha, aggregate):
        super(MultiHeadDotAttLayer, self).__init__()

        self.d_k = d_output
        d_v = d_output
        self.WK = nn.Parameter(torch.empty(n_heads, d_input, self.d_k))
        self.WQ = nn.Parameter(torch.empty(n_heads, d_input, self.d_k))
        self.WV = nn.Parameter(torch.empty(n_heads, d_input, d_v))

        self.n_heads = n_heads
        self.aggregate = aggregate
        self.att_dp_rate = att_dp_rate

    # self.LeakyReLU = nn.LeakyReLU(alpha)

    def forward(self, h, edge_idx):
        # h~(n_node,d_input), edge_idx~(2,E)
        n_node = h.size(0)

        K = torch.matmul(h, self.WK)  # (n_heads,n_node,d_k), h is broadcasted
        Q = torch.matmul(h, self.WQ)  # (n_heads,n_node,d_k), h is broadcasted
        V = torch.matmul(h, self.WV)  # (n_heads,n_node,d_v), h is broadcasted
        # OOM
        e = torch.sum(Q[:, edge_idx[0, :], :] * K[:, edge_idx[1, :], :], dim=2, keepdim=False) / math.sqrt(
            self.d_k)  # (n_heads,E)

        h_output = []
        # torch.sparse.tensor support 3D tensors, but torch.sparse.mm() only support 2D matrices
        for i in range(self.n_heads):

            att = torch.sparse.softmax(torch.sparse.FloatTensor(edge_idx, e[i], [n_node, n_node]),
                                       dim=1)  # (n_node,n_node) sparse
            # torch.sparse.FloatTensor will be requires_grad and on cuda if e is requires_grad and on cuda
            # the input and output of torch.sparse.softmax() are all sparse tensors, non specified elements in input are interpreted as -inf, not 0.

            if self.att_dp_rate:  # dropout on "the normalized attention coefficients"
                att = torch.sparse.FloatTensor(edge_idx, F.dropout(att.coalesce().values(), self.att_dp_rate,
                                                                   training=self.training), [n_node, n_node])

            h_prime = torch.sparse.mm(att, V[i])  # (n_node,d_v)
            # torch.sparse.mm: sparse*sparse->sparse; sparse*dense->dense
            # so h_prime is dense
            h_output.append(h_prime)

        if self.aggregate == 'concat':
            return torch.cat(h_output, dim=1)  # (n_node,n_heads*d_output), concat
        else:
            return sum(h_output) / self.n_heads  # (n_node,d_output), mean


class ConstMultiHeadGATLayer(nn.Module):

    def __init__(self, d_input, d_output, n_heads, att_dp_rate, alpha, aggregate):
        super(ConstMultiHeadGATLayer, self).__init__()

        self.W = nn.Parameter(torch.empty(n_heads, d_input, d_output))
        # self.a = nn.Parameter(torch.empty(n_heads,2*d_output,1))

        self.n_heads = n_heads
        self.aggregate = aggregate
        self.att_dp_rate = att_dp_rate

    # self.LeakyReLU = nn.LeakyReLU(alpha)

    def forward(self, h, edge_idx):
        # h~(n_node,d_input), edge_idx~(2,E)
        n_node = h.size(0)

        Wh = torch.matmul(h, self.W)  # (n_heads,n_node,d_output), h is broadcasted

        e = torch.zeros(self.n_heads, edge_idx.size(1), device=h.device.type)  # (n_heads,E)

        h_output = []
        # torch.sparse.tensor support 3D tensors, but torch.sparse.mm() only support 2D matrices
        for i in range(self.n_heads):
            att = torch.sparse.softmax(torch.sparse.FloatTensor(edge_idx, e[i], [n_node, n_node]),
                                       dim=1)  # (n_node,n_node) sparse
            # torch.sparse.FloatTensor will be requires_grad and on cuda if e is requires_grad and on cuda
            # the input and output of torch.sparse.softmax() are all sparse tensors, non specified elements in input are interpreted as -inf, not 0.

            if self.att_dp_rate:  # dropout on "the normalized attention coefficients"
                att = torch.sparse.FloatTensor(edge_idx, F.dropout(att.coalesce().values(), self.att_dp_rate,
                                                                   training=self.training), [n_node, n_node])

            h_prime = torch.sparse.mm(att, Wh[i])  # (n_node,d_output)
            # torch.sparse.mm: sparse*sparse->sparse; sparse*dense->dense
            # so h_prime is dense
            h_output.append(h_prime)

        if self.aggregate == 'concat':
            return torch.cat(h_output, dim=1)  # (n_node,n_heads*d_output), concat
        else:
            return sum(h_output) / self.n_heads  # (n_node,d_output), mean
