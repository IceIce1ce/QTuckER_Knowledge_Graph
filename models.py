import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import empty, matmul, tensor
import torch
from torch.cuda import empty_cache
from torch.nn import Parameter, Module
from torch.nn.functional import normalize
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import math
import numpy as np
from gnn_layers import *

class QTuckER(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1, **kwargs):
        super(QTuckER, self).__init__()
        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (hid_dim, hid_dim, hid_dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.lst_qgnn = torch.nn.ModuleList()
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_qgnn.append(QGNN_layer(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_qgnn.append(QGNN_layer(hid_dim, hid_dim, act=torch.tanh))
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.bn0 = torch.nn.BatchNorm1d(hid_dim)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        for _layer in range(self.num_layers):
            X = self.lst_qgnn[_layer](X, self.adj)
        e1 = X[e1_idx] 
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = X[r_idx + self.n_entities]
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = F.leaky_relu(x)
        x = torch.mm(x, self.input_dropout(X[:self.n_entities]).t()) #x = torch.mm(x, X[:self.n_entities].t())
        pred = torch.sigmoid(x)
        return pred

class OTuckER(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1, **kwargs):
        super(OTuckER, self).__init__()
        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (hid_dim, hid_dim, hid_dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.lst_qgnn = torch.nn.ModuleList()
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_qgnn.append(OGNN_layer(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_qgnn.append(OGNN_layer(hid_dim, hid_dim, act=torch.tanh))
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.bn0 = torch.nn.BatchNorm1d(hid_dim)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        for _layer in range(self.num_layers):
            X = self.lst_qgnn[_layer](X, self.adj)
        e1 = X[e1_idx] 
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = X[r_idx + self.n_entities]
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = F.leaky_relu(x)
        x = torch.mm(x, self.input_dropout(X[:self.n_entities]).t()) #x = torch.mm(x, X[:self.n_entities].t())
        pred = torch.sigmoid(x)
        return pred