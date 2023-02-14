import torch as th
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import math

from modules.layers import *
from modules.functions import *
from modules.embedding import *
from modules.viz import att_animation, get_attention_map
from optims import NoamOpt
from loss import LabelSmoothing, SimpleLossCompute
from dataset import get_dataset, GraphPool

VIZ_IDX = 3

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
from torch.nn import LayerNorm

device = ['cuda' if th.cuda.is_available() else 'cpu']
dataset = get_dataset('multi30k')

class StudentNet(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
           num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
           input_dim: dimensionality of input features
           hidden_dim: dimensionality of hidden units at ALL layers
           output_dim: number of classes for prediction
        '''

        super(StudentNet, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = th.nn.ModuleList()
            self.batch_norms = th.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

def run_epoch(data_iter, model, is_train=True):
    for i, g in enumerate(data_iter):
        with th.set_grad_enabled(is_train):
            output = model(g).to(device[0])
            loss = th.nn.NLLLoss()
            print('loss: {}'.loss)

print("Default Device : {}".format(th.Tensor([4, 5, 6]).device))
print("Total GPU Count :{}".format(th.cuda.device_count()))
print("Total CPU Count :{}".format(th.cuda.os.cpu_count()))

def student_main():
    batch_size = 128

    # load dataset
    dataset = get_dataset("multi30k")
    V = dataset.vocab_size
    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    dim_model = 128

    graph_pool = GraphPool()

    # Display dataset
    data_iter = dataset(graph_pool, mode='train', batch_size=1, device=device[0])
    for graph in data_iter:
        print(graph.nids['enc'])  # encoder node ids
        print(graph.nids['dec'])  # decoder node ids
        print(graph.eids['ee'])  # encoder-encoder edge ids
        print(graph.eids['ed'])  # encoder-decoder edge ids
        print(graph.eids['dd'])  # decoder-decoder edge ids
        print(graph.src[0])  # Input word index list
        print(graph.src[1])  # Input positions
        print(graph.tgt[0])  # Output word index list
        print(graph.tgt[1])  # Ouptut positions
        break

    # Create model
    model = StudentNet(num_layers=8, input_dim=128, hidden_dim=64, output_dim=128).to(device[0])

    optimizer = th.optim.Adam( model.parameters(), lr=0.01, weight_decay=0.0005)
    loss = th.nn.NLLLoss()
    student_history = []

    for epoch in range(100):
        train_iter = dataset(graph_pool, mode='train', batch_size=batch_size, device=device[0])
        valid_iter = dataset(graph_pool, mode='valid', batch_size=batch_size, device=device[0])
        print('Epoch: {} Training...'.format(epoch))
        model.train(True).to(device[0])
        run_epoch(train_iter, model, is_train=True)
        print('Epoch: {} Evaluating...'.format(epoch))
        model.att_weight_map = None
        model.eval().to(device[0])
        run_epoch(valid_iter, model, is_train=False)

    th.save(model.state_dict(), "student.pt")
    return model, student_history


student_model, student_history = student_main()

from torchinfo import summary
if __name__== "__main__" :
    np.random.seed(1111)
    student_main()
    summary(StudentNet())