"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = 10
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device

        self.embeding = nn.Embedding(num_embeddings=input_dim, embedding_dim=10)

        self.g = Gate(self.input_dim, self.hidden_dim, torch.tanh)
        self.i = Gate(self.input_dim, self.hidden_dim, torch.sigmoid)
        self.f = Gate(self.input_dim, self.hidden_dim, torch.sigmoid)
        self.o = Gate(self.input_dim, self.hidden_dim, torch.sigmoid)

        self.Wph = nn.Parameter(torch.zeros(self.hidden_dim, self.num_classes))
        self.bp = nn.Parameter(torch.zeros(self.num_classes))

        for name, param in self.named_parameters():
            if "W" in name:
                torch.nn.init.kaiming_normal_(param.data, nonlinearity='linear')


    def forward(self, x):
        h_t = torch.zeros(x.shape[0], self.hidden_dim).to(self.device)
        c_t = torch.zeros(x.shape[0], self.hidden_dim).to(self.device)

        x = self.embeding(x.squeeze().long())
        for t in range(self.seq_length):
            x_t = x[:,t]

            g_t = self.g(x_t, h_t)
            i_t = self.i(x_t, h_t)
            f_t = self.f(x_t, h_t)
            o_t = self.o(x_t, h_t)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

            p_t = h_t @ self.Wph + self.bp
        return p_t

class Gate(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation):
        super(Gate, self).__init__()

        self.activation = activation

        self.Wx = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self.Wh = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
    
    def forward(self, x, h):
        return self.activation(x @ self.Wx + h @ self.Wh + self.b)

