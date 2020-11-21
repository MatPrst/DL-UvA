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

        self.Wgx = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        self.Wgh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.bg = nn.Parameter(torch.zeros(self.hidden_dim))

        self.Wix = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        self.Wih = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.bi = nn.Parameter(torch.zeros(self.hidden_dim))

        self.Wfx = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        self.Wfh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.bf = nn.Parameter(torch.zeros(self.hidden_dim))

        self.Wox = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        self.Woh = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        self.bo = nn.Parameter(torch.zeros(self.hidden_dim))

        self.Wph = nn.Parameter(torch.zeros(self.hidden_dim, self.num_classes))
        self.bp = nn.Parameter(torch.zeros(self.num_classes))

        for name, param in self.named_parameters():
            if name.startswith("W"):
                torch.nn.init.kaiming_normal_(param.data, nonlinearity='linear')


    def forward(self, x):
        h_t = torch.zeros(x.shape[0], self.hidden_dim).to(self.device)
        c_t = torch.zeros(x.shape[0], self.hidden_dim).to(self.device)

        x = self.embeding(x.squeeze().long())
        for t in range(self.seq_length):
            x_t = x[:,t]

            g_t = torch.tanh(x_t @ self.Wgx + h_t @ self.Wgh + self.bg)
            i_t = torch.sigmoid(x_t @ self.Wix + h_t @ self.Wih + self.bi)
            f_t = torch.sigmoid(x_t @ self.Wfx + h_t @ self.Wfh + self.bf)
            o_t = torch.sigmoid(x_t @ self.Wox + h_t @ self.Woh + self.bo)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

            p_t = h_t @ self.Wph + self.bp
        return p_t
