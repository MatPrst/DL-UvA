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

        self.embeding = nn.Embedding(num_embeddings=input_dim, embedding_dim=10).to(self.device)

        self.Wgx = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.Wgh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bg = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.Wix = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.Wih = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bi = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.Wfx = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.Wfh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bf = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.Wox = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.Woh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bo = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.Wph = nn.Parameter(torch.Tensor(self.hidden_dim, self.num_classes))
        self.bp = nn.Parameter(torch.Tensor(self.num_classes))

        for name, param in self.named_parameters():
            if name.startswith("W"):
                torch.nn.init.kaiming_normal_(param.data, nonlinearity='linear')


    def forward(self, x):
        h_t = torch.Tensor(x.shape[0], self.hidden_dim).to(self.device)
        c_t = torch.Tensor(x.shape[0], self.hidden_dim).to(self.device)
        # print("FORWARD")
        # print(x[:,:,0])
        # print(x[:,:,0].shape)
        # print(x.shape)
        for t in range(self.seq_length):
            x_t = self.embeding(x[:,:,t].long())
            # x_t = self.embeding(x_t)
            
            # print("x_t", x_t.shape)
            # print("Wgx", self.Wgx.shape)
            # print("h_t", h_t.shape)
            # print("Wgh", self.Wgh.shape)

            # print(self.Wgx)
            # print(x_t @ self.Wgx)
            # print(x_t @ self.Wgx + h_t @ self.Wgh + self.bg)
            # return
            g_t = torch.tanh(x_t @ self.Wgx + h_t @ self.Wgh + self.bg)
            i_t = torch.sigmoid(x_t @ self.Wix + h_t @ self.Wih + self.bi)
            f_t = torch.sigmoid(x_t @ self.Wfx + h_t @ self.Wfh + self.bf)
            o_t = torch.sigmoid(x_t @ self.Wox + h_t @ self.Woh + self.bo)

            # print(g_t)
            # print(i_t)
            # print(f_t)
            # print(o_t)

            # print(g_t.device)
            # print(i_t.device)
            # print(f_t.device)
            # print(o_t.device)
            # print(c_t.device)
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

            p_t = h_t @ self.Wph + self.bp
            # y_t = torch.softmax(p_t)
        
        return p_t
