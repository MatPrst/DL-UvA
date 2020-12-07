"""
This module implements a GRU in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(GRU, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = device

        self._embedding_size = 10

        self.embedding = nn.Embedding(self._input_dim, self._embedding_size)
        self.Wz = nn.Parameter(torch.zeros(self._hidden_dim + self._embedding_size, self._hidden_dim))
        self.Wr = nn.Parameter(torch.zeros(self._hidden_dim + self._embedding_size, self._hidden_dim))
        self.W = nn.Parameter(torch.zeros(self._hidden_dim + self._embedding_size, self._hidden_dim))

        self.Wph = nn.Parameter(torch.zeros(self._hidden_dim, self._num_classes))
        self.bp = nn.Parameter(torch.zeros(self._num_classes))

        for name, param in self.named_parameters():
            if "W" in name:
                torch.nn.init.kaiming_normal_(param, nonlinearity='linear')


    def forward(self, x):
        h_t = torch.zeros(x.shape[0], self._hidden_dim).to(self._device)

        x = self.embedding(x.squeeze().long())
        for t in range(self._seq_length):
            x_t = x[:,t]
            c = torch.cat((h_t, x_t), dim=1)

            z_t = torch.sigmoid(c @ self.Wz)
            r_t = torch.sigmoid(c @ self.Wr)

            cp = torch.cat((r_t * h_t, x_t), dim=1)
            h_hat_t = torch.tanh(cp @ self.W)
            h_t = (1 - z_t) * h_t + z_t * h_hat_t

            p_t = h_t @ self.Wph + self.bp
        
        return p_t

