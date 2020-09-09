import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt



class CS_LSTM(nn.Module):
    def __init__(self, options):
        super(CS_LSTM, self).__init__()

        matrices = {}
        f = h5py.File(options.mat_file)
        for v, k in f.items():
            matrices[v] = k
        A_name = 'A{0}'.format(options.measure_rate)

        self.A = nn.Parameter(torch.tensor(matrices[A_name], dtype=torch.float32, requires_grad=False).t())
        self.rnn = nn.LSTM(input_size=options.measure_rate, hidden_size=options.dim, num_layers= options.num_iters, \
                              batch_first=True)

    def forward(self, y):
        x = torch.matmul(self.A, y)
        x = x.transpose(1,2)
        y_re, _ = self.rnn(x)
        y_re = y_re.transpose(1,2)

        return y_re



