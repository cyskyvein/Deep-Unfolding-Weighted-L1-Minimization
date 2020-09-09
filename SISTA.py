import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import main


class SISTA_RNN(nn.Module):
    def __init__(self, options):
        super(SISTA_RNN, self).__init__()

        matrices = {}
        f = h5py.File(options.mat_file)
        for v, k in f.items():
            matrices[v] = k
        A_name = 'A{0}'.format(options.measure_rate)

        self.dim = options.dim
        self.num_iters = options.num_iters

        self.A = nn.Parameter(torch.tensor(matrices[A_name], dtype=torch.float32, requires_grad=True).t())
        self.D = nn.Parameter(torch.tensor(matrices['TM'], dtype=torch.float32, requires_grad=True))
        self.F = nn.Parameter(torch.eye(self.dim, dtype=torch.float32, requires_grad=True))

        self.lambda_1 = nn.Parameter(torch.tensor(0.05, requires_grad=True))
        self.lambda_2 = nn.Parameter(torch.tensor(0.02, requires_grad=True))
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.h_0_K = nn.Parameter(torch.zeros((self.dim, 1), dtype=torch.float32, requires_grad=True))

        self.I = torch.eye(self.dim, dtype=torch.float32, requires_grad=False).to(main.device)

        U = torch.ones(self.dim, 1, dtype=torch.float32)
        # U[0:16] = 0.0
        # U[16:32] = 0.5
        # U[32:64] = 1.0
        # U[64:128] = 2.0
        self.U = nn.Parameter(U, requires_grad=False)


    def soft_threshold(self, z, U, b):
        threshold = nn.Threshold(0, 0)
        sign = torch.sign(z)
        abs_z = torch.abs(z) - U * b
        threshold_z = sign * threshold(abs_z)
        return threshold_z


    def forward(self, y):

        P = self.D.t().mm(self.F).mm(self.D)
        S = self.I - (1.0 / self.alpha) * (self.D.t().mm(self.A.t().mm(self.A) + self.lambda_2 * self.I).mm(self.D))
        V = (1 / self.alpha) * self.D.t().mm(self.A.t())



        x = torch.matmul(self.A, y)

        h = torch.zeros(*y.size(), requires_grad=True).to(main.device)
        h_pre_K = self.h_0_K
        for t in range(x.size(-1)):
            h_t = torch.matmul(P, h_pre_K)
            for k in range(1, self.num_iters+1):
                z = torch.matmul(S, h_t) + \
                    torch.matmul(V, x[:,:,t].unsqueeze(2))+ \
                    torch.matmul((self.lambda_2 / self.alpha) * P, h_pre_K)

                h_t = self.soft_threshold(z, self.U, self.lambda_1 / self.alpha)
                #plt.plot(h_r[:, t].detach().numpy())
                #plt.plot(bh_z[0].cpu().detach().numpy())
                #plt.plot(bh_h_t[0].cpu().detach().numpy())
                #plt.show()
            h_pre_K = h_t
            h[:, :, t] = h_t.squeeze(2)

        y_re = torch.matmul(self.D, h)

            #plt.plot(h_t.detach().numpy())
            #plt.plot(y[:, t].numpy())
            #plt.plot(y_re[:, t].detach().numpy())
            #plt.show()
        return y_re








