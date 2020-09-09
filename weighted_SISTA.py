import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import main


class Weighted_SISTA(nn.Module):
    def __init__(self, options):
        super(Weighted_SISTA, self).__init__()

        matrices = {}
        f = h5py.File(options.mat_file)
        for v, k in f.items():
            matrices[v] = k
        A_name = 'A{0}'.format(options.measure_rate)

        f_D = h5py.File("../../data/256/Sparse_Dics.mat")
        for v, k in f_D.items():
            matrices[v] = k
        D_name = 'TM_{0}layers'.format(options.D_level)

        #print(matrices)

        self.dim = options.dim
        self.num_iters = options.num_iters

        self.A = nn.Parameter(torch.tensor(matrices[A_name], dtype=torch.float32).t(), requires_grad=True)
        self.D = nn.Parameter(torch.tensor(matrices[D_name], dtype=torch.float32), requires_grad=True)
        self.F = nn.Parameter(torch.eye(self.dim, dtype=torch.float32), requires_grad=True)

        self.lambda_1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.lambda_2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.h_0_K = nn.Parameter(torch.zeros((self.dim, 1), dtype=torch.float32), requires_grad=True)

        self.I = nn.Parameter(torch.eye(self.dim, dtype=torch.float32), requires_grad=False)

        U = torch.ones(self.dim, 1, dtype=torch.float32)
        U[0:16] = 0.0
        #U[16:32] = 1.0
        #U[32:64] = 1.0
        #U[64:128] = 1.0
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

        # print(self.D.t()[:, 0].cpu().detach().numpy())
        # plt.plot(self.D.t()[0, :].cpu().detach().numpy())
        # plt.show()
        # x1 = torch.matmul(self.D, y)
        # for i in range(15):
        #     plt.plot(y[0, :, i].cpu().detach().numpy())
        #     plt.show()
        #     plt.plot(x1[0, :, i].cpu().detach().numpy())
        #     plt.show()


        x = torch.matmul(self.A, y)

        h = torch.zeros(*y.size(), requires_grad=True).to(main.device)
        h_pre_K = self.h_0_K
        for t in range(x.size(-1)):
            h_t = torch.matmul(P, h_pre_K)
            for k in range(1, self.num_iters+1):
                z = torch.matmul(S, h_t) + \
                    torch.matmul(V, x[:, :, t].unsqueeze(2)) + \
                    torch.matmul((self.lambda_2 / self.alpha) * P, h_pre_K)

                h_t = self.soft_threshold(z, self.U, self.lambda_1 / self.alpha)
            # plt.plot(h_t[0].cpu().detach().numpy())
            # plt.plot(y[0, :, t].cpu().detach().numpy())
            # plt.plot(bh_z[0].cpu().detach().numpy())
            # plt.plot(h_t[0].cpu().detach().numpy())
            # plt.show()
            h_pre_K = h_t
            h[:, :, t] = h_t.squeeze(2)

        y_re = torch.matmul(self.D, h)
        # for i in range(y.size(1)):
        #     #plt.plot(h[:, i].cpu().detach().numpy())
        #     plt.plot(y[0, :, i].cpu().numpy())
        #     plt.plot(y_re[0, :, i].cpu().detach().numpy())
        #     plt.show()
        return y_re








