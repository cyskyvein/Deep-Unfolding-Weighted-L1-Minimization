import torch
import torchvision
import torchvision.transforms as transforms
import SISTA
import cs_lstm
import weighted_SISTA
from datasets import Caltech
from torch.utils.data import DataLoader
import train
import argparse
import cv2
import os
import glob
from utils import *
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="-----[RNN-Compressed sensing]-----")
parser.add_argument("--dim", default=128, type=int, help="the width/height of images")
parser.add_argument("--measure_rate", default=50, type=float, help="sampling rate of compressed sensing")
parser.add_argument("--num_iters", default=3, type=int, help="the number of iterations")
parser.add_argument("--interval", default=1, type=int, help="interval number of images as training samples")
parser.add_argument("--epoch", default=100, type=int, help="the number of max epoch")
parser.add_argument("--train_batch_size", default=64, type=int, help="the batch size of train data")
parser.add_argument("--test_batch_size", default=256, type=int, help="the batch size of test data")
parser.add_argument("--lr", default=1e-5, type=float, help="the learning rate of optimer")
parser.add_argument("--data_path", default="../../data/256_ObjectCategories", help="the path of dataset")
parser.add_argument("--train_dir", default="../../data/256/train/", help="the path of train images")
parser.add_argument("--test_dir", default="../../data/256/test/", help="the path of test images")
parser.add_argument("--reimgs_dir", default="../../data/reimgs/", help="the path of reconstructed images")
parser.add_argument("--data_prop", default=0.8, type=float, help="the propotion of train and test data")
parser.add_argument("--mat_file", default="../../data/256/AandD8.mat", help="matlab file including matrix A and D")
parser.add_argument("--D_level", default=2, type=float)
parser.add_argument("--model_dir", default='./saved_model/', help="file of result")



options = parser.parse_args()

def main():
    #divide data to train data and test data
    #divide_data(options)

    # read train data
    train_imgs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE)/255.0 for file in glob.glob(options.train_dir+"*.jpg")]
    # cv2.imshow('a', images[0])
    # cv2.waitKey(0)
    # print(train_imgs[0])

    # read test data and prepare testload
    test_imgs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) / 255.0 for file in glob.glob(options.test_dir + "*.jpg")]
    #train_ECG, test_ECG = read_ECG('../data/heart/data_PTB_ECG_Dataset_15Channels_1024.mat')
    test_dataset = Caltech(test_imgs)
    test_load = DataLoader(test_dataset, options.test_batch_size, shuffle=False)


    #train the model

    for options.measure_rate in [64]:
        # prepare training dataset
        for options.interval in [4]:
            train_dataset = Caltech(train_imgs, options.interval)
            train_load = DataLoader(train_dataset, options.train_batch_size, shuffle=False)
            for model in [weighted_SISTA.Weighted_SISTA]:
                for options.D_level in range(3, 4):
                    net = model(options)
                    train.train_model(net, train_load, test_load, options)
    



if __name__ == '__main__':
    main()

    # test_load = read_ordered_imgs(options)
    # for modeltype in ["Weighted_SISTA"]:
    #     for config in [64]:
    #         for sample_interval in [1]:
    #             modelname = "{0}_{1}_{2}".format(modeltype, config, sample_interval)
    #             each_criterion(modelname, test_load, options, psnr)

    # max_n_idx = diff_psnr("Weighted_SISTA_64_1", "SISTA_RNN_64_1", 100, options)
    # #
    #
    # for modeltype in ["Weighted_SISTA", "CS_LSTM", "SISTA_RNN", ]:
    #     for sample_len in ['64']:
    #         for sample_interval in [1]:
    #             modelname = "{0}_{1}_{2}".format(modeltype, sample_len, sample_interval)
    #             path = options.reimgs_dir + modelname
    #             if not os.path.exists(path):
    #                 os.makedirs(path)
    #             save_re_img(max_n_idx, modelname, options)

    # for modeltype in ["Weighted_SISTA"]:
    #     for sample_len in ['64']:
    #         for sample_interval in [1]:
    #             for options.D_level in range(2, 5):
    #                 # model_name = "{0}_{1}_{2}.pkl".format(modeltype, sample_len, sample_interval)
    #                 model_name = "{0}_{1}_{2}_D{3}.pkl".format(modeltype, sample_len, sample_interval, options.D_level)
    #                 net = torch.load(options.model_dir + model_name, map_location=device)
    #                 W = net.U.squeeze().data.cpu().numpy()
    #                 D = net.D.data.cpu().numpy()
    #                 # plt.plot(range(128), W)
    #                 # plt.show()
    #                 # D = net.D.data.cpu().numpy()
    #                 # A = net.A.data.cpu().numpy()
    #                 #
    #                 # np.savetxt("trained_D.txt", D)
    #                 np.savetxt("./result/model_param/{0}_W.txt".format(model_name), W)
    #                 np.savetxt("./result/model_param/{0}_D.txt".format(model_name), D)




