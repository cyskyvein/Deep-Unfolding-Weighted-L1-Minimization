import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
import random
import scipy
import cv2
import glob
import time
import os
import h5py
import pytorch_ssim
from datasets import load_data, Caltech
from torch.utils.data import DataLoader
import main


def img_show(img):
    plt.imshow(img.numpy(),cmap='gray')
    plt.show()

def imgs_display(ori_imgs, re_imgs):
    imgs_num = ori_imgs.size(0)
    fig=plt.figure(figsize=(6, imgs_num*4))
    rows = imgs_num
    cols = 2
    for i in range(imgs_num):
        fig.add_subplot(rows, cols, 2*i+1)
        plt.imshow(ori_imgs[i].cpu().numpy(),cmap='gray')
        fig.add_subplot(rows, cols, 2*i+2)
        plt.imshow(re_imgs[i].cpu().numpy(),cmap='gray')
    plt.show()

def imgs_save(phase, i, ori_imgs, re_imgs):
    for j, (ori_img, re_img) in enumerate(zip(ori_imgs, re_imgs)):
        torchvision.utils.save_image(ori_img, './images/{0}_{1}_batch_{2}_ori.jpg'.format(phase,i, j))
        torchvision.utils.save_image(re_img, './images/{0}_{1}_batch_{2}_re.jpg'.format(phase, i, j))

def divide_data(options):
    data, _, _, _ = load_data((options.dim, options.dim), options.data_path)
    random.shuffle(data)
    length = len(data)
    bound = int(length * options.data_prop)
    for i in range(bound):
        scipy.misc.imsave('../data/train/train_{0}.jpg'.format(i), data[i])
    for i in range(bound, length):
        scipy.misc.imsave('../data/test/test_{0}.jpg'.format(i-bound), data[i])

def read_ordered_imgs(options):
    # read test images and prepare testload
    imgs_num = len(os.listdir(options.test_dir))
    test_imgs = []
    for i in range(imgs_num):
        filename = "test_{0}.jpg".format(i)
        filepath = options.test_dir + filename
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) / 255.0
        test_imgs.append(image)
    test_dataset = Caltech(test_imgs)
    test_load = DataLoader(test_dataset, options.test_batch_size, shuffle=False)
    return test_load


def each_criterion(model_name, test_load, options, criterion):
    net = torch.load(options.model_dir+model_name+".pkl", map_location=main.device)

    p_sum = 0
    num = 0
    with open("./result/each_{0}/{1}_{0}.txt".format(criterion.__name__, model_name), mode='w') as f:
        with torch.no_grad():
            for imgs in test_load:
                imgs = imgs.to(main.device)
                re_imgs = net(imgs)

                # restrict the value of images in 0 to 1
                re_imgs = re_imgs.clamp(min=0, max=1)


                # for image, re_image in zip(imgs, re_imgs):
                #     p = psnr(image, re_image)
                #     f.write("{0:.2f}\t".format(p))
                criterion_batch = criterion(imgs, re_imgs)
                if torch.isnan(criterion_batch).sum() > 0 or torch.isinf(criterion_batch).sum() > 0:
                    criterion_batch[torch.isnan(criterion_batch)] = 0
                    criterion_batch[torch.isinf(criterion_batch)] = 0
                for cri_val in criterion_batch:
                    f.write("{0:.2f}\t".format(cri_val))

                p_sum += torch.sum(criterion_batch)
                num += criterion_batch.size(0)

    mean_criterion = p_sum / num
    print("{0}:{1:.2f}".format(model_name, mean_criterion))
    return mean_criterion

def save_re_img(img_Nos, modelname, options):
    with torch.no_grad():
        imgs = [cv2.imread("{0}test_{1}.jpg".format(options.test_dir, img_No), cv2.IMREAD_GRAYSCALE) / 255.0 for img_No in img_Nos]
        imgs = torch.tensor(imgs, dtype=torch.float32).to(main.device)
        net = torch.load(options.model_dir + modelname + ".pkl", map_location=main.device)

        re_imgs = net(imgs)
        # restrict the value of images in 0 to 1
        re_imgs = re_imgs.clamp(min=0, max=1)

        psnr_batch = psnr(imgs, re_imgs)
        print(psnr_batch)
        #imgs_display(imgs, re_imgs)
    for i, re_img in enumerate(re_imgs):
        torchvision.utils.save_image(re_img, "{0}/{2}/re_{1}_{2}.bmp".format(options.reimgs_dir, img_Nos[i], modelname))

def list_psnr(img_Nos, modelname, options):
    imgs = [cv2.imread("{0}test_{1}.jpg".format(options.test_dir, img_No), cv2.IMREAD_GRAYSCALE) / 255.0 for img_No in
            img_Nos]
    imgs = torch.tensor(imgs, dtype=torch.float32).to(main.device)
    net = torch.load(options.model_dir + modelname + ".pkl", map_location=main.device)
    with torch.no_grad():
        re_imgs = net(imgs)

    # restrict the value of images in 0 to 1
    re_imgs = re_imgs.clamp(min=0, max=1)

    imgs_display(imgs, re_imgs)

    f = open("./result/list_psnr.txt", mode='a+')
    f.write("\n\t\t")
    for No in img_Nos:
        f.write("\t{0}".format(No))
    f.write("\n")
    f.write(modelname + ":\t\t")
    psnr_batch = psnr(imgs, re_imgs)
    for p in psnr_batch:
        f.write('{0:.2f}\t'.format(p))


def psnr(imgs, re_imgs):
    crit = nn.MSELoss(reduction='none')
    mse = torch.mean(crit(imgs, re_imgs), dim=(1, 2))
    psnr = 10*torch.log10(1.0/mse)
    return psnr


def ssim(imgs, re_imgs):
    ssim_list = torch.zeros(imgs.size(0)).to(main.device)
    for i, (img, re_img) in enumerate(zip(imgs, re_imgs), 0):
        img = img.unsqueeze(0)
        re_img = re_img.unsqueeze(0)
        img = img.unsqueeze(0)
        re_img = re_img.unsqueeze(0)
        # img, re_img = img*255, re_img*255
        ssim_list[i] = pytorch_ssim.ssim(img, re_img)
    return ssim_list




def prd(imgs, re_imgs):
    diff = imgs - re_imgs
    return diff.norm(dim=(1, 2))/imgs.norm(dim=(1, 2))*100

def read_psnr_file(filepath):
    with open(filepath, mode='r') as f:
        psnr_str = f.read()
        psnr_list = psnr_str.strip().split()
        psnr_list = list(map(float, psnr_list))
    return  psnr_list

def diff_psnr(model1,model2, n, options):
    filepath1 = './result/each_psnr/' + model1 + '_psnr.txt'
    psnr_list1 = np.array(read_psnr_file(filepath1))
    filepath2 = './result/each_psnr/' + model2 + '_psnr.txt'
    psnr_list2 = np.array(read_psnr_file(filepath2))
    diff = psnr_list1 - psnr_list2
    print(diff)
    idx = (-diff).argsort()[:n]
    print(idx)
    print(diff[idx])
    print(psnr_list1[idx])
    print(psnr_list2[idx])
    return idx


def read_ECG(file):
    data = h5py.File(file)
    arrays = {}
    for k, v in data.items():
        arrays[k] = np.array(v)
    return arrays['Train_Data'], arrays['Test_Data']

# x = read_ECG('../data/heart/data_PTB_ECG_Dataset_15Channels_1024.mat')
