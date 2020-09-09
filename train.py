import torch.optim as optim
import torch.nn as nn
import torch
import time
import torchvision
from utils import imgs_display, psnr, imgs_save, prd
import scipy.io as sio
import main


def train_model(net, train_load, test_load, options):

    '''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    '''

    net.to(main.device)

    criterion = nn.MSELoss(reduction='sum')

    #matrix_params = filter(lambda p:)
    #optimizer = optim.Adam(net.parameters(),lr=options.lr)

    #w_params = list(map(id, net.U))

    base_params = []
    u_param = []
    for name, param in net.named_parameters():
        if name == 'U':
            u_param.append(param)
        else:
            base_params.append(param)
    #base_params = filter(lambda p: p!=net.U, net.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {"params": u_param, 'lr': 1e-2}
    ], lr=options.lr)

    f = open('./result/'+type(net).__name__+'.txt', mode='a+')
    f.write('\nThe result of {0} , measure length is {1}, sample inveral is {2}:\n'.\
            format(type(net).__name__, options.measure_rate, options.interval))

    psnr = test(net, test_load)
    print('The pnsr of test set is:{0:.2f}'.format(psnr))
    f.write('{0:.2f}\t'.format(psnr))

    for epoch in range(options.epoch):
        running_loss = 0.0
        epoch_loss = 0.0
        batch_num = 0
        #adjust_learning_rate(optimizer, epoch)
        for i, imgs in enumerate(train_load):
            optimizer.zero_grad()
            imgs = imgs.to(main.device)
            re_imgs = net(imgs)
            loss = criterion(re_imgs, imgs)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item()>100000*options.train_batch_size:
                imgs_save("train", i, imgs, re_imgs)
                print(loss)
                continue
            #loss = my_loss(imgs_re, h, imgs, net)
            loss.backward()#(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_num += 1
            if batch_num%10 == 0:
                print('[%d, %5d] loss: %.2f' %
                      (epoch+1, batch_num, running_loss/10))
                running_loss = 0

        print('[%d] epoch loss: %.2f' %
              (epoch+1, epoch_loss/(batch_num+1e-8)))

        mean_psnr = test(net, test_load)
        print('The pnsr of test set is:{0:.2f}'.format(mean_psnr))
        f.write('{0:.2f}\t'.format(mean_psnr))
        f.flush()

    torch.save(net, './saved_model/{0}_{1}_{2}_D{3}.pkl'.format(type(net).__name__, options.measure_rate, options.interval, options.D_level))

    f.write('\n')
    f.close()

    print('Finished Training')
    return net

def test(net, test_load):
    p_sum = 0
    num = 0
    with torch.no_grad():
        for i, imgs in enumerate(test_load, 0):
            imgs = imgs.to(main.device)
            start_time = time.time()
            re_imgs = net(imgs)
            #re_imgs.cpu()
            end_time = time.time()
            time_cost = (end_time - start_time)/ 256
            print("cost time:{0}".format(time_cost))

            # restrict the value of images in 0 to 1
            re_imgs = re_imgs.clamp(min=0, max=1)

            psnr_batch = psnr(imgs, re_imgs)
            if torch.isnan(psnr_batch).sum()>0 or torch.isinf(psnr_batch).sum()>0:
                print(psnr_batch)
                continue
            p_sum += torch.sum(psnr_batch)
            num += psnr_batch.size(0)

            # imgs_display(imgs.cpu(), re_imgs.cpu())
        mean_psnr = p_sum / num
    return mean_psnr

