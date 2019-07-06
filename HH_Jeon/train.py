import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

from IS_dataloader_paper import IMGRTDataset
from torch.utils.data import Dataset, DataLoader


import cv2
#import pytorch_model as my_model
import my_resnet_paper as my_resnet
import feature_loss_paper as custom_feature
import temporal_loss_paper as custom_temporal
#from torchviz import make_dot


#import pytorch_model as my_model
import matplotlib.pyplot as plt

from visdom import Visdom
from torchsummary import summary
import util_visdom as utils
import torch.nn.functional as F

# pytorch version check
# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))




# DB_path = "/media/deep/db/dataset/DeepStab/test"
DB_path = "/media/deep/db/dataset/DeepStab"

folder_name = []
folder_name.append("/stable")
folder_name.append("/unstable")
folder_name.append("/images")

path_info = [DB_path, folder_name]


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for object detection from a CSV file.')
    
    parser.add_argument('--train_path', help='Path to CSV file for training (required)')

    parser.add_argument('--epochs', help='Path to a CSV file containing class label mapping (optional)', default=80)
    parser.add_argument('--val_path', help='Path to CSV file for validation (optional')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).',
                        default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=8, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default = 'csv')
    parser.add_argument('--verbose', help='Print for debug.', default=1, type=int)
    parser.add_argument('--im-size', help='Size of the image.', default=600, type=int)
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')


   
    return parser.parse_args()


def count_parameters(model):
    #To compute the number of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_param(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def imshow(img):
    #img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow((np.transpose(npimg, (1, 2, 0))*255).astype(np.uint8))
    


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def main(args = None):
    # parse arguments
    args = parse_args()

    global plotter, plotter_total, plotter_RT, plotter_pixel, plotter_feature
    plotter = utils.VisdomLinePlotter(env_name='IS_RT Plots')
    plotter_total = utils.VisdomLinePlotter(env_name='IS_RT Plots')
    plotter_RT = utils.VisdomLinePlotter(env_name='IS_RT Plots')
    plotter_pixel = utils.VisdomLinePlotter(env_name='IS_RT Plots')
    plotter_feature = utils.VisdomLinePlotter(env_name='IS_RT Plots')
    plotter_temporal = utils.VisdomLinePlotter(env_name='IS_RT Plots')

    # object to store & plot the losses
    losses = utils.AverageMeter()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    epochs = args.epochs
    batch_size = args.batch_size
    #train_path = 1
    verbose = args.verbose
    #print("1")
    # Create the data loaders
    if args.dataset == 'coco':
        if args.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

    elif args.dataset == 'csv':
        
        args.train_path = "./DB/total/test.csv"
        if args.train_path is None:
            raise ValueError('Must provide --csv_train_src when training on COCO,')
        

        dataset_train = IMGRTDataset(train_file=path_info)
        
    print(dataset_train.image_path_list[0])
    random.shuffle(dataset_train.image_path_list)
    print(dataset_train.image_path_list[0])
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=batch_size, drop_last=True, shuffle=True)

    # Create the model
    model = my_resnet.resnet50(pretrained=True)

    print("models : ", model)
    #summary(model, input_size=(14,512,288))
    print("params : ", count_parameters(model))
    
    use_gpu = True
    
    # if use_gpu:
    #     model = model.cuda()
        
    # model = torch.nn.DataParallel(model).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1 and use_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.training = True

    # optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    
    model.train()
    #model.module.freeze_bn()
    iter_size = len(dataset_train)
    print('Num training images: {}'.format(iter_size))

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    fig, axs = plt.subplots(3, 2, constrained_layout=True)

    prev_cnt = 0
    prev_idx = 0
    identity_flag = 1

    epoch_loss = 0
    plot_cnt = 0
    gamma = 1

    for epoch_num in range(args.epochs):
        
        model.train()
        model.module.freeze_bn()

        for iter_num, data in enumerate(dataloader_train):

            optimizer.zero_grad()

            
            if (data[2] is not None):

                if (skip_flag == 0):

                    set_t_1, set_t_0, M, data_st_0_0, new_src_pts, new_dst_pts, pt_num, tp_M = data

                    out_img_t_0, out_img_t_1, x_t_1, x_t_0, theta_t_1, theta_t_0 = model([input_t_1, input_t_0, input_it_1_1, input_it_0_0])

                    RT_MSE_loss = criterion(x_t_0, RT)

                    loss = RT_MSE_loss

                    if bool(loss == 0):
                        continue

                    loss.backward()

                    optimizer.step()
                    loss_value = loss.data.cpu().numpy()
                    RT_loss_value = RT_MSE_loss.data.cpu().numpy()
                    pixel_loss_value = pixel_MSE_loss.data.cpu().numpy()
                    feature_loss_value = feature_MSE_loss.data.cpu().numpy()
                    temporal_loss_value = temporal_loss.data.cpu().numpy()

                    losses.update(loss_value, RT.size(0))

                    print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch_num + 1, iter_num + 1, loss_value))

                    print('Epoch: {} | Iteration: {} | Loss: {}'.format(epoch_num + 1, iter_num + 1, loss_value),
                          end='\r')

                    if (iter_num % 100) == 0 or ((iter_num == 0) and (epoch_num == 0)):
                        # plotter.plot('loss', 'train', 'Class Loss', epoch_num, losses.avg)
                        plotter_total.plot('loss', 'train', 'Total Loss', plot_cnt, losses.avg)
                        plotter_RT.plot('loss', 'train', 'RT Loss', plot_cnt, 50 * RT_loss_value)
                        plotter_pixel.plot('loss', 'train', 'pixel Loss', plot_cnt, 5 * pixel_loss_value)
                        plotter_feature.plot('loss', 'train', 'feature Loss', plot_cnt, 1 * feature_loss_value)
                        plotter_temporal.plot('loss', 'train', 'temporal Loss', plot_cnt, 10 * temporal_loss_value)
                        plot_cnt = plot_cnt + 1

                    if (epoch_num%2 == 0 and iter_num == 1):
                        torch.save(model.state_dict(), "/media/deep/db/deep/utils/my_tutorial/IS_paper/weights/IS_" + str(epoch_num) + '_' + str(loss_value))

        scheduler.step(np.mean(loss_value), epoch_num)

                # scheduler.step(np.mean(epoch_loss))
        
if __name__ == '__main__':
    main()    
    print('Training complete, exiting.')






