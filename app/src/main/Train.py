import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
from val import val
import argparse


parser = argparse.ArgumentParser(description="DisPro")
parser.add_argument('--gpus', nargs='+', type=str, default='0', help='gpus')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=100, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_path', type=str, default='./', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True

def train(dataset_root_dir):
    train_folder = dataset_root_dir + args.dataset_type + "/training/frames"
    test_folder = dataset_root_dir + args.dataset_type + "/testing/frames"

    train_dataset = DataLoader(train_folder, transforms.Compose([
                transforms.ToTensor(),          
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                                shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)


    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    params_encoder =  list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    optimizer = torch.optim.Adam(params, lr = args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
    model.cuda()


    log_dir = os.path.join('../app/src/main/exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    loss_func_mse = nn.MSELoss(reduction='none')

    print("{}, Start Training!".format(time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())))

    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
    best_auc = 0.
    best_epoch = 0
    iter = 0

    for epoch in range(args.epochs):
        print('----------------------------------------')
        print('Epoch:', epoch+1)
        print('----------------------------------------')
        labels_list = []
        model.train()
        
        start = time.time()
        for j,(imgs) in enumerate(train_batch):
            
            iter = iter + 1
            imgs = Variable(imgs).cuda()
            rgbdiff = imgs[:,:12] - torch.cat([imgs[:,12:], imgs[:,12:], imgs[:,12:], imgs[:,12:]], 1)

            outputs, output_rgbdiff, _, _, m_items, softmax_score_query, softmax_score_memory, _, _, J3 = model.forward(imgs[:,0:12], m_items, True)       
            
            optimizer.zero_grad()
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
            loss_rgbdiff = torch.mean(loss_func_mse(output_rgbdiff, rgbdiff))

            loss = loss_pixel + 0.1 * loss_rgbdiff + J3
            loss.backward(retain_graph=True)
            optimizer.step()

            if j % 200 == 0:
                print('Iter:', j)
                print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}/ disProLoss {:.6f}/ RGBdiffLoss {:.6f}'.format(loss_pixel.item(), compactness_loss.item(), separateness_loss.item(), J3.item(), 0.1*loss_rgbdiff.item()))
            
            if iter % 500 == 0:
                auc = val(args, model=model, m_items=m_items)
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + 1

                    torch.save(model, os.path.join(log_dir, 'epoch_' + str(iter) + '_model.pth'))
                    torch.save(m_items, os.path.join(log_dir, 'epoch_' + str(iter) + '_keys.pt'))
                print(f'Best_AUC_is_{best_epoch}_Epoch: {best_auc}\n')
                model.train()

        scheduler.step()
        

    print('Training is finished')

