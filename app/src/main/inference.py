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
import glob
from collections import OrderedDict
import argparse
import ipdb

parser = argparse.ArgumentParser(description="DisPro")
parser.add_argument('--gpus', nargs='+', type=str, default='0', help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_path', type=str, default='./', help='directory of data')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')

args = parser.parse_args()


model_dir = '../app/src/main/exp/epoch_17200_model.pth'
m_items_dir = '../app/src/main/exp/epoch_17200_keys.pt'
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


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


def inference(video_dir):
    video = video_dir
    video_name = video.split('/')[-1]
        
    videos = OrderedDict()
    videos[video_name] = {}
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])	
    
    def video_process():
        transform = transforms.Compose([transforms.ToTensor()])
        frames = []
        for i in range(len(videos[video_name]['frame'])-args.t_length-1):
            frames.append(videos[video_name]['frame'][i])

        batch = []
            
        for k in range(len(frames)):
            for i in range(k, k + args.t_length-1 + 1):
                image = np_load_frame(videos[video_name]['frame'][i], args.h, args.w)
                batch.append(transform(image))
        
        return np.concatenate(batch, axis=0)

                                
    loss_func_mse = nn.MSELoss(reduction='none')

    model = torch.load(model_dir)
    model.cuda()
    m_items = torch.load(m_items_dir)


    labels = np.load('../app/src/main/data/frame_labels_'+args.dataset_type+'.npy')
    if args.dataset_type == 'shanghai':
        labels = np.expand_dims(labels, 0)

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Anomaly Score of Each Frame of ', video)

    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    m_items_test = m_items.clone()

    model.eval()
    test_batch = video_process()  # [2610, 256, 256]


    for i in range(len(videos[video_name]['frame'])-args.t_length-1):
        
        imgs = Variable(torch.from_numpy(test_batch[i * 3 : i * 3 + 15])).cuda()
        imgs = imgs.unsqueeze(dim=0)

        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()
        
        

        psnr_list[video.split('/')[-1]].append(psnr(mse_imgs))
        feature_distance_list[video.split('/')[-1]].append(mse_feas)


    anomaly_score_total_list = []
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                    anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    return anomaly_score_total_list

