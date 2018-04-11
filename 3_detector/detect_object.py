# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Grad CAM Implementation
#
# Description : detect_cell.py
# The main code for grad-CAM image localization.
# ***********************************************************

from __future__ import print_function, division

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import config as cf
import torchvision
import time
import copy
import os
import sys
import argparse
import csv
import operator

from grad_cam import *
from time import sleep
from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image
from misc_functions import save_class_activation_on_image
from grad_cam import BackPropagation, GradCAM, GuidedBackPropagation

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--subtype', default=None, type=str, help='Type to find')
args = parser.parse_args()

# Phase 1 : Model Upload
print('\n[Phase 1] : Model Weight Upload')
use_gpu = torch.cuda.is_available()

# upload labels
data_dir = cf.aug_base
trainset_dir = cf.data_base.split("/")[-1]+os.sep

dsets = datasets.ImageFolder(data_dir, None)
H = datasets.ImageFolder(os.path.join(cf.aug_base, 'train'))
dset_classes = H.classes

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'resnet'):
        file_name = 'resnet-%s' %(args.depth)
    elif (args.net_type == 'densenet'):
        file_name = 'densenet-%s' %(args.depth)
    else:
        print('[Error]: Network should be either [alexnet / vgget / resnet / densenet]')
        sys.exit(1)

    return file_name

def random_crop(image, dim):
    if len(image.shape):
        W, H, D = image.shape
        w, h, d = dim
    else:
        W, H = image.shape
        w, h = size

    left, top = np.random.randint(W-w+1), np.random.randint(H-h+1)
    return image[left:left+w, top:top+h], left, top

def return_class_idx(class_name):
    global dset_classes

    for i,j in enumerate(dset_classes):
        if class_name == j:
            return i

    print(class_name + " is not an appropriate class to search.")
    sys.exit(1) # Wrong class name input

def generate_sliding_windows(image, stepSize, windowSize):
    list_windows = []
    cnt = 0

    for x in xrange(0, image.size[0], stepSize):
        for y in range(0, image.size[1], stepSize):
            if(x+windowSize <= image.size[0] and y+windowSize <= image.size[1]):
                list_windows.append(image.crop((x,y,x+windowSize,y+windowSize)))
            elif (x+windowSize > image.size[0] and y+windowSize > image.size[1]) :
                list_windows.append(image.crop((image.size[0]-windowSize,image.size[1]-windowSize,image.size[0],image.size[1])))
            elif (x+windowSize > image.size[0]):
                list_windows.append(image.crop((image.size[0]-windowSize,y,image.size[0],y+windowSize)))
            elif (y+windowSize > image.size[1]):
                list_windows.append(image.crop((x,image.size[1]-windowSize,x+windowSize,image.size[1])))

    return list_windows

def generate_padding_image(image, mode='cv2'):
    if (mode == 'cv2'):
        border_x = int((args.stepSize - ((image.shape[0]-args.windowSize)%args.stepSize)))
        border_y = int((args.stepSize - ((image.shape[1]-args.windowSize)%args.stepSize)))
        pad_image = cv2.copyMakeBorder(image, 0, border_x, 0, border_y, cv2.BORDER_CONSTANT, value=[255,255,255])
    elif (mode == 'PIL'):
        border_x = args.stepSize - ((image.size[0]-args.windowSize)%args.stepSize)
        border_y = args.stepSize - ((image.size[1]-args.windowSize)%args.stepSize)
        pad_image = Image.new("RGB", (image.size[0]+border_x, image.size[1]+border_y), color=255)
        pad_image.paste(image, (0, 0))

    return pad_image

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        print("Creating "+in_dir+"...")
        os.makedirs(in_dir)

if __name__ == "__main__":
    # uploading the model
    print("| Loading checkpoint model for grad-CAM...")
    assert os.path.isdir('../2_classifier/checkpoint'),'[Error]: No checkpoint directory found!'
    assert os.path.isdir('../2_classifier/checkpoint/'+trainset_dir),'[Error]: There is no model weight to upload!'
    file_name = getNetwork(args)
    checkpoint = torch.load('../2_classifier/checkpoint/'+trainset_dir+file_name+'.t7')
    model = checkpoint['model']

    if use_gpu:
        model.cuda()
        cudnn.benchmark = True

    model.eval()

    sample_input = Variable(torch.randn(1,3,224,224), volatile=False)
    if use_gpu:
        sampe_input = sample_input.cuda()

    def is_image(f):
        return f.endswith(".png") or f.endswith(".jpg")

    test_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])

    #@ Code for extracting a grad-CAM region for a given class
    gcam = GradCAM(model._modules.items()[0][1], cuda=use_gpu)

    print("\n[Phase 2] : Gradient Detection")
    if args.subtype != None:
        search_id = return_class_idx(args.subtype)

        if not (args.subtype in dset_classes):
            print("The given subtype does not exists!")
            args.subtype = None

    if args.subtype == None:
        print("| Checking Activated Regions for the Top-1 Class")
    else:
        print("| Checking Activated Regions for " + dset_classes[search_id] + "...")

    for subdir, dirs, files in os.walk(cf.test_dir):
        for f in files:
            file_path = os.path.join(subdir, f)
            if (is_image(f)):
                image = Image.open(file_path)
                #original = cv2.imread(file_path)
                if test_transform is not None:
                    image = test_transform(image)
                inputs = image
                inputs = Variable(inputs, volatile=False)
                if use_gpu:
                    inputs = inputs.cuda()
                inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                probs, idx = gcam.forward(inputs)

                if (args.subtype == None):
                    comp_idx = idx[0]
                    item_id = 0
                else:
                    comp_idx = search_id
                    item_id = (np.where(idx.cpu().numpy() == (search_id)))[0][0]

                gcam.backward(idx=comp_idx)
                output = gcam.generate(target_layer = 'layer4.2')

                heatmap = output
                original = inputs.data.cpu().numpy()
                print(original)
                original = np.transpose(original, (0,2,3,1))[0]
                original = original * cf.std + cf.mean
                original = np.uint8(original * 255.0)
                #original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
                #original = cv2.resize(original, (224, 224))
                mask = np.uint8(heatmap * 255.0)

                check_and_mkdir("./results/heatmaps")
                check_and_mkdir("./results/masks")

                save_dir = "./results/heatmaps/"+f
                mask_dir = "./results/masks/"+f

                print(save_dir)
                print(mask_dir)

                print(original.shape)

                gcam.save(save_dir, heatmap, original)
                cv2.imwrite("./results/"+f, original)
                cv2.imwrite(mask_dir, mask)
