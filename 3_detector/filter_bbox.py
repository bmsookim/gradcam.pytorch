# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Grad CAM Implementation
#
# Description : filter_bbox.py
# The code for filtering the regions by the first prediction.
# ***********************************************************

import cv2
import os
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import argparse
import config as cf
import operator
import csv

from torchvision import datasets, models, transforms
from networks import *
from torch.autograd import Variable
from PIL import Image

parser = argparse.ArgumentParser(description='Pytorch Cell Classification weight upload')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=int, help='depth of model')
parser.add_argument('--start', default=1, type=int, help='starting index')
parser.add_argument('--finish', default=21, type=int, help='finishing index')
args = parser.parse_args()

# Phase 1 : Model Upload
print('\n[Test Phase] : Model Weight Upload')
use_gpu = torch.cuda.is_available()

# upload labels
data_dir = cf.aug_base
trainset_dir = cf.data_base.split("/")[-1]+os.sep
dsets = datasets.ImageFolder(data_dir, None)

H = datasets.ImageFolder(os.path.join(cf.aug_base, 'train'))
H_NH = datasets.ImageFolder(os.path.join(cf.aug_dir, 'WBC_NH', 'train'))
H_LH = datasets.ImageFolder(os.path.join(cf.aug_dir, 'WBC_LH', 'train'))
dset_classes = H.classes
H1_classes = H_NH.classes
H2_classes = H_LH.classes

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getNetwork(args):
    if (args.net_type == 'alexnet'):
        file_name = 'alexnet'
    elif (args.net_type == 'vggnet'):
        file_name = 'vgg-%s' %(args.depth)
    elif (args.net_type == 'resnet'):
        file_name = 'resnet-%s' %(args.depth)
    else:
        print('[Error]: Network should be either [alexnet / vgget / resnet]')
        sys.exit(1)

    return file_name

# uploading the model
print("| Loading checkpoint model for crop inference...")
assert os.path.isdir('../3_classifier/checkpoint'),'[Error]: No checkpoint directory found!'
assert os.path.isdir('../3_classifier/checkpoint/'+trainset_dir),'[Error]: There is no model weight to upload!'
file_name = getNetwork(args)
checkpoint = torch.load('../3_classifier/checkpoint/'+trainset_dir+file_name+'.t7')
checkpoint_nh = torch.load('../3_classifier/checkpoint/WBC_NH/'+file_name+'.t7')
checkpoint_lh = torch.load('../3_classifier/checkpoint/WBC_LH/'+file_name+'.t7')
model = checkpoint['model']
model_NH = checkpoint_nh['model']
model_LH = checkpoint_lh['model']

if use_gpu:
    model.cuda()
    cudnn.benchmark = True

model.eval()
model_NH.eval()
model_LH.eval()

sample_input = Variable(torch.randn(1,3,224,224), volatile=False)
if use_gpu:
    sampe_input = sample_input.cuda()

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean, cf.std)
])

h1_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.h1_mean, cf.h1_std)
])

h2_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.h2_mean, cf.h2_std)
])

def is_image(f):
    return f.endswith(".png") or f.endswith(".jpg")

def return_thresh(img, thresh_min, thresh_max=255, is_gray=True):
    if(is_gray):
        input_img = img
    else:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(input_img, thresh_min, thresh_max, cv2.THRESH_BINARY_INV)

    return thresh

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

# directory check
in_dir = './results/cropped/'
out_dir = './results/filtered/'
check_and_mkdir(out_dir)

# main function
if __name__ == "__main__":
    for file_number in range(1, 27+1):
        print("Filtering TEST%d..." %file_number)
        save_dir = os.path.join(out_dir, 'TEST%d' %file_number)
        check_and_mkdir(save_dir)
        with open(os.path.join(in_dir, 'TEST%d' %file_number, 'TEST%d.csv' %file_number), 'r') as csvfile:
            with open(os.path.join(out_dir, 'TEST%d' %file_number, 'TEST%d.csv' %file_number), 'w') as outfile:
                fieldnames = ['prediction', 'x', 'y', 'w', 'h']
                reader = csv.reader(csvfile)
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                img = cv2.imread('/home/bumsoo/Data/test/CT_20/TEST%d.png' %file_number)
                original_img = img
                for row in reader:
                    x, y, w, h = map(int, row[1:])
                    print(x, y, w, h)
                    subtype = row[0]

                    pred = original_img[y:y+h, x:x+w]

                    th_img = pred[:,:,2] - pred[:,:,0]

                    thr = return_thresh(th_img, th_img.mean())
                    _, contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    max_cnt = 0
                    max_area = 0

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if(area > max_area):
                            max_cnt = cnt
                            max_area = area

                    if (max_area < ((pred.shape[0] * pred.shape[1])/3)):
                        pred_x, pred_y, pred_w, pred_h = cv2.boundingRect(max_cnt)
                        real_x, real_y, real_w, real_h = x+pred_x, y+pred_y, pred_w, pred_h
                        crop = pred[pred_y:pred_y+pred_h, pred_x:pred_x+pred_h]
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) # Swap the image into RGB input

                        if test_transform is not None:
                            img = test_transform(Image.fromarray(crop, mode='RGB'))

                        inputs = img
                        inputs = Variable(inputs, volatile=True)

                        if use_gpu:
                            inputs = inputs.cuda()
                        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                        outputs = model(inputs)
                        softmax_res = softmax(outputs.data.cpu().numpy()[0])
                        index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

                        if (subtype == 'RBC'):
                            print("RBC")
                            pass
                        else:
                            if ("Neutrophil" in dset_classes[index] and score < 0.9):
                                if h1_transform is not None:
                                    img = h1_transform(Image.fromarray(crop, mode='RGB'))
                                inputs = img
                                inputs = Variable(inputs, volatile=True)

                                if use_gpu:
                                    inputs = inputs.cuda()
                                inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                                H1_outputs = model_NH(inputs)
                                hr_softmax = softmax(H1_outputs.data.cpu().numpy()[0])
                                idx, sc = max(enumerate(hr_softmax), key=operator.itemgetter(1))

                                subtype = H1_classes[idx]
                            elif "Lymphocyte" in dset_classes[index] and score < 0.9:
                                if h2_transform is not None:
                                    img = h2_transform(Image.fromarray(crop, mode='RGB'))
                                inputs = img
                                inputs = Variable(inputs, volatile=True)

                                if use_gpu:
                                    inputs = inputs.cuda()
                                inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                                H2_outputs = model_NH(inputs)
                                hr_softmax = softmax(H2_outputs.data.cpu().numpy()[0])
                                idx, sc = max(enumerate(hr_softmax), key=operator.itemgetter(1))

                                subtype = H2_classes[idx]
                            else:
                                subtype = dset_classes[index]

                    else:
                        real_x, real_y, real_w, real_h = x, y, w, h

                    writer.writerow({
                        'prediction': subtype,
                        'x': real_x,
                        'y': real_y,
                        'w': real_w,
                        'h': real_h
                    })
