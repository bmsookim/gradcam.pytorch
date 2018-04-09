import os
import cv2
import sys
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

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=50, type=str, help='depth of model')
args = parser.parse_args()

# Phase 1 : Model Upload
print('\n[Test Phase] : Model Weight Upload')
use_gpu = torch.cuda.is_available()

# upload labels
data_dir = cf.aug_dir+'Only_WBC'
trainset_dir = 'Only_WBC/'
dsets = datasets.ImageFolder(data_dir, None)

H = datasets.ImageFolder(os.path.join(data_dir, 'train'))
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
model = checkpoint['model']

if use_gpu:
    model.cuda()
    cudnn.benchmark = True

model.eval()

sample_input = Variable(torch.randn(1,3,224,224), volatile=False)
if use_gpu:
    sampe_input = sample_input.cuda()

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean, cf.std)
])

def check_and_mkdir(in_dir):
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

check_and_mkdir('results/baseline/')
background_root = '/home/bumsoo/Data/test/CT_20/'

for thresh in [200, 1]:
    print("| Baseline with Threshold : %d" %thresh)
    check_and_mkdir('results/baseline/%d' %thresh)
    for test_num in range(1, 27+1):
        print("\t| Inferencing TEST%d..." %test_num)
        baseline_dir = '/home/bumsoo/Data/baseline_info/%d_TEST%d.csv' %(thresh, test_num)

        with open(baseline_dir, 'r') as csvfile:
            reader = csv.reader(csvfile)

            check_and_mkdir('results/baseline/%d/TEST%d/' %(thresh, test_num))
            with open('results/baseline/%d/TEST%d/TEST%d.csv' %(thresh, test_num, test_num), 'w') as wrfile:
                fieldnames = ['prediction', 'x', 'y', 'w', 'h']
                writer = csv.DictWriter(wrfile, fieldnames=fieldnames)

                original_img = cv2.imread(background_root + 'TEST%d.png' %test_num)

                for row in reader:
                    x,y,w,h = map(int, row)
                    crop = original_img[y:y+h, x:x+w]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                    if test_transform is not None:
                        img = test_transform(Image.fromarray(crop, mode='RGB'))

                    inputs = img
                    inputs = Variable(inputs, volatile=True)

                    if use_gpu :
                        inputs = inputs.cuda()

                    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

                    outputs = model(inputs)
                    softmax_res = softmax(outputs.data.cpu().numpy()[0])
                    index, score = max(enumerate(softmax_res), key=operator.itemgetter(1))

                    pred = dset_classes[index]

                    writer.writerow({
                        'prediction': pred,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h
                    })


