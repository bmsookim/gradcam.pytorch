# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Grad CAM Implementation
#
# Description : misc_function.py
# The main code for grad-CAM image localization.
# ***********************************************************
import os
import cv2
import copy
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models

def preprocess_image(cv2im, resize_im=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2,0,1)

    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)

    im_as_var = Variable(im_as_ten, requires_grad = True)

    return im_as_var

def save_gradient_images(gradient, file_name):
    """
    @ func:
        exports the original gradient image.

    @ args:
        gradient : Numpy array of the gradient with shape (3, 224, 224)
        file_name : File name to be exported
    """

    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('./results', file_name+'.jpg')

    # Convert RGB to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)

def save_class_activation_on_image(org_img, activation_map, file_name):
    """
    @ func:
        Saves CAM(Class Activation Map) on the original image.

    @ args:
        org_img (PIL img): Original image
        activation_map : Numpy array of the activation map in grayscale (0~255)
        file_name : String for the file name of the exported image.
    """

    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)

    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    path_to_file = os.path.join('./results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)

    # Heatmap on picture
    org_img = cv2.resize(org_img, (224,224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255*img_with_heatmap))

def recreate_image(im_var):
    """
    @ func:
        Recreated image from a torch variable.

    @ args:
        im_var : Image to recreate

    @ return:
        recreated_im : Numpy array of the recreated img
    """

    reverse_mean = map(lambda x : -x, cf.mean)
    reverse_std = map(lambda x : 1/x, cf.std)

    recreated_im = copy.copy(im_as_var.data.numpy()[0])

    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]

    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)

    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im

def get_positive_negative_saliency(gradient): # What is saliency?
    """
    @ func:
        Generates positive and negative saliency maps based on the gradient

    @ args:
        gradient : Numpy array of the gradients to visualize

    @ returns:
        Positive and Negative Saliency Maps of the gradient
    """

    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())

    return pos_saliency, neg_saliency

