import os
import cv2
import sys
import random
import numpy as np
import config as cf
import file_function as ff
import math

def generate_padding_image(image, stepSize, windowSize):
    border_x = int(stepSize - ((image.shape[0]-windowSize)%stepSize))
    border_y = int(stepSize - ((image.shape[1]-windowSize)%stepSize))

    pad_image = cv2.copyMakeBorder(image, 0, border_x, 0, border_y, cv2.BORDER_CONSTANT, value=[255,255,255])

    return pad_image

def random_crop(image, dim):
    if(len(image.shape)):
        W, H, D = image.shape
        w, h, d = dim
    else:
        W, H = image.shape
        w, h = size

    left, top = np.random.randint(W-w+1), np.random.randint(H-h+1)

    return image[left:left+w, top:top+h]

def change_img_dir(in_dir, out_dir):
    img_cnt = 0
    ff.check_and_mkdir(out_dir)

    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            if ff.is_image(f):
                file_path = os.path.join(subdir, f)
                img = cv2.imread(file_path)

                img_cnt += 1
                out_path = os.path.join(out_dir, "RBC_%d.png" %img_cnt)
                print("Saving %s ..." %out_path)
                cv2.imwrite(out_path, img)

def save_random_crop(in_dir, out_dir):
    windowSize = random.randint(80, 100)
    img_cnt = 0
    subdir_cnt = 0
    ff.check_and_mkdir(out_dir)

    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            if ff.is_image(f):
                file_path = os.path.join(subdir, f)
                image = cv2.imread(file_path)

                print("Saving from %s ... " %f)
                subdir_cnt += 1

                for i in range(5):
                    crop = random_crop(image, (windowSize, windowSize, 3))
                    img_cnt += 1
                    save_dir = os.path.join(out_dir, "Crop_%d.png" %img_cnt)
                    print("\tSaving %s ..." %save_dir)
                    cv2.imwrite(save_dir, crop)

    print(subdir_cnt)

def save_sliding_windows(in_dir, out_dir, stepSize, windowSize):
    img_cnt = 0
    ff.check_and_mkdir(out_dir)

    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            if ff.is_image(f):
                file_path = os.path.join(subdir, f)
                image = cv2.imread(file_path)
                windowSize = random.randint(80, 100)
                #print(image.shape)

                #image = generate_padding_image(image, stepSize, windowSize)
                #print(image.shape)

                for x in range(0, image.shape[0], stepSize):
                    for y in range(0, image.shape[0], stepSize):
                        if (x+windowSize <= image.shape[0] and y+windowSize <= image.shape[0]):
                            save_img = image[x:x+windowSize, y:y+windowSize, :]

                            if (save_img.shape[0] == save_img.shape[1]):
                                img_cnt += 1
                                save_dir = os.path.join(out_dir, "Crop_%d.png" %img_cnt)
                                print("Saving %s ..." %save_dir)
                                cv2.imwrite(save_dir, save_img)

def pick_random_slides(in_dir, out_dir):
    ff.check_and_mkdir(out_dir)

    path, dirs, files = os.walk(in_dir).next()
    file_count = len(files)

    lst = random.sample(range(0, file_count), (1000-377+25))

    for file_num in lst:
        file_path = os.path.join(in_dir, "Crop_%d.png" %file_num)
        img = cv2.imread(file_path)

        save_path = os.path.join(out_dir, "Crop_%d.png" %file_num)
        print(save_path)
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    # for atypical sets
    #in_dir = "/home/bumsoo/Junhyun/atypical"
    #out_dir = "/home/bumsoo/Data/resized/RBC/RBC"

    # for crop
    in_dir = "/home/bumsoo/Junhyun/smudge"; out_dir = "/home/bumsoo/Data/resized/Cropped" #os.path.join(cf.resized_base, "RBC")

    # for random pick
    #in_dir = "/home/bumsoo/Data/resized/Cropped"; out_dir = "/home/bumsoo/Data/resized/RBC/RBC"

    #change_img_dir(in_dir, out_dir)
    #save_sliding_windows(in_dir, out_dir, stepSize=50, windowSize=100)
    save_random_crop(in_dir, out_dir)
    #pick_random_slides(in_dir, out_dir)
