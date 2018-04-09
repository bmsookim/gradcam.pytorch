# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Preprocessing Implementation
#
# Module : 1_preprocessor
# Description : main.py
# The main code for data mangement.
# ***********************************************************

import cv2
import os
import sys
import file_function as ff
import config as cf

def print_menu():
    print("\nSelect a mode by its name or number.\nType [exit] to terminate.\n")
    print("################## [ Options ] ###########################")
    print("# Mode 1 'print'  : Print names of image data file")
    print("# Mode 2 'read'   : [original/resized] Read names data")
    print("# Mode 3 'resize' : [target_size]  Resize & Orgnaize data")
    print("# Mode 4 'split'  : Create a train-validation split of data")
    print("# Mode 5 'count'  : Check the distribution of raw data")
    print("# Mode 6 'check'  : Check the distribution of train/val split")
    print("# Mode 7 'aug'    : Augment the training data samples")
    print("# Mode 8 'meanstd': Return the meanstd value of the training set")
    print("# Mode 9 'test'   : Preprocess the test data samples")
    print("##########################################################")


if __name__ == "__main__":
    while(1):
        print_menu()
        mode = raw_input('\nEnter mode name : ')

        ##############################################
        # @ Module 1 : Print names of image data file
        if (mode == 'print' or mode == '1'):
            ff.print_all_imgs(cf.data_base)

        #############################################
        # @ Module 2 : Read all images
        elif (mode == 'read' or mode == '2'):
            path = raw_input('Enter [original/resized] : ')
            if (not path in ['original', 'resized']):
                print("[Error] : Please define the mode between [original/resized].")
            else:
                if(path == 'original'):
                    ff.read_all_imgs(cf.data_base)
                elif(path == 'resized'):
                    ff.read_all_imgs(cf.resize_dir)

        #############################################
        # @ Module 3 : Resize and check images
        elif (mode == 'resize' or mode == '3'):
            ff.check_and_mkdir(cf.resize_base)
            target_size = int(raw_input('Enter size : '))
            ff.resize_images(cf.data_base, cf.resize_dir, target_size)
            # ff.resize_and_contrast(cf.data_base, cf.resize_dir, target_size)

        #############################################
        # @ Module 4 : Train-Validation split
        elif (mode == 'split' or mode == '4'):
            ff.check_and_mkdir(cf.split_base)
            split_dir = ff.create_train_val_split(cf.resize_dir, cf.split_dir)
            print("Train-Validation split directory = " + cf.split_dir)

        ############################################
        # @ Module 5 : Check the dataset
        elif (mode == 'count' or mode == '5'):
            print("| " + cf.resize_dir.split("/")[-1] + " dataset : ")
            ff.count_each_class(cf.resize_dir)
        elif (mode == 'check' or mode == '6'):
            ff.get_split_info(cf.split_dir)

        ############################################
        # @ Module 6 : Training data augmentation
        elif (mode == 'aug' or mode == '7'):
            #if (len(sys.argv) < 3):
            #    print("[Error] : Please define size in the second arguement.")
            #else:
            ff.aug_train(cf.split_dir, cf.rotate_mode)#sys.argv[2])

        #############################################
        # @ Module 7 : Retrieve Training data meanstd
        elif (mode == 'meanstd' or mode == '8'):
            mean = ff.train_mean(cf.split_dir)
            std = ff.train_std(cf.split_dir, mean)

            print("mean = " + str(mean))
            print("std = " + str(std))

        #############################################
        # @ Module 8 : Preprocess test data
        elif (mode == 'test' or mode == '9'):
            # [TO DO] : Implement Test Preprocessor
            print("[TO DO] : Implement Test Preprocessor")

        #############################################
        elif (mode == 'exit'):
            print("\nGood Bye!\n")
            sys.exit(0)
        else:
            print("[Error] : Wrong input in 'mode name', please enter again.")
        #############################################
