Cell Preprocessor module
================================================================================================
Cell Image Preprocessor module of CellNet

# Requirements
- python 2.7
- [OpenCV](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)

# Input directory
The input directory should be in the given format:
```bash

[:folder]
    |-[:class 0]
        |-[:img 0]
        |-[:img 1]
        |-[:img 2]
        ...
    |-[:class 1]
    |-[:class 2]
    ...
        ...
            ...

```

# Menu Options
If you run the program, you will meet a menu script that will help you through various processes.

```bash
$ python main.py

################## [ Options ] ###########################
# Mode 1 'print' : Print names of image data file
# Mode 2 'read'  : [original/aug] Read names data
# Mode 3 'resize': [target_size]  Resize & Orgnaize data
# Mode 4 'split' : Create a train-validation split of data
# Mode 5 'count' : Check the distribution of raw data
# Mode 6 'check' : Check the distribution of train/val split
# Mode 7 'aug'   : Augment the training data sample
# Mode 8 'exit'  : Terminate the program
##########################################################

Enter mode name : 

```

If you enter the mode name in the given line, the code will run the function that has been typed.

# Modules

## 1. print
```bash
Enter mode name : print
```
This module will print all the the file names of image related file formats(".jpg", ".png")

## 2. read
```bash
Enter mode name : read
```
This module will read all the images and print out the spacial dimension of image related files.

## 3. resize
```bash
Enter mode name : resize
```
This module will save all the resized images into your given directory

## 4. split
```bash
Enter mode name : split
```
This module will organize your input file directory into the following format.
You should manually set how much validation sets you want in your val class in val_num from [config.py](./config.py).

```bash
[:folder]
    |-train
        |-[:class 0]
            |-[:img 0]
            |-[:img 1]
            |-[:img 2]
            ...
        |-[:class 1]
        |-[:class 2]
        ...
            ...
                ...
    |-val
        |-[:class 0]
            |-[:img 0]
            |-[:img 1]
            |-[:img 2]
            ...
        |-[:class 1]
        |-[:class 2]
        ...
            ...
                ...

```

## 5. count
```bash
Enter mode name : count
```
This will count the number of images within each sub-categories in the data.
An example for the file directory after running module 5 (count) is as below.
```bash
$ Enter mode name : count

| Cat_vs_Dog dataset : 
        | cat       12500
        | dog       12500
```

## 6. check
```bash
Enter mode name : check
```
This will check how your train/validation split is consisted.
An example for the file directory after running module 4 (split) is as below.
```bash
$ Enter mode name : check

| train set : 
        | cat        100
        | dog        100
| val set : 
        | cat        100
        | dog        100

```

## 7. augmentation
```bash
Enter mode name : aug
```
This module will apply various image augmentations and enlarge your training set.
The input should be the splitted directory after running module 4 (split)


