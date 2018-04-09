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

| CELL_PATCHES dataset : 
        | RBC_Target        157
        | WBC_Neutrophil_Band   200
        | RBC_Elliptocyte    78
        | RBC_Echinocyte    284
        | WBC_Neutrophil_Segmented   168
        | RBC_TearDrop       17
        | WBC_Basophil      118
        | WBC_Monocyte      115
        | RBC_Spherocyte    210
        | WBC_Eosinophil     75
        | WBC_Metamyelocyte   221
        | WBC_Smudge        122
        | RBC_Stomatocyte    44
        | WBC_Myelocyte     149
        | RBC_Acanthocyte   214
        | RBC_Schistocyte    75
        | WBC_Lymphocyte_atypical   114
        | WBC_Lymphocyte    127
        | RBC_Normal        168
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
        | RBC_Target        147
        | WBC_Neutrophil_Band   190
        | RBC_Elliptocyte    68
        | RBC_Echinocyte    274
        | WBC_Neutrophil_Segmented   158
        | RBC_TearDrop        7
        | WBC_Basophil      108
        | WBC_Monocyte      105
        | RBC_Spherocyte    200
        | WBC_Eosinophil     65
        | WBC_Metamyelocyte   211
        | WBC_Smudge        112
        | RBC_Stomatocyte    34
        | WBC_Myelocyte     139
        | RBC_Acanthocyte   204
        | RBC_Schistocyte    65
        | WBC_Lymphocyte_atypical   104
        | WBC_Lymphocyte    117
        | RBC_Normal        158
| val set : 
        | RBC_Target         10
        | WBC_Neutrophil_Band    10
        | RBC_Elliptocyte    10
        | RBC_Echinocyte     10
        | WBC_Neutrophil_Segmented    10
        | RBC_TearDrop       10
        | WBC_Basophil       10
        | WBC_Monocyte       10
        | RBC_Spherocyte     10
        | WBC_Eosinophil     10
        | WBC_Metamyelocyte    10
        | WBC_Smudge         10
        | RBC_Stomatocyte    10
        | WBC_Myelocyte      10
        | RBC_Acanthocyte    10
        | RBC_Schistocyte    10
        | WBC_Lymphocyte_atypical    10
        | WBC_Lymphocyte     10
        | RBC_Normal         10

```

## 7. augmentation
```bash
Enter mode name : aug
```
This module will apply various image augmentations and enlarge your training set.
The input should be the splitted directory after running module 4 (split)


