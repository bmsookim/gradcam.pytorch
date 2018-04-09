############# Configuration file #############

# Name of dataset
name = 'MICCAI_TRAIN'
#name = 'WBC_NH'
#name = 'WBC_LH'

# Base directory for data formats
data_base = '/mnt/datasets/' + name

# Base directory for augmented data formats
resize_base = '/home/bumsoo/Data/resized/'
split_base = '/home/bumsoo/Data/split/'

# Directory for data formats
resize_dir = resize_base + name
split_dir = split_base + name

# Train augmentation
rotate_mode = 'strict'

# Validation split
split = 'ratio' # [ratio/fix]
val_ratio = 0.2
val_num = 25
