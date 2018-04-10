############# Configuration file #############

# Name of dataset
name = 'Cat_vs_Dog'

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
split = 'fix' # [ratio/fix]
val_ratio = 0.2
val_num = 100
