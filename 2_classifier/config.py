#################### Configuration File ####################

# Base directory for data formats
name = 'Cat_vs_Dog'

data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'

# Databases for each formats
data_base = data_dir + name
aug_base = aug_dir + name
test_base = '../3_detector/results/inferenced/'
test_dir = aug_dir + name + '/val'

# model option
batch_size = 16
num_epochs = 40
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.76937065622596712, 0.62102846073902673, 0.62280950464590923]
std = [0.20634491876598526, 0.27042467744347987, 0.24373346183373229]
