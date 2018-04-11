#################### Configuration File ####################

# Base directory for data formats
name = 'Cat_vs_Dog'

# Specific directories
data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'

data_base = data_dir + name
aug_base = aug_dir + name
test_dir = '/home/bumsoo/Data/test/' + name

# model directory
model_dir = '../2_classifier/checkpoints'

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.42352142932368259, 0.46167925008138017, 0.49023161345837163]
std = [0.22595048333178538, 0.22503028985594206, 0.23220585942785971]
