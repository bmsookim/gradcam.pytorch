#################### Configuration File ####################

# Base directory for data formats
name = 'MICCAI_TRAIN'
#name = 'WBC_NH'
#name = 'WBC_LH'
#name = 'Only_WBC'

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

# NH meanstd
#mean = [0.72968820508788612, 0.52224113128933247, 0.54099372274735391]
#std = [0.208528564775461, 0.30056530735626585, 0.27138967466099473]

# LH meanstd
#mean = [0.7571956979879545, 0.55694333649406613, 0.56854173074367431]
#std = [0.20890086199641186, 0.31668580372231542, 0.28084878897340337]
