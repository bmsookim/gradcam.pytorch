#################### Configuration File ####################

# Base directory for data formats
name = 'MICCAI_TRAIN'
data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'
test_dir = '/home/bumsoo/Data/test/'

data_base = data_dir + name
aug_base = aug_dir + name
test_base = test_dir + name

# model directory
model_dir = '../2_classifier/checkpoints'
test_dir = '/home/bumsoo/Data/test/MICCAI_TEST/TEST' # Number on the back

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.76937065622596712, 0.62102846073902673, 0.62280950464590923]
std = [0.20634491876598526, 0.27042467744347987, 0.24373346183373229]

# H1 meanstd (NH)
h1_mean = [0.72968820508788612, 0.52224113128933247, 0.54099372274735391]
h1_std = [0.208528564775461, 0.30056530735626585, 0.27138967466099473]

# H2 meanstd (LH)
h2_mean = [0.75086572277254915, 0.54344990735699861, 0.56189840210810549]
h2_std = [0.19795568869316291, 0.2989786366520818, 0.26473830163404605]
