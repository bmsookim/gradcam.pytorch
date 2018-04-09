Cell Detector module
================================================================================================
Cell detector module of CellNet

## Unsupervised Detection
In real-world medical data, localization labels are much harder to obtain than classification label.
Moreover, compared with classification labels which only needs a single class category for each image,
localization needs much more careful labelling including the bounding box regression answers and classification answers.

This module is designed to overcome this issue, by performing accurate localization through solely trained classification wieghts.
This is done by implementing an activation map, processed with a method called [Grad-CAM](http://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf).

## Grad-CAM++
The basic idea of Grad-CAM++ is ....
[TODO]

## Basic Setups
Open [config.py](./config.py), and edit the lines below to your data directory.

```bash
name = [:The name of your dataset that you trained on module 3 (classifier)]
data_base = [:dir to your original dataset]
aug_base =  [:dir to your actually trained dataset]
```

For training, your data file system should be in the following hierarchy.
Organizing codes for your data into the given requirements will be provided in the [preprocessor module](../1_preprocessor)

```bash
[:data file name]

    |-train
        |-[:class 0]
        |-[:class 1]
        |-[:class 2]
        ...
        |-[:class n]
    |-val
        |-[:class 0]
        |-[:class 1]
        |-[:class 2]
        ...
        |-[:class n]
```

## How to run
After you have cloned the repository, you can train the dataset by running the script below.

You can set the dimension of the additional layer in [config.py](./config.py)

```bash
# grad-cam exploits
python launch_model --net_type [:net_type] --depth [:depth]

# For example, for the resnet-50 model I've trained, type
python launch_model --net_type resnet --depth 50
```

## Test out various networks
Before testing out the networks, make sure that you have a trained weight obtained in the checkpoint file of the [classifier module](../3_classifier)

Supporting networks
- AlexNet [:TODO]
- VGGNet  [:TODO]
- ResNet

## Results

- Original Image

![alt_tag](../imgs/input.png)

- Grad-CAM Image

![alt_tag](../imgs/heatmap_out.png)

- Inference & IoU with ground truth

![alt_tag](../imgs/output.png)
