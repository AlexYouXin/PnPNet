# PnPNet

This repo is the official implementation for: PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion


# Network
![image](https://github.com/AlexYouXin/PnPNet/blob/main/network.png)

# Abstract
![image](https://github.com/AlexYouXin/PnPNet/blob/main/abstract.png)


# Dataset Link
[Pulmonary Lobe Dataset from LUNA16: Image](https://luna16.grand-challenge.org/)[ / Ground truth](https://github.com/deep-voxel/automatic_pulmonary_lobe_segmentation_using_deep_learning)

[COVID-19 CT Lung and Infection Segmentation Dataset](https://ieee-dataport.org/open-access/pulmonary-lobe-segmentation-covid-19-ct-scans)

[VerSe'19: Large Scale Vertebrae Segmentation Challenge](https://verse2019.grand-challenge.org/)  

LA/LAA: To be released

# Preprocess
We follow the z-score normalization strategy in [nnUNet](https://github.com/MIC-DKFZ/nnUNet) to preprocess clean and fused lobe datasets, VerSe'19 and LA/LAA dataset.

# Requirements
* python 3.7  
* pytorch 1.8.1
* torchvision 0.9.1 
* simpleitk 2.0.2
* monai 0.9.0
* medpy 0.4.0


# Usage
If you want to train the model from scratch, please follow the next steps.  
1. fix dataset settings according to ./config file.
2. confirm model settings according to ./network_configs file.
3. `python train.py`


# Citation
