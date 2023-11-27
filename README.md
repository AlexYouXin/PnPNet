# PnPNet

This repo is the official implementation for: [PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion](https://arxiv.org/abs/2303.17967)


# Dataset Link
[BraTS 2020: Multimodal Brain Tumor Segmentation Challenge 2020](https://www.med.upenn.edu/cbica/brats2020/data.html)  

[VerSe'19: Large Scale Vertebrae Segmentation Challenge](https://verse2019.grand-challenge.org/)  

[LA/LAA: To be released] 

# Preprocess
We follow the z-score normalization strategy in [nnUNet](https://github.com/MIC-DKFZ/nnUNet) to preprocess clean and fused lobe datasets, VerSe'19 and LA/LAA dataset.

# Requirements
* python 3.7  
* pytorch 1.8.0  
* torchvision 0.9.0  
* simpleitk 2.0.2
* monai 0.9.0

