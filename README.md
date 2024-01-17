# PnPNet

This repo is the official implementation for: PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion

# Boundary confusion
![image](https://github.com/AlexYouXin/PnPNet/blob/main/uncertain_type.png)

# Boundary refinement

<img src="https://github.com/AlexYouXin/PnPNet/blob/main/clean_lobe1.gif" width="330"><img src="https://github.com/AlexYouXin/PnPNet/blob/main/fused_lobe1.gif" width="330"><img src="https://github.com/AlexYouXin/PnPNet/blob/main/verse1.gif" width="330">


# Abstract
Precise boundary segmentation of volumetric images is a critical task for image-guided diagnosis and computer-assisted intervention, especially for confused boundaries in clinical practice. However, U-shape deep networks cannot effectively resolve this challenge due to the lack of boundary shape constraints. Besides, existing methods of refining boundaries overemphasize the slender structure, which results in the overfitting phenomenon due to neural networks' limited ability to model tiny structures. In this paper, we reconceptualize the mechanism of boundary generation by encompassing the dynamics of interactions with adjacent regions. Moreover, we propose a unified network termed PnPNet to model shape characteristics of confused boundaries. The core ingredients of PnPNet contain the pushing and pulling branches. Specifically, based on diffusion theory, we devise the semantic difference guidance module (SDM) from the pushing branch to squeeze the boundary region. Additionally, motivated by the K-means clustering algorithm, the class clustering module (CCM) from the pulling branch is introduced to stretch the boundary region. These two branches furnish two adversarial forces to enhance models' representation abilities for the boundary region, then promote models to output a more precise delineation of inter-class boundaries. We carry out quantitative and qualitative experiments on three challenging public datasets and one in-house dataset, containing three types of boundary confusion respectively. Experimental results demonstrate the superiority of PnPNet over other segmentation networks, especially on the evaluation metrics of Hausdorff Distance and Average Symmetric Surface Distance. Besides, pushing and pulling branches can serve as plug-and-play modules to refine classic U-shape baseline models.


# Pull-Push mechanism
![image](https://github.com/AlexYouXin/PnPNet/blob/main/pull-push.png)


# Network
![image](https://github.com/AlexYouXin/PnPNet/blob/main/network.png)


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
1. fix dataset settings according to configs/config_for_datasets.py.
2. confirm model settings according to the network_configs folder.
3. `python train.py`

# Acknowledgements
Part of codes are borrowed from other open-source github projects. Thanks for the codes from [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt), [nnUNet](https://github.com/MIC-DKFZ/nnUNet), [Swin UNETR](https://github.com/Project-MONAI/research-contributions) and [TransUNet](https://github.com/Beckschen/TransUNet).

# Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:  
```
@article{you2023pnpnet,
  title={PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion},
  author={You, Xin and Ding, Ming and Zhang, Minghui and Zhang, Hanxiao and Yu, Yi and Yang, Jie and Gu, Yun},
  journal={arXiv preprint arXiv:2312.08323},
  year={2023}
}
```
