import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import h5py
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import SimpleITK as sitk
from augmentation import intensity_shift, intensity_scale, random_rotate, flip_xz_yz



class verse_config():

    object = 'vertebrae'

    # settings
    batch_size = 2
    gpu_num = 2
    benchmark = True
    manualseed = 1234

    # data
    input_size = [128, 160, 96]
    train_data_num = 80
    eval_data_num = 40
    test_data_num = 40
    in_channel = 1
    class_num = 26

    # data augmentation
    augmentations = ['intensity_shift', 'intensity_scale', 'random_rotate', 'flip_xz_yz']

    # data split
    split_file = '../split_list/verse19'

    # way of patch cropping
    # label: Ground truth
    index = np.nonzero(label)
    index = np.transpose(index)

    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])


    z_middle = np.int((z_min + z_max) / 2)
    y_middle = np.int((y_min + y_max) / 2)
    x_middle = np.int((x_min + x_max) / 2)

    Delta_z = np.int((z_max - z_min) / 3)
    Delta_y = np.int((y_max - y_min) / 4)
    Delta_x = np.int((x_max - x_min) / 4)

    y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
    x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
    
    thre = z_min + Delta_z + np.int(self.output_size[0] / 2)
    if z_middle > thre:
        delta_Z = z_middle - z_min - np.int(self.output_size[0] / 4)
        z_random = random.randint(z_middle - delta_Z, z_middle + delta_Z)
    else:
        z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)

    # model
    optimizer = 'optim.AdamW'
    scheduler = 'optim.lr_scheduler.CosineAnnealingWarmRestarts'
    lr = 5e-4
    train_epochs = 1000
    num_workers = 8

    loss = 'dice' + 'ce'




class lobe_config():

    object = 'lung lobe'

    # settings
    batch_size = 1
    gpu_num = 2
    benchmark = True
    manualseed = 1234

    # data
    input_size = [16, 336, 448]
    train_data_num = 35
    eval_data_num = 6
    test_data_num = 10
    in_channel = 1
    class_num = 6

    # data augmentation
    augmentations = ['intensity_shift', 'intensity_scale', 'random_rotate']

    # data split
    split_file = '../split_list/clean_lung_lobe'

    # way of patch cropping
    # label: Ground truth
    index = np.nonzero(label)
    index = np.transpose(index)

    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])

    z_middle = np.int((z_min + z_max) / 2)
    y_middle = np.int((y_min + y_max) / 2)
    x_middle = np.int((x_min + x_max) / 2)

    if random.random() > 0.3:
        Delta_z = np.int((z_max - z_min) / 3)  # 3
        Delta_y = np.int((y_max - y_min) / 8)  # 8
        Delta_x = np.int((x_max - x_min) / 8)  # 8
    
    else:
        Delta_z = np.int((z_max - z_min) / 2) + self.output_size[0]
        Delta_y = np.int((y_max - y_min) / 8)
        Delta_x = np.int((x_max - x_min) / 8)

    z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
    y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
    x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)


    # model
    optimizer = 'optim.AdamW'
    scheduler = 'optim.lr_scheduler.CosineAnnealingWarmRestarts'
    lr = 5e-4
    train_epochs = 1500
    num_workers = 8

    loss = 'dice' + 'ce'



class LAA_config():

    object = 'LA & LAA'

    # settings
    batch_size = 1
    gpu_num = 2
    benchmark = True
    manualseed = 1234

    # data
    input_size = [160, 160, 192]
    train_data_num = 70
    eval_data_num = 25
    test_data_num = 35
    in_channel = 1
    class_num = 3

    # data augmentation
    augmentations = ['intensity_shift', 'intensity_scale', 'random_rotate']

    # data split
    split_file = '../split_list/LAA'

    # way of patch cropping
    # label: Ground truth
    index = np.nonzero(label)
    index = np.transpose(index)


    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])

    patch_z = np.int(self.output_size[0] / 2 * 1.25)
    patch_y = np.int(self.output_size[1] / 2)
    patch_x = np.int(self.output_size[2] / 2 * 1.25)

    z_middle = np.int((z_min + z_max) / 2)
    y_middle = np.int((y_min + y_max) / 2)
    x_middle = np.int((x_min + x_max) / 2)

    Delta_z = np.int((z_max - z_min) / 3)
    Delta_y = np.int((y_max - y_min) / 3)
    Delta_x = np.int((x_max - x_min) / 3)

    if random.random() > 0.2:

        z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)

    else:
        z_random = random.randint(z_middle - Delta_z - patch_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y - patch_y, y_middle + Delta_y + patch_y)
        x_random = random.randint(x_middle - Delta_x - patch_x, x_middle + Delta_x + patch_x)
        

    # model
    optimizer = 'optim.AdamW'
    scheduler = 'optim.lr_scheduler.CosineAnnealingWarmRestarts'
    lr = 5e-4
    train_epochs = 1500
    num_workers = 8

    loss = 'dice' + 'ce'


if __name__ == '__main__':
    verse_config = verse_config()
    lobe_config = lobe_config()
    LAA_config = LAA_config()
