import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import SimpleITK as sitk
from augmentation import intensity_shift, intensity_scale, random_rotate



class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth

        index = np.nonzero(label)
        index = np.transpose(index)


        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        
        # middle point
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
        
        
        # random number of x, y, z
        z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
        

        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)


        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0]))
            image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)

            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)

            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)

            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]


        label = np.round(label)

        # data augmentation
        if self.mode == 'train':
            if random.random() > 0.5:
                image = intensity_shift(image)
            if random.random() > 0.5:
                image = intensity_scale(image)
            if random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)

        image = torch.from_numpy(image.astype(np.float)).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.float32)).float()


        sample = {'image': image, 'label': label.long()}
        return sample


class lobe_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, num_classes, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir + '/image', slice_name)
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir + '/label', slice_name)
            label = sitk.ReadImage(label_path)
            origin = np.array(image.GetOrigin())
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            

        elif self.split == "val":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir + '/image', slice_name)
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir + '/label', slice_name)
            label = sitk.ReadImage(label_path)
            origin = np.array(image.GetOrigin())
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir + '/image', slice_name)
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir + '/label', slice_name)
            label = sitk.ReadImage(label_path)
            
            origin = np.array(image.GetOrigin())
            spacing = np.array(image.GetSpacing())
            
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)

        label[label < 0.5] = 0.0
        label[label > 5.5] = 0.0
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')

        sample['origin'] = origin
        sample['spacing'] = spacing
        return sample
