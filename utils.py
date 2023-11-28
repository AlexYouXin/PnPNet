import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import time
from sklearn.metrics import confusion_matrix

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    # print('Total number of parameters: %d' % num_params)
    return num_params, net

class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        # print(intersect.item())
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # print(loss.item())
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        weight = self.weight

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])

            class_wise_dice.append(1.0 - dice_loss.item())
            loss += dice_loss * weight[i]

        return loss / self.n_classes

def calculate_metric_percase(pred, gt, space):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=space)
        assd = metric.binary.assd(pred, gt, voxelspacing=space)
        return dice, hd95, assd

    else:
        return 0, 30, 30




# slide window, with z axis overlap
# padding to the bottom and top part
# slide from top to bottom

def test_single_volume(image_, label_, net0, net1, net2, classes, patch_size, test_save_path=None, case=None, origin=None, spacing=None):   # patch_size: [256, 256]
    image_, label_ = image_.squeeze(0).cpu().detach().numpy(), label_.squeeze(0).cpu().detach().numpy()
    edge = 3
    # preprocess
    label_[label_ < 0.5] = 0.0  # maybe some voxels is a minus value
    label_[label_ > 5.5] = 0.0
    
    # label_ = np.round(label_)
    
    print('previous image shape: ', image_.shape[0], image_.shape[1], image_.shape[2])
    
    min_value = np.min(image_)


    
    image = image_.copy()
    label = label_.copy()
    
    
    index = np.nonzero(label)
    index = np.transpose(index)
    z_min = np.min(index[:, 0])
    z_max = np.max(index[:, 0])
    y_min = np.min(index[:, 1])
    y_max = np.max(index[:, 1])
    x_min = np.min(index[:, 2])
    x_max = np.max(index[:, 2])
    
    # z padding
    image = np.pad(image, ((np.int(patch_size[0]/8), np.int(patch_size[0]/8)), (0, 0), (0, 0)), 'constant', constant_values=min_value)
    label = np.pad(label, ((np.int(patch_size[0]/8), np.int(patch_size[0]/8)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # y padding
    image = np.pad(image, ((0, 0), (np.int(patch_size[1]/4), np.int(patch_size[1]/4)), (0, 0)), 'constant', constant_values=min_value)
    label = np.pad(label, ((0, 0), (np.int(patch_size[1]/4), np.int(patch_size[1]/4)), (0, 0)), 'constant', constant_values=0)
    # x padding
    image = np.pad(image, ((0, 0), (0, 0), (np.int(patch_size[2]/8), np.int(patch_size[2]/8))), 'constant', constant_values=min_value)
    label = np.pad(label, ((0, 0), (0, 0), (np.int(patch_size[2]/8), np.int(patch_size[2]/8))), 'constant', constant_values=0)
    

    image = image[z_min: z_max + np.int(patch_size[0]/4), y_min: y_max + np.int(patch_size[1]/2), x_min: x_max + np.int(patch_size[2]/4)]    
    label = label[z_min: z_max + np.int(patch_size[0]/4), y_min: y_max + np.int(patch_size[1]/2), x_min: x_max + np.int(patch_size[2]/4)]
    
    print('cropped image shape: ', image.shape[0], image.shape[1], image.shape[2])
    z, y, x = image.shape[0], image.shape[1], image.shape[2]
    
    step_size_z = np.int(patch_size[0]/2)
    step_size_y = np.int(patch_size[1]/8)
    step_size_x = np.int(patch_size[2]/8)
    
    
    if len(image.shape) == 3:


        z_num = np.ceil(image.shape[0] / step_size_z).astype(int)
        y_num = np.ceil(image.shape[1] / step_size_y).astype(int)
        x_num = np.ceil(image.shape[2] / step_size_x).astype(int)

        # add padding to size： n * step
        delta_z = np.int(z_num * step_size_z - image.shape[0])
        delta_y = np.int(y_num * step_size_y - image.shape[1])
        delta_x = np.int(x_num * step_size_x - image.shape[2])

        # z_padding
        if delta_z % 2 == 0:
            delta_z_d = np.int(delta_z / 2)
            delta_z_u = np.int(delta_z / 2)
        else:
            delta_z_d = np.int(delta_z / 2)
            delta_z_u = np.int(delta_z / 2) + 1
        image = np.pad(image, ((delta_z_d, delta_z_u), (0, 0), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((delta_z_d, delta_z_u), (0, 0), (0, 0)), 'constant', constant_values=0.0)
        # y_padding
        if delta_y % 2 == 0:
            delta_y_d = np.int(delta_y / 2)
            delta_y_u = np.int(delta_y / 2)
        else:
            delta_y_d = np.int(delta_y / 2)
            delta_y_u = np.int(delta_y / 2) + 1
        image = np.pad(image, ((0, 0), (delta_y_d, delta_y_u), (0, 0)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (delta_y_d, delta_y_u), (0, 0)), 'constant', constant_values=0.0)
        # x_padding
        if delta_x % 2 == 0:
            delta_x_d = np.int(delta_x / 2)
            delta_x_u = np.int(delta_x / 2)
        else:
            delta_x_d = np.int(delta_x / 2)
            delta_x_u = np.int(delta_x / 2) + 1
        image = np.pad(image, ((0, 0), (0, 0), (delta_x_d, delta_x_u)), 'constant', constant_values=min_value)
        label = np.pad(label, ((0, 0), (0, 0), (delta_x_d, delta_x_u)), 'constant', constant_values=0.0)
        
        print('padding image shape:', image.shape[0], image.shape[1], image.shape[2])
        
        
        prediction = np.zeros_like(label)

        z_num = np.int((image.shape[0] - patch_size[0]) / step_size_z) + 1
        y_num = np.int((image.shape[1] - patch_size[1]) / step_size_y) + 1
        x_num = np.int((image.shape[2] - patch_size[2]) / step_size_x) + 1

        ######
        torch.cuda.synchronize()
        start_time = time.time()

        ######
        torch.cuda.synchronize()
        start_time = time.time()


        with torch.no_grad():
            pred = np.zeros((classes, image.shape[0], image.shape[1], image.shape[2]))
            ###########
            for h in range(z_num):
                for r in range(y_num):
                    for c in range(x_num):
                        # numpy to tensor
                        input = torch.from_numpy(image[h * step_size_z: h * step_size_z + patch_size[0],
                                                 r * step_size_y: r * step_size_y + patch_size[1],
                                                 c * step_size_x: c * step_size_x + patch_size[2]]).unsqueeze(0).unsqueeze(
                            0).float().cuda()
                        label_slice = torch.from_numpy(label[h * step_size_z: h * step_size_z + patch_size[0],
                                                 r * step_size_y: r * step_size_y + patch_size[1],
                                                 c * step_size_x: c * step_size_x + patch_size[2]]).unsqueeze(
                            0).float().cuda()

                        outputs0, _ = net0(input, label_slice)
                        outputs1, _ = net1(input, label_slice)
                        outputs2, _ = net2(input, label_slice)

                        outputs0 = torch.softmax(outputs0, dim=1).squeeze(0)
                        outputs1 = torch.softmax(outputs1, dim=1).squeeze(0)
                        outputs2 = torch.softmax(outputs2, dim=1).squeeze(0)


                        
                        outputs = outputs0 + outputs1 + outputs2
                        outputs = outputs.cpu().detach().numpy()

                        pred[:, h * step_size_z: h * step_size_z + patch_size[0],
                                                 r * step_size_y: r * step_size_y + patch_size[1],
                                                 c * step_size_x: c * step_size_x + patch_size[2]] += outputs
                                                 
                        
            out = np.argmax(pred, axis=0)
            prediction = out
        torch.cuda.synchronize()
        end_time = time.time()
        
        time_cost = end_time - start_time

    # 2D data
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            
    image = image[delta_z_d: z + delta_z_d, delta_y_d: y + delta_y_d, delta_x_d: x + delta_x_d]
    prediction = prediction[delta_z_d: z + delta_z_d, delta_y_d: y + delta_y_d, delta_x_d: x + delta_x_d]
    label = label[delta_z_d: z + delta_z_d, delta_y_d: y + delta_y_d, delta_x_d: x + delta_x_d]
    index = np.nonzero(label)
    index = np.transpose(index)
    z_min_ = np.min(index[:, 0])
    z_max_ = np.max(index[:, 0])
    y_min_ = np.min(index[:, 1])
    y_max_ = np.max(index[:, 1])
    x_min_ = np.min(index[:, 2])
    x_max_ = np.max(index[:, 2])
    
    flatten_label = label.flatten()
    list_label = flatten_label.tolist()
    set_label = set(list_label)
    print('different values:', set_label)
    length = len(set_label)
    list_label_ = list(set_label)
    list_label_ = np.array(list_label_).astype(np.int)
    
    origin = origin.flatten()
    spacing = spacing.flatten()
    origin = origin.numpy()
    spacing = spacing.numpy()
    
    print('origin and spacing: ', origin, spacing)  
    # construct confusion matrix
    c_matrix = confusion_matrix(label.astype(np.int).reshape(-1), prediction.astype(np.int).reshape(-1))
    
    index = np.zeros(classes)
    metric_list = np.zeros((classes, 3))
    for i in range(1, length):
        metric_list[list_label_[i], :] = calculate_metric_percase(prediction[z_min_: z_max_, y_min_: y_max_, x_min_: x_max_] == list_label_[i], label[z_min_: z_max_, y_min_: y_max_, x_min_: x_max_] == list_label_[i], spacing)
        index[list_label_[i]] += 1
    

    binary_map = prediction[z_min_: z_max_, y_min_: y_max_, x_min_: x_max_].copy()
    binary_map[binary_map >= 1] = 1
    binary_map[binary_map < 1] = 0
    
    binary_label = label[z_min_: z_max_, y_min_: y_max_, x_min_: x_max_].copy()
    binary_label[binary_label >= 1] = 1
    binary_label[binary_label < 1] = 0
    
    binary_metric = calculate_metric_percase(binary_map, binary_label, spacing)

    prediction_ = np.zeros_like(label_)
    size_z, size_y, size_x = prediction.shape
    print('remove delta: ', prediction.shape)
    prediction_[z_min: z_max, y_min: y_max, x_min: x_max] = prediction[np.int(patch_size[0]/8): size_z - np.int(patch_size[0]/8), np.int(patch_size[1]/4): size_y - np.int(patch_size[1]/4), np.int(patch_size[2]/8): size_x - np.int(patch_size[2]/8)]
    print('prediction_ shape: ', prediction_.shape)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image_.astype(np.float32))            # .astype(np.float32)
        prd_itk = sitk.GetImageFromArray(prediction_.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label_.astype(np.float32))
        
        # origin and spacing:  [[ 168.12409973  172.32409668 -560.25      ]] [[0.67578101 0.67578101 5.        ]]    ->    not work
        # 
        img_itk.SetOrigin(origin)
        img_itk.SetSpacing(spacing)
        prd_itk.SetOrigin(origin)
        prd_itk.SetSpacing(spacing)
        lab_itk.SetOrigin(origin)
        lab_itk.SetSpacing(spacing)

        
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list, index, binary_metric, time_cost, c_matrix
