import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_lobe import lobe_dataset
from utils import test_single_volume

from network_configs.PnPNet.unet import network as network

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/dssg/home/acct-medcb/medcb-cb1/YX/medical data/verse19', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--edge_path', type=str,
                    default='/dssg/home/acct-medcb/medcb-cb1/YX/medical data/verse19', help='edge dir for train and val data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='/dssg/home/acct-medcb/medcb-cb1/YX/shape_prior/SPM1/verse19/3dunet/lists/lists_Synapse', help='list dir')
parser.add_argument('--max_iterations', type=int,default=15000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=1500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,                     # 24
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=[16, 336, 448], help='input patch size of network input')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--is_savenii', action="store_true", default='True', help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-2, help='segmentation network learning rate')       # 0.01
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()



def inference(args, model0, model1, model2, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", num_classes=args.num_classes, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model0.eval()
    model1.eval()
    model2.eval()
    metric_list = 0.0
    index = 0.0
    binary_metric_list = 0.0
    total_time = 0.0
    ave_dice = 0
    ave_hd = 0
    ave_assd = 0
    dice_case = []
    hd_case = []
    assd_case = []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        l, h, w = sampled_batch["image"].size()[1:]
        image, label, case_name, origin, spacing = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0], sampled_batch["origin"], sampled_batch["spacing"]
        metric_i, index_i, binary_metric_i, time_i, c_matrix_i = test_single_volume(image, label, model0, model1, model2, classes=args.num_classes, patch_size=args.img_size,         # maybe changed into the x and y dimensions
                                      test_save_path=test_save_path, case=case_name, origin=origin, spacing=spacing)
        
        
        binary_metric_list += np.array(binary_metric_i)
        num = np.count_nonzero(index_i)
        mean_dice = np.sum(metric_i, axis=0)[0] / num
        mean_hd = np.sum(metric_i, axis=0)[1] / num
        mean_assd = np.sum(metric_i, axis=0)[2] / num
        logging.info('idx: %d, case: %s, mean_dice: %f, mean_hd95: %f, mean assd: %f' % (i_batch, case_name, mean_dice, mean_hd, mean_assd))
        logging.info('idx: %d, case: %s, binary dice: %f, binary hd95: %f, binary assd: %f' % (i_batch, case_name, binary_metric_i[0], binary_metric_i[1], binary_metric_i[2]))
        ave_dice = ave_dice + mean_dice
        ave_hd = ave_hd + mean_hd
        ave_assd = ave_assd + mean_assd
        dice_case.append(mean_dice)
        hd_case.append(mean_hd)
        assd_case.append(mean_assd)
        index += index_i
        metric_list += metric_i
        total_time += time_i
        
        logging.info('confusion matrix: {}'.format(c_matrix_i))

        
        
    # mean for all cases, but a single class
    for i in range(1, args.num_classes):
        metric_list[i, :] = metric_list[i, :] / index[i]
        logging.info('Mean class: %d, mean_dice %f, mean_hd95 %f, mean_assd %f' % (i, metric_list[i][0], metric_list[i][1], metric_list[i][2]))
        
    binary_metric_list = binary_metric_list / len(db_test)
    mean_time = total_time / len(db_test)
    # mean for all classes and all cases
    performance = np.sum(metric_list, axis=0)[0] / (args.num_classes - 1)
    mean_hd95 = np.sum(metric_list, axis=0)[1] / (args.num_classes - 1)
    mean_ASSD = np.sum(metric_list, axis=0)[2] / (args.num_classes - 1)
    logging.info('Testing performance on classes: mean_dice : %f mean_hd95 : %f, mean_assd: %f' % (performance, mean_hd95, mean_ASSD))
    c_dice = np.sum(metric_list[1:3, :], axis=0)[0] / 2
    c_hd = np.sum(metric_list[1:3, :], axis=0)[1] / 2
    c_assd = np.sum(metric_list[1:3, :], axis=0)[2] / 2
    t_dice = np.sum(metric_list[3:6, :], axis=0)[0] / 3
    t_hd = np.sum(metric_list[3:6, :], axis=0)[1] / 3
    t_assd = np.sum(metric_list[3:6, :], axis=0)[2] / 3
    
    logging.info('Testing performance on classes: left_dice : %f left_hd95 : %f left_assd: %f' % (c_dice, c_hd, c_assd))
    logging.info('Testing performance on classes: right_dice : %f right_hd95 : %f, right assd: %f' % (t_dice, t_hd, t_assd))

    
    logging.info('Testing performance on classes: binary mean_dice : %f binary mean_hd95 : %f binary mean_assd : %f' % (binary_metric_list[0], binary_metric_list[1], binary_metric_list[2]))
    ave_dice = ave_dice / len(db_test)
    ave_hd = ave_hd / len(db_test)
    ave_assd = ave_assd / len(db_test)
    logging.info('Testing performance on cases: mean_dice : %f mean_hd95 : %f, mean_assd: %f' % (ave_dice, ave_hd, ave_assd))

    std_dice = np.std(dice_case)
    std_hd = np.std(hd_case)
    std_assd = np.std(assd_case)
    logging.info('Testing performance on cases: std_dice : %f std_hd95 : %f std_assd : %f' % (std_dice, std_hd, std_assd))
    
    median_dice = np.median(dice_case)
    median_hd = np.median(hd_case)
    median_assd = np.median(assd_case)
    logging.info('Testing performance on cases: median_dice : %f median_hd95 : %f median_assd : %f' % (median_dice, median_hd, median_assd))
    
    

    logging.info('Testing time in best val model: %f' % (mean_time))
    return "Testing Finished!"
    


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'lobe': {
            'Dataset': lobe_dataset,
            'volume_path': '/lustre/home/acct-eeyj/eeyj-wr/youxin/medical_dataset/lung_lobe/luna',
            'list_dir': '/lustre/home/acct-eeyj/eeyj-wr/youxin/uncertain_boundary/lung_lobe/clean_data/3dunet/3dunet/lists/lists_Synapse',
            'num_classes': 6,
            'z_spacing': 1,
        },
    }


    CONFIGS_ViT_seg = {
        'ViT-B_16': configs.get_b16_config(),
        'ViT-B_32': configs.get_b32_config(),
        'ViT-L_16': configs.get_l16_config(),
        'ViT-L_32': configs.get_l32_config(),
        'ViT-H_14': configs.get_h14_config(),
        'R50-ViT-B_16': configs.get_r50_b16_config(),
        'R50-ViT-L_16': configs.get_r50_l16_config(),
        'testing': configs.get_testing(),
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # lr
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    if dataset_name == 'Synapse':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 100000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path

    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.batch_size = args.batch_size
    # number of patches
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size[0] / args.vit_patches_size), int(args.img_size[1] / args.vit_patches_size), int(args.img_size[2] / args.vit_patches_size))
    ###
    config_vit.n_patches = int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    config_vit.n_patches = int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    config_vit.h = int(args.img_size[0] / args.vit_patches_size)
    config_vit.w = int(args.img_size[1] / args.vit_patches_size)
    config_vit.l = int(args.img_size[2] / args.vit_patches_size)

    
    net0 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    net1 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    net2 = network(in_channel=3, out_channel=args.num_classes, training=False, config=config_vit).cuda()
    
    if args.n_gpu > 1:
        net0 = nn.DataParallel(net0)
        net1 = nn.DataParallel(net1)
        net2 = nn.DataParallel(net2)

    snapshot0 = os.path.join(snapshot_path, 'best_model.pth')
    snapshot1 = os.path.join(snapshot_path, 'epoch_1500.pth')
    snapshot2 = os.path.join(snapshot_path, 'epoch_1450.pth')
    
    print(snapshot0)
    if not os.path.exists(snapshot0): snapshot0 = snapshot0.replace('best_model', 'epoch_'+str(args.max_epochs))
    print(snapshot0)
    
    net0.load_state_dict(torch.load(snapshot0))
    net1.load_state_dict(torch.load(snapshot1))
    net2.load_state_dict(torch.load(snapshot2))
    
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net0, net1, net2, test_save_path)


