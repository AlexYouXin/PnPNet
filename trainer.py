import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, print_network
from torchvision import transforms
from torch.cuda.amp import autocast as autocast


def train(model, writer, optimizer, dice_weight, ce_weight, args, epoch_num, trainloader, snapshot_path):
    num_classes = args.num_classes
    max_iterations = args.max_epochs * len(trainloader)
    max_epoch = args.max_epochs
    iter_num = epoch_num * len(trainloader)
    model.train()
    class_weights = torch.FloatTensor(ce_weight).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes, dice_weight)

    best_performance = 0.0

    mean_loss = 0
    mean_dice = 0
    mean_loss_ce = 0
    mean_intra_loss = 0
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        outputs, loss0 = model(image_batch, label_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:].long(), softmax=True)

        
        loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.1 * loss0
        dice = 1 - loss_dice
        mean_loss += loss
        mean_dice += dice
        mean_loss_ce += loss_ce
        mean_intra_loss += loss0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr_ = param_group['lr']
        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/train_total_loss', loss, iter_num)
        # writer.add_scalar('info/train_loss_ce', loss_ce, iter_num)
        writer.add_scalar('info/train_dice', dice, iter_num)

        if iter_num % 8 == 0:
            logging.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f, train intra loss: %f' % (
                epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item(), loss0.item()))

        
    # mean loss and mean dice
    mean_loss = float(mean_loss / len(trainloader))
    mean_dice = float(mean_dice / len(trainloader))
    mean_loss_ce = float(mean_loss_ce / len(trainloader))
    mean_intra_loss = float(mean_intra_loss / len(trainloader))
    writer.add_scalar('info/epoch_train_total_loss', mean_loss, epoch_num)
    writer.add_scalar('info/epoch_train_dice', mean_dice, epoch_num)
    writer.add_scalar('info/epoch_train_total_loss_ce', mean_loss_ce, epoch_num)
    # print('epoch :', epoch_num, 'Train Loss :', mean_loss, 'Train dice :', mean_dice)
    logging.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f, mean train intra loss : %f' % (epoch_num, mean_loss, mean_loss_ce, mean_dice, mean_intra_loss))

    save_interval = 50
    if (epoch_num + 1) >= 350 and (epoch_num + 1) % save_interval == 0:
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num + 1) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    return writer, mean_dice


def validation(model, writer, dice_weight, ce_weight, args, epoch_num, valloader, snapshot_path):
    model.eval()
    iter_num = epoch_num * len(valloader)
    num_classes = args.num_classes
    class_weights = torch.FloatTensor(ce_weight).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = DiceLoss(num_classes, dice_weight)
    mean_loss = 0
    mean_dice = 0
    mean_loss_ce = 0
    mean_intra_loss = 0
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs, loss0 = model(image_batch, label_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            dice = 1 - loss_dice
            loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.1 * loss0
            mean_loss += loss
            mean_dice += dice
            mean_loss_ce += loss_ce
            mean_intra_loss += loss0
            writer.add_scalar('info/val_total_loss', loss, iter_num)
            writer.add_scalar('info/val_dice', dice, iter_num)

            iter_num += 1

            if iter_num % 6 == 0:
                logging.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f, val intra loss: %f' % (
                    epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item(), loss0.item()))



    # mean loss and mean dice
    mean_loss = float(mean_loss / len(valloader))
    mean_dice = float(mean_dice / len(valloader))
    mean_loss_ce = float(mean_loss_ce / len(valloader))
    mean_intra_loss = float(mean_intra_loss / len(valloader))
    writer.add_scalar('info/epoch_val_total_loss', mean_loss, epoch_num)
    writer.add_scalar('info/epoch_val_dice', mean_dice, epoch_num)
    writer.add_scalar('info/epoch_val_total_loss_ce', mean_loss_ce, epoch_num)
    logging.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f, mean val intra loss : %f' % (epoch_num, mean_loss, mean_loss_ce, mean_dice, mean_intra_loss))

    return writer, mean_dice


def run_main(args, model, snapshot_path):
    from datasets.dataset_lobe import lobe_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    batch_size = args.batch_size * args.n_gpu
    db_train = lobe_dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train", num_classes=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size, mode = 'train')]))

    print("The length of train set is: {}".format(len(db_train)))
    db_val = lobe_dataset(base_dir=args.val_root_path, list_dir=args.list_dir, split="val", num_classes=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size, mode = 'val')]))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    parameter_num, net = print_network(model)
    logging.info("Total number of network parameters: {}".format(parameter_num))
    logging.info("network structure: {}".format(net))

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)          


    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    writer = SummaryWriter(snapshot_path + '/log')
    # load progress bar
    iterator = tqdm(range(max_epoch), ncols=70)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25, eta_min=5e-6)                             # warm up for the first 500 iterations, about 20 epochs, last_epoch表示使用这个optimalizer已经使用了多少epochs
    #  CosineAnnealingWarmRestarts
    # record best model with highest train and val dice
    highest_train_dice = 0
    epo_train = 1
    highest_val_dice = 0
    epo_val = 1
    # weights need to be changed
    dice_weight = [0.5] + [1.1] * 5
    ce_weight = [0.5] + [1.1] * 5
    ### train and validation
    for epoch_num in iterator:

        
        writer, train_dice = train(model, writer, optimizer, dice_weight, ce_weight, args, epoch_num, trainloader, snapshot_path)
        writer, val_dice = validation(model, writer, dice_weight, ce_weight, args, epoch_num, valloader, snapshot_path)
        if train_dice > highest_train_dice:
            highest_train_dice = train_dice
            epo_train = epoch_num + 1
        if val_dice > highest_val_dice:
            highest_val_dice = val_dice
            epo_val = epoch_num + 1
            save_mode_path = os.path.join(snapshot_path, 'best_model' + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save best model to {} at epoch {}, val dice: {}".format(save_mode_path, epo_val, val_dice))

        lr_scheduler.step()
        torch.cuda.empty_cache()

    logging.info('highest train dice: %f at epoch %d, highest val dice : %f at epoch %d' % (
        highest_train_dice, epo_train, highest_val_dice, epo_val))


    writer.close()
    return "Training and validation Finished!"