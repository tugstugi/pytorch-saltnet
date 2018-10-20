#!/usr/bin/env python
"""Train UNet for the Kaggle TGS salt identification challenge: https://www.kaggle.com/c/tgs-salt-identification-challenge"""
__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import time
import argparse
import os
from datetime import datetime
import socket
from pathlib import Path
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import *
import torchvision.utils as vutils

from utils.metrics import calc_metric
from datasets import *
from transforms import *
import models
from utils import create_optimizer, choose_device, create_lr_scheduler

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')
parser.add_argument("--data-fold", default='fold0', choices=['fold{}'.format(s) for s in ['01'] + list(range(10))],
                    help='name of data split fold')
parser.add_argument("--batch-size", type=int, default=64, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=0.0001, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam', 'adamw'], default='sgd',
                    help='choices of optimization algorithms')
parser.add_argument('--fp16-loss-scale', default=None, type=float,
                    help='loss scale factor for mixed-precision training, 0 means dynamic loss scale')
parser.add_argument('--gradient-accumulation', type=int, default=1,
                    help='accumulate gradients over number of batches')
parser.add_argument("--learning-rate", type=float, default=0.01, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step', 'milestones', 'cos', 'findlr', 'noam', 'clr'],
                    default='step', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=15,
                    help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=100,
                    help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1,
                    help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--lr-scheduler-warmup", type=int, default=10,
                    help='The number of epochs to linearly increase the learning rate. (noam only)')
parser.add_argument("--max-epochs", type=int, default=350, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument('--resume-without-optimizer', action='store_true', help='resume but don\'t use optimizer state')
parser.add_argument("--model", choices=['unet', 'danet'], default='unet', help='model of NN')
parser.add_argument("--loss-on-center", action='store_true', help='loss on image without padding')
parser.add_argument("--drop-mask-threshold", type=int, default=0, help='drop problematic masks during training')
parser.add_argument("--debug", action='store_true', help='write debug images')
parser.add_argument("--disable-cutout", action='store_true', help='disable cutout data augmentation')
parser.add_argument('--pretrained', default='imagenet', choices=('imagenet', 'coco', 'oid'),
                    help='dataset name for pretrained model')
parser.add_argument("--basenet", choices=models.BASENET_CHOICES, default='resnet34', help='model of basenet')
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
default_log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
parser.add_argument('--log-dir', type=str, default=default_log_dir, help='Location to save logs and checkpoints')
parser.add_argument('--vtf', action='store_true', help='validation time flip augmentation')
parser.add_argument('--resize', action='store_true', help='resize to 128x128 instead of reflective padding')
args = parser.parse_args()

if args.resize:
    # if resize is used, loss on center doesn't make sense
    args.loss_on_center = False

device = choose_device(args.device)
use_gpu = device.type == 'cuda'

orig_img_size = 101
img_size = 128
padding = compute_padding(orig_img_size, orig_img_size, img_size)

geometric_transform_prob = 0.5 * 0.25
geometric_transform = Compose([RandomApply([CropAndRescale(max_scale=0.2)], p=geometric_transform_prob),
                               RandomApply([HorizontalShear(max_scale=0.07)], p=geometric_transform_prob),
                               RandomApply([Rotation(max_angle=15)], p=geometric_transform_prob),
                               RandomApply([ElasticDeformation(max_distort=0.15)], p=geometric_transform_prob)])
brightness_transform_prob = 0.5 * 0.33
brightness_transform = Compose([RandomApply([BrightnessShift(max_value=0.1)], p=brightness_transform_prob),
                                RandomApply([BrightnessScaling(max_value=0.08)], p=brightness_transform_prob),
                                RandomApply([GammaChange(max_value=0.08)], p=brightness_transform_prob)])
train_transform = Compose([PrepareImageAndMask(),
                           RandomApply([Cutout(1, 30)], p=0.0 if args.disable_cutout else 0.5),
                           RandomApply([HorizontalFlip()]),
                           geometric_transform,
                           brightness_transform,
                           ResizeToNxN(img_size) if args.resize else PadToNxN(img_size), HWCtoCHW()])
valid_transform = Compose([PrepareImageAndMask(),
                           ResizeToNxN(img_size) if args.resize else PadToNxN(img_size), HWCtoCHW()])
data_fold_id = args.data_fold[len('fold'):]
if len(data_fold_id) == 1:
    list_train = 'list_train{}_3600'
    list_vaild = 'list_valid{}_400'
elif len(data_fold_id) == 2:
    list_train = 'list_train{}_3200'
    list_vaild = 'list_valid{}_800'
else:
    raise RuntimeError("unknown fold {}".format(args.data_fold))

train_dataset = SaltIdentification(mode='train', name=list_train.format(data_fold_id),
                                   transform=train_transform, preload=True, mask_threshold=args.drop_mask_threshold)
valid_dataset = SaltIdentification(mode='train', name=list_vaild.format(data_fold_id),
                                   transform=valid_transform, preload=True)
if args.vtf:
    flipped_valid_transform = Compose([PrepareImageAndMask(), HorizontalFlip(),
                                       ResizeToNxN(img_size) if args.resize else PadToNxN(img_size), HWCtoCHW()])
    flipped_valid_dataset = SaltIdentification(mode='train', name='list_valid{}_400'.format(data_fold_id),
                                               transform=flipped_valid_transform, preload=True)
    valid_dataset = ConcatDataset([valid_dataset, flipped_valid_dataset])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.dataload_workers_nums, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                              num_workers=args.dataload_workers_nums)

# a name used to save checkpoints etc.
full_name = '%s_%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (
    args.model, args.data_fold, args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
if args.comment:
    full_name = '%s_%s' % (full_name, args.comment)

model = models.create(args.model, basenet=args.basenet, pretrained=args.pretrained)

model, optimizer = create_optimizer(model, args.optim, args.learning_rate, args.weight_decay,
                                    momentum=0.9,
                                    fp16_loss_scale=args.fp16_loss_scale,
                                    device=device)

lr_scheduler = create_lr_scheduler(optimizer, **vars(args))

start_timestamp = int(time.time() * 1000)
start_epoch = 0
best_loss = 1e10
best_metric = 0
best_accuracy = 0
global_step = 0

if args.resume:
    print("resuming a checkpoint '%s'" % args.resume)
    if os.path.exists(args.resume):
        saved_checkpoint = torch.load(args.resume)
        old_model = models.load(saved_checkpoint['model_file'])
        model.module.load_state_dict(old_model.state_dict())
        model.float()

        if not args.resume_without_optimizer:
            optimizer.load_state_dict(saved_checkpoint['optimizer'])
            lr_scheduler.load_state_dict(saved_checkpoint['lr_scheduler'])
            best_loss = saved_checkpoint.get('best_loss', best_loss)
            best_metric = saved_checkpoint.get('best_metric', best_metric)
            best_accuracy = saved_checkpoint.get('best_accuracy', best_accuracy)
            start_epoch = saved_checkpoint.get('epoch', start_epoch)
            global_step = saved_checkpoint.get('step', global_step)

        del saved_checkpoint  # reduce memory
        del old_model
    else:
        print(">\n>\n>\n>\n>\n>")
        print(">Warning the checkpoint '%s' doesn't exist! training from scratch!" % args.resume)
        print(">\n>\n>\n>\n>\n>")


def get_lr():
    return optimizer.param_groups[0]['lr']


print("logging into {}".format(args.log_dir))
writer = SummaryWriter(log_dir=args.log_dir)
checkpoint_dir = Path(args.log_dir) / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
models_dir = Path(args.log_dir) / 'models'
models_dir.mkdir(parents=True, exist_ok=True)


def remove_padding(data):
    d_y0, d_y1, d_x0, d_x1 = padding
    y0, y1, x0, x1 = d_y0, d_y0 + orig_img_size, d_x0, d_x0 + orig_img_size
    if data.dim() == 3:
        return data[:, y0:y1, x0:x1]
    elif data.dim() == 4:
        return data[:, :, y0:y1, x0:x1]
    raise RuntimeError("unspported dim {}".format(data.dim()))


def train(epoch, phase='train'):
    global global_step, best_loss, best_metric, best_accuracy

    if phase == 'train':
        writer.add_scalar('%s/learning_rate' % phase, get_lr(), epoch)

    model.train() if phase == 'train' else model.eval()
    torch.set_grad_enabled(True) if phase == 'train' else torch.set_grad_enabled(False)
    dataloader = train_dataloader if phase == 'train' else valid_dataloader

    running_loss, running_metric, running_accuracy = 0.0, 0.0, 0.0
    worst_loss, worst_metric = best_loss, best_metric
    it, total = 0, 0

    if phase == 'valid':
        total_probs = []
        total_truth = []

    pbar_disable = False if epoch == start_epoch else None
    pbar = tqdm(dataloader, unit="images", unit_scale=dataloader.batch_size, disable=pbar_disable)
    for batch in pbar:
        image_ids, inputs, targets = batch['image_id'], batch['input'], batch['mask']

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward
        logit, logit_pixel, logit_image = model(inputs)

        # look at the center only
        if args.loss_on_center:
            logit = remove_padding(logit)
            logit_pixel = (remove_padding(l) for l in logit_pixel)
            targets = remove_padding(targets)

        truth_pixel = targets
        truth_image = (truth_pixel.sum(dim=(1, 2)) > 0).float()
        loss = models.deep_supervised_criterion(logit, logit_pixel, logit_image, truth_pixel, truth_image)

        if not args.loss_on_center and not args.resize:
            logit = remove_padding(logit)
            targets = remove_padding(targets)

        probs = torch.sigmoid(logit).squeeze(1)
        # predictions = probs.squeeze(1) > 0.5

        if phase == 'train':
            # backward
            optimizer.backward(loss / args.gradient_accumulation)
            if it % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

        # statistics
        it += 1
        global_step += 1
        loss = loss.item()
        running_loss += (loss * targets.size(0))
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss, global_step)

        targets_numpy = targets.cpu().numpy()
        probs_numpy = probs.cpu().detach().numpy()
        predictions_numpy = probs_numpy > 0.5  # predictions.cpu().numpy()
        metric_array = calc_metric(targets_numpy, predictions_numpy, type='iou', size_average=False)
        metric = metric_array.mean()
        running_metric += metric_array.sum()

        running_accuracy += calc_metric(targets_numpy, predictions_numpy, type='pixel_accuracy',
                                        size_average=False).sum()

        if phase == 'valid':
            total_truth.append(targets_numpy)
            total_probs.append(probs_numpy)

        visualize_output = False
        if worst_loss > loss:
            worst_loss = loss
            visualize_output = True
        if worst_metric < metric:
            worst_metric = metric
            visualize_output = True
        if visualize_output and args.debug:
            # sort samples by metric
            ind = np.argsort(metric_array)
            images = remove_padding(inputs.cpu())
            images = images[ind]
            probs = probs[ind].cpu()
            predictions = predictions[ind].cpu()
            targets = targets[ind].cpu()

            preds = torch.cat([probs] * 3, 1)
            mask = torch.cat([targets.unsqueeze(1)] * 3, 1)
            all = images.clone()
            all[:, 0] = torch.max(images[:, 0], predictions.float())
            all[:, 1] = torch.max(images[:, 1], targets)
            all = torch.cat((torch.cat((all, images), 3), torch.cat((preds, mask), 3)), 2)
            all_grid = vutils.make_grid(all, nrow=4, normalize=False, pad_value=1)
            writer.add_image('%s/img-mask-pred' % phase, all_grid, global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / total),
            'metric': "%.03f" % (running_metric / total)
        })

    epoch_loss = running_loss / total
    epoch_metric = running_metric / total
    epoch_accuracy = running_accuracy / total
    writer.add_scalar('%s/metric' % phase, epoch_metric, epoch)
    writer.add_scalar('%s/accuracy' % phase, epoch_accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    if phase == 'valid':

        def save_checkpoint(name):
            cycle = ('-cycle%d' % (epoch // args.lr_scheduler_step_size)) if args.lr_scheduler == 'clr' else ''
            model_name = name + '-model'
            model_file_name = '%d-%s-%s%s.pth' % (start_timestamp, model_name, full_name, cycle)
            model_file = models_dir / model_file_name
            models.save(model, model_file)
            mode_file_simple = Path(models_dir / (model_name + '-%s%s.pth' % (args.data_fold, cycle)))
            if mode_file_simple.is_symlink() or mode_file_simple.exists():
                mode_file_simple.unlink()
            mode_file_simple.symlink_to(model_file.relative_to(mode_file_simple.parent))

            checkpoint = {
                'epoch': epoch,
                'step': global_step,
                'model_file': str(model_file),
                'best_loss': best_loss,
                'best_metric': best_metric,
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            checkpoint_filename = name + '-checkpoint-%s%s.pth' % (full_name, cycle)
            checkpoint_file = checkpoint_dir / checkpoint_filename
            torch.save(checkpoint, checkpoint_file)
            checkpoint_file_simple = Path(checkpoint_dir / (name + '-checkpoint-%s%s.pth' % (args.data_fold, cycle)))
            if checkpoint_file_simple.is_symlink() or checkpoint_file_simple.exists():
                checkpoint_file_simple.unlink()
            checkpoint_file_simple.symlink_to(checkpoint_file.relative_to(checkpoint_file_simple.parent))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint('best-loss')
        if epoch_metric > best_metric:
            best_metric = epoch_metric
            save_checkpoint('best-metric')
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            save_checkpoint('best-accuracy')

        save_checkpoint('last')
    return epoch_loss, epoch_metric, epoch_accuracy


print("training %s..." % args.model)
pbar_epoch = trange(start_epoch, args.max_epochs)

# import cProfile
# pr = cProfile.Profile()
# pr.enable()

for epoch in pbar_epoch:
    if args.lr_scheduler != 'plateau':
        if args.lr_scheduler == 'clr':
            if epoch % args.lr_scheduler_step_size == 0:
                # reset best loss and metric for every cycle
                best_loss = 1e10
                best_metric = 0
            lr_scheduler.step(epoch % args.lr_scheduler_step_size)
        else:
            lr_scheduler.step()

    train_epoch_loss, train_epoch_metric, train_epoch_epoch_accuracy = train(epoch, phase='train')
    valid_epoch_loss, valid_epoch_metric, valid_epoch_epoch_accuracy = train(epoch, phase='valid')

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=valid_epoch_loss)

    pbar_epoch.set_postfix({'lr': '%.02e' % get_lr(),
                            'train': '%.03f/%.03f/%.03f' % (
                            train_epoch_loss, train_epoch_metric, train_epoch_epoch_accuracy),
                            'val': '%.03f/%.03f/%.03f' % (
                            valid_epoch_loss, valid_epoch_metric, valid_epoch_epoch_accuracy),
                            'best val': '%.03f/%.03f/%.03f' % (best_loss, best_metric, best_accuracy)},
                           refresh=False)
#    break
# pr.disable()
# pr.print_stats('cumulative')
# pr.dump_stats('test.profile')

print("finished data fold {}".format(args.data_fold))
print("best valid loss: %.05f, best valid metric: %.03f%%" % (best_loss, best_metric))
