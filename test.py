#!/usr/bin/env python
"""Test UNet and create a Kaggle submission."""
__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import time
import argparse
from tqdm import tqdm

import pandas as pd

import torch
from torch.utils.data import DataLoader
import models
from torchvision.transforms import *

from datasets import *
from transforms import *

from utils import rlenc, rlenc_np, FasterRle, gzip_save

orig_img_size = 101
img_size = 128
padding = compute_padding(orig_img_size, orig_img_size, img_size)
d_y0, d_y1, d_x0, d_x1 = padding
y0, y1, x0, x1 = d_y0, d_y0 + orig_img_size, d_x0, d_x0 + orig_img_size


def predict(model, batch, flipped_batch, use_gpu):
    image_ids, inputs = batch['image_id'], batch['input']
    if use_gpu:
        inputs = inputs.cuda()
    outputs, _, _ = model(inputs)
    probs = torch.sigmoid(outputs)

    if flipped_batch is not None:
        flipped_image_ids, flipped_inputs = flipped_batch['image_id'], flipped_batch['input']
        # assert image_ids == flipped_image_ids
        if use_gpu:
            flipped_inputs = flipped_inputs.cuda()
        flipped_outputs, _, _ = model(flipped_inputs)
        flipped_probs = torch.sigmoid(flipped_outputs)

        probs += torch.flip(flipped_probs, (3,))  # flip back and add
        probs *= 0.5

    probs = probs.squeeze(1).cpu().numpy()
    if args.resize:
        probs = np.swapaxes(probs, 0, 2)
        probs = cv2.resize(probs, (orig_img_size, orig_img_size), interpolation=cv2.INTER_LINEAR)
        probs = np.swapaxes(probs, 0, 2)
    else:
        probs = probs[:, y0:y1, x0:x1]
    return probs


def test():
    test_transform = Compose([PrepareImageAndMask(),
                              ResizeToNxN(img_size) if args.resize else PadToNxN(img_size), HWCtoCHW()])
    test_dataset = SaltIdentification(mode='test', transform=test_transform, preload=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.dataload_workers_nums)

    flipped_test_transform = Compose([PrepareImageAndMask(), HorizontalFlip(),
                                      ResizeToNxN(img_size) if args.resize else PadToNxN(img_size), HWCtoCHW()])
    flipped_test_dataset = SaltIdentification(mode='test', transform=flipped_test_transform, preload=False)
    flipped_test_dataloader_iter = iter(DataLoader(flipped_test_dataset, batch_size=args.batch_size,
                                                   num_workers=args.dataload_workers_nums))

    model.eval()
    torch.set_grad_enabled(False)

    prediction = {}
    submission = {}
    pbar = tqdm(test_dataloader, unit="images", unit_scale=test_dataloader.batch_size, disable=None)

    empty_images_count = 0
    for batch in pbar:
        if args.tta:
            flipped_batch = next(flipped_test_dataloader_iter)
        else:
            flipped_batch = None

        probs = predict(model, batch, flipped_batch, use_gpu=use_gpu)
        pred = probs > args.threshold
        empty_images_count += (pred.sum(axis=(1, 2)) == 0).sum()

        probs_uint16 = (65535 * probs).astype(dtype=np.uint16)

        image_ids = batch['image_id']
        prediction.update(dict(zip(image_ids, probs_uint16)))
        rle = rlenc_np(pred)
        submission.update(dict(zip(image_ids, rle)))

    empty_images_percentage = empty_images_count / len(prediction)
    print("empty images: %.2f%% (in public LB 38%%)" % (100 * empty_images_percentage))

    gzip_save('-'.join([args.output_prefix, 'probabilities.pkl.gz']), prediction)
    sub = pd.DataFrame.from_dict(submission, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('-'.join([args.output_prefix, 'submission.csv']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=30, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--tta', action='store_true', help='test time augmentation')
    parser.add_argument('--seed', type=int, default=None, help='manual seed for deterministic')
    parser.add_argument("--threshold", type=float, default=0.5, help='probability threshold')
    parser.add_argument("--output-prefix", type=str, default='noprefix', help='prefix string for output files')
    parser.add_argument('--resize', action='store_true', help='resize to 128x128 instead of reflective padding')
    parser.add_argument("model", help='a pretrained neural network model')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)

    print("loading model...")
    model = models.load(args.model)
    model.float()

    if use_gpu:
        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
        model.cuda()

    print("testing %s..." % args.model)
    since = time.time()
    test()
    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                     time_elapsed % 60)
    print("finished")
