"""Transforms for UNet.

Many methods are taken from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63974.
"""

__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import cv2
import random

import numpy as np


class PrepareImageAndMask(object):
    """Prepare images and masks like fixing channel numbers."""

    def __call__(self, data):
        img = data['input']
        img = img[:, :, :3]  # max 3 channels
        img = img / 255

        if 'mask' in data:
            mask = data['mask']
        else:
            mask = np.zeros(img.shape[:2], dtype=img.dtype)

        data['input'] = img.astype(np.float32)
        data['mask'] = mask.astype(np.float32)
        return data


class ResizeToNxN(object):
    """Resize input images to rgb NxN and the masks into gray NxN."""

    def __init__(self, n=128):
        self.n = n

    def __call__(self, data):
        n = self.n
        data['input'] = cv2.resize(data['input'], (n, n), interpolation=cv2.INTER_LINEAR)
        data['mask'] = cv2.resize(data['mask'], (n, n), interpolation=cv2.INTER_NEAREST)
        return data


def compute_padding(h, w, n=128):
    if h % n == 0:
        dy0, dy1 = 0, 0
    else:
        dy = n - h % n
        dy0 = dy // 2
        dy1 = dy - dy0

    if w % n == 0:
        dx0, dx1 = 0, 0
    else:
        dx = n - w % n
        dx0 = dx // 2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


class PadToNxN(object):
    """Pad to image size NxN using border reflection."""

    def __init__(self, n=128):
        self.n = n

    def __call__(self, data):
        n = self.n
        h, w = data['input'].shape[:2]
        dy0, dy1, dx0, dx1 = compute_padding(h, w, n)

        data['input'] = cv2.copyMakeBorder(data['input'], dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
        data['mask'] = cv2.copyMakeBorder(data['mask'], dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
        return data


class HorizontalFlip(object):
    """Flip input and masks horizontally."""

    def __call__(self, data):
        data['input'] = cv2.flip(data['input'], 1)
        data['mask'] = cv2.flip(data['mask'], 1)
        return data


class BrightnessShift(object):
    """Brightness shift."""

    def __init__(self, max_value=0.1):
        self.max_value = max_value

    def __call__(self, data):
        img = data['input']
        img += np.random.uniform(-self.max_value, self.max_value)
        data['input'] = np.clip(img, 0, 1)
        return data


class BrightnessScaling(object):
    """Brightness scaling."""

    def __init__(self, max_value=0.08):
        self.max_value = max_value

    def __call__(self, data):
        img = data['input']
        img *= np.random.uniform(1 - self.max_value, 1 + self.max_value)
        data['input'] = np.clip(img, 0, 1)
        return data


class GammaChange(object):
    """Gamma change."""

    def __init__(self, max_value=0.08):
        self.max_value = max_value

    def __call__(self, data):
        img = data['input']
        img = img ** (1.0 / np.random.uniform(1 - self.max_value, 1 + self.max_value))
        data['input'] = np.clip(img, 0, 1)
        return data


def do_elastic_transform(image, mask, grid=10, distort=0.2):
    # https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
    # https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                      borderValue=(0, 0, 0,))
    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
                     borderValue=(0, 0, 0,))

    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class ElasticDeformation(object):
    """Elastic deformation."""

    def __init__(self, grid=10, max_distort=0.15):
        self.grid = grid
        self.max_distort = max_distort

    def __call__(self, data):
        distort = np.random.uniform(0, self.max_distort)
        img, mask = do_elastic_transform(data['input'], data['mask'], self.grid, distort)

        data['input'] = img
        data['mask'] = mask
        return data


def do_rotation_transform(image, mask, angle=0):
    height, width = image.shape[:2]
    cc = np.cos(angle / 180 * np.pi)
    ss = np.sin(angle / 180 * np.pi)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2, height / 2])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101,
                                borderValue=(0, 0, 0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101,
                               borderValue=(0, 0, 0,))
    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class Rotation(object):
    """Rotation."""

    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, data):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        img, mask = do_rotation_transform(data['input'], data['mask'], angle)

        data['input'] = img
        data['mask'] = mask
        return data


def do_crop_and_rescale(image, mask, x0, y0, x1, y1):
    height, width = image.shape[:2]
    image = image[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class CropAndRescale(object):
    """Crop and rescale back to the initial size."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        img, mask = data['input'], data['mask']
        h, w = img.shape[:2]

        dy = int(h * self.max_scale)
        dx = int(w * self.max_scale)

        img, mask = do_crop_and_rescale(img, mask, np.random.randint(0, dx), np.random.randint(0, dy),
                                        w - np.random.randint(0, dx), h - np.random.randint(0, dy))

        data['input'] = img
        data['mask'] = mask
        return data


def do_horizontal_shear(image, mask, scale=0):
    height, width = image.shape[:2]
    dx = int(scale * width)

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101, borderValue=(0, 0, 0,))
    # mask = (mask > 0.5).astype(np.float32)
    return image, mask


class HorizontalShear(object):

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        scale = np.random.uniform(-self.max_scale, self.max_scale)
        img, mask = do_horizontal_shear(data['input'], data['mask'], scale)

        data['input'] = img
        data['mask'] = mask
        return data


class HWCtoCHW(object):
    def __call__(self, data):
        data['input'] = data['input'].transpose((2, 0, 1))
        return data


# adopt from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, mean=(0.485, 0.456, 0.406)):
        self.n_holes = n_holes
        self.length = length
        self.mean = mean

    def __call__(self, data):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = data['input']
        mask = data['mask']
        mask_pixel_count = mask.sum()

        h, w, _ = img.shape

        cut_mask = np.zeros((h, w), np.bool)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            if mask_pixel_count > 0:
                cut_mask_pixel_count = mask[y1: y2, x1: x2].sum()
                if cut_mask_pixel_count / mask_pixel_count > 0.5:
                    continue  # don't cut more than half of mask

            cut_mask[y1: y2, x1: x2] = 1

        img[cut_mask] = self.mean
        return data


class SaltAndPepper(object):

    def __init__(self, probability=0.01):
        self.probability = probability
        self.threshold = 1 - probability

    def __call__(self, data):
        img = data['input']
        h, w = img.shape[:2]
        noise = np.random.rand(h, w)
        img[noise < self.probability] = 0
        img[noise > self.threshold] = 1
        data['input'] = img
        return data
