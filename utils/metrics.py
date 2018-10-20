"""Copied from https://www.kaggle.com/robertkag/metric-script"""

import numpy as np
import scipy.stats


def calc_enptropy(probs, base=2):
    _, counts = np.unique(probs, return_counts=True)
    return scipy.stats.entropy(counts, base=base)


def calc_iou(actual, pred):
    intersection = np.count_nonzero(actual * pred)
    union = np.count_nonzero(actual + pred)
    iou_result = intersection / union if union != 0 else 0.
    return iou_result


def calc_ious(actuals, preds):
    ious_ = np.array([calc_iou(a, p) for a, p in zip(actuals, preds)])
    return ious_


def calc_precisions(thresholds, ious):
    thresholds = np.reshape(thresholds, (1, -1))
    ious = np.reshape(ious, (-1, 1))
    ps = ious > thresholds
    mps = ps.mean(axis=1)
    return mps


def indiv_scores(masks, preds):
    ious = calc_ious(masks, preds)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = calc_precisions(thresholds, ious)

    ###### Adjust score for empty masks
    emptyMasks = np.count_nonzero(masks.reshape((len(masks), -1)), axis=1) == 0
    emptyPreds = np.count_nonzero(preds.reshape((len(preds), -1)), axis=1) == 0
    adjust = (emptyMasks == emptyPreds).astype(np.float)
    precisions[emptyMasks] = adjust[emptyMasks]
    ###################
    return precisions


def calc_metric(masks, preds, type='iou', size_average=True):
    if type == 'iou':
        m = indiv_scores(masks, preds)
    elif type == 'pixel_accuracy':
        m = pixel_accuracy(masks, preds)
    else:
        raise NotImplementedError(type)
    if size_average:
        m = m.mean()
    return m


def pixel_accuracy(masks, preds):
    correct = (preds == masks).astype(np.float).sum(axis=(1,2)) / masks[0, :, :].size
    return correct
