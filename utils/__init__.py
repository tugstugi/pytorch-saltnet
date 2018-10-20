import os
import gzip
import pickle
from contextlib import contextmanager
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from .lr_scheduler import FindLR, NoamLR

from .rle import RLenc as rlenc
from .rle import toRunLength as rlenc_np
from .rle import FasterRle


def choose_device(device):
    if not isinstance(device, str):
        return device

    if device not in ['cuda', 'cpu']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        assert torch.cuda.is_available()

    device = torch.device(device)
    return device


@contextmanager
def save_cuda_memory(model):
    # from cuda to cpu if necessary, for saving cuda RAM
    # NOTE: assume model in single device and single dtype
    param = next(model.parameters())
    device = param.device
    model.to(device='cpu')
    yield
    model.to(device=device)


def optimizer_cpu_state_dict(optimizer):
    # save cuda RAM
    optimizer_state_dict = optimizer.state_dict()

    dict_value_to_cpu = lambda d: {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                   for k, v in d.items()}

    if 'optimizer_state_dict' in optimizer_state_dict:
        #  FP16_Optimizer
        cuda_state_dict = optimizer_state_dict['optimizer_state_dict']
    else:
        cuda_state_dict = optimizer_state_dict

    if 'state' in cuda_state_dict:
        cuda_state_dict['state'] = {k: dict_value_to_cpu(v)
                                    for k, v in cuda_state_dict['state'].items()}

    return optimizer_state_dict


def get_num_workers(jobs):
    """
    Parameters
    ----------
    jobs How many jobs to be paralleled. Negative or 0 means number of cpu cores left.

    Returns
    -------
    How many subprocess to be used
    """
    num_workers = jobs
    if num_workers <= 0:
        num_workers = os.cpu_count() + jobs
    if num_workers < 0 or num_workers > os.cpu_count():
        raise RuntimeError("System doesn't have so many cpu cores: {} vs {}".format(jobs, os.cpu_count()))
    return num_workers


def create_optimizer(net, name, learning_rate, weight_decay, momentum=0, fp16_loss_scale=None,
                     optimizer_state=None, device=None):
    net.float()

    use_fp16 = fp16_loss_scale is not None
    if use_fp16:
        from apex import fp16_utils
        net = fp16_utils.network_to_half(net)

    device = choose_device(device)
    print('use', device)
    if device.type == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net = net.to(device)

    # optimizer
    parameters = [p for p in net.parameters() if p.requires_grad]
    print('N of parameters', len(parameters))

    if name == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adamw':
        from .adamw import AdamW
        optimizer = AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(name)

    if use_fp16:
        from apex import fp16_utils
        if fp16_loss_scale == 0:
            opt_args = dict(dynamic_loss_scale=True)
        else:
            opt_args = dict(static_loss_scale=fp16_loss_scale)
        print('FP16_Optimizer', opt_args)
        optimizer = fp16_utils.FP16_Optimizer(optimizer, **opt_args)
    else:
        optimizer.backward = lambda loss: loss.backward()

    if optimizer_state:
        if use_fp16 and 'optimizer_state_dict' not in optimizer_state:
            # resume FP16_Optimizer.optimizer only
            optimizer.optimizer.load_state_dict(optimizer_state)
        elif use_fp16 and 'optimizer_state_dict' in optimizer_state:
            # resume optimizer from FP16_Optimizer.optimizer
            optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
        else:
            optimizer.load_state_dict(optimizer_state)

    return net, optimizer


def create_lr_scheduler(optimizer, lr_scheduler, **kwargs):
    if not isinstance(optimizer, optim.Optimizer):
        # assume FP16_Optimizer
        optimizer = optimizer.optimizer

    if lr_scheduler == 'plateau':
        patience = kwargs.get('lr_scheduler_patience', 10) // kwargs.get('validation_interval', 1)
        factor = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, eps=0)
    elif lr_scheduler == 'step':
        step_size = kwargs['lr_scheduler_step_size']
        gamma = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'cos':
        max_epochs = kwargs['max_epochs']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    elif lr_scheduler == 'milestones':
        milestones = kwargs['lr_scheduler_milestones']
        gamma = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif lr_scheduler == 'findlr':
        max_steps = kwargs['max_steps']
        lr_scheduler = FindLR(optimizer, max_steps)
    elif lr_scheduler == 'noam':
        warmup_steps = kwargs['lr_scheduler_warmup']
        lr_scheduler = NoamLR(optimizer, warmup_steps=warmup_steps)
    elif lr_scheduler == 'clr':
        step_size = kwargs['lr_scheduler_step_size']
        learning_rate = kwargs['learning_rate']
        lr_scheduler_gamma = kwargs['lr_scheduler_gamma']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=step_size,
                                                                  eta_min=learning_rate * lr_scheduler_gamma)
    else:
        raise NotImplementedError("unknown lr_scheduler " + lr_scheduler)
    return lr_scheduler


def gzip_save(filename, obj):
    """save objects into a compressed diskfile"""
    with gzip.open(filename, 'wb') as fil:
        pickle.dump(obj, fil)


def gzip_load(filename):
    """reload objects from a compressed diskfile"""
    with gzip.open(filename, 'rb') as fil:
        return pickle.load(fil)


def pickle_load(filename):
    """load pickle"""
    with open(filename, 'rb') as fil:
        return pickle.load(fil)

