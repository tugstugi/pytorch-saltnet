
import torch
from queue import Queue

from .functions import *
from .abn import ABN


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var,
                           self.training or not self.track_running_stats,
                           exponential_average_factor, self.eps, self.activation, self.slope)


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with cross-GPU synchronization

    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DataParallel`.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 activation="leaky_relu", slope=0.01, devices=None):
        """Creates a synchronized, InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        devices : list of int or None
            IDs of the GPUs that will run the replicas of this module.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats, activation=activation, slope=slope)
        self.devices = devices if devices else list(range(torch.cuda.device_count()))

        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        if len(self.devices) < 2:
            # fallback for CPU mode or single GPU mode
            return super().forward(x)

        if x.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }

        return inplace_abn_sync(x, self.weight, self.bias, self.running_mean, self.running_var,
                                extra, self.training, self.momentum, self.eps, self.activation, self.slope)

    def extra_repr(self):
        rep = super().extra_repr()
        rep += ', devices={devices}'.format(**self.__dict__)
        return rep
