
import torch.nn as nn
import torch.nn.functional as functional


class ABN(nn.BatchNorm2d):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 activation="leaky_relu", slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
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
                         track_running_stats=track_running_stats)

        if activation not in ("leaky_relu", "elu", "none"):
            raise NotImplementedError(activation)

        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = super().forward(x)

        if self.activation == "leaky_relu":
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == "elu":
            return functional.elu(x, inplace=True)
        else:
            return x

    def extra_repr(self):
        rep = super().extra_repr()
        rep += ', activation={activation}'.format(**self.__dict__)
        if self.activation == "leaky_relu":
            rep += ', slope={slope}'.format(**self.__dict__)
        return rep