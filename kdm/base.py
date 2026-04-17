import torch
import torch.nn as nn


class Kernel(nn.Module):
    """Abstract base class for KDM kernel layers.

    Subclasses implement `forward(A, B)` returning a kernel matrix and
    `log_weight()` returning a log-normalization constant used by the
    generative log-likelihood term in KDM layers.

    `log_weight()` may return either a Python float or a 0-d torch tensor;
    it is added to per-batch log probabilities, so autograd-friendly tensors
    are preferred when the constant depends on trainable parameters.
    """

    def forward(self, A, B):
        raise NotImplementedError

    def log_weight(self):
        return 0.0
