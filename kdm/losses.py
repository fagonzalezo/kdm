"""Training-loop helpers that replace the removed Keras `add_loss` calls.

Users compose these explicitly in their training step, alongside the primary
task loss. The layers themselves return only their principal output; all
regularization and the generative term are exposed here (or via
`KDMLayer.log_marginal`).
"""
import torch
import torch.nn.functional as F

from .utils import gauss_entropy_lb

__all__ = ["l1_norm", "gauss_entropy_lb"]


def l1_norm(vals: torch.Tensor) -> torch.Tensor:
    """Row-normalized L1 regularizer.

    Matches the `l1_loss` helper in the legacy KDMLayer: each row is
    L2-normalized to unit norm, then the mean of absolute entries is returned.

    Arguments:
        vals: tensor of shape (batch_size, n)
    """
    bs = vals.shape[0]
    normed = F.normalize(vals, p=2.0, dim=1, eps=1e-12)
    return normed.abs().sum() / bs
