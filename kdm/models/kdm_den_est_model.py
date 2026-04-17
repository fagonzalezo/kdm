import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from ..layers import KDMProjLayer, RBFKernelLayer


class KDMDenEstModel(nn.Module):
    """KDM density estimation model.

    `forward(x)` returns per-sample log-density under the learned KDM. The
    training loop minimizes `-forward(x).mean()`.
    """

    def __init__(
        self,
        dim_x: int,
        sigma: float,
        n_comp: int,
        trainable_sigma: bool = True,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(
            sigma=sigma, dim=dim_x, trainable=trainable_sigma
        )
        self.kdmproj = KDMProjLayer(self.kernel, dim_x=dim_x, n_comp=n_comp)
        self.eps = 1e-7

    def forward(self, x):
        return torch.log(self.kdmproj(x) + self.eps) + self.kernel.log_weight()

    def get_distrib(self) -> MixtureSameFamily:
        comp_w = self.kdmproj.c_w.detach().abs() + self.eps
        comp_w = comp_w / comp_w.sum()
        scale = self.kernel.sigma.detach() / math.sqrt(2.0)
        return MixtureSameFamily(
            mixture_distribution=Categorical(probs=comp_w),
            component_distribution=Independent(
                Normal(loc=self.kdmproj.c_x.detach(), scale=scale),
                reinterpreted_batch_ndims=1,
            ),
        )
