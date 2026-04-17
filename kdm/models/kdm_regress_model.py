import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import KDMLayer, RBFKernelLayer
from ..layers.rbf_kernel_layer import _softplus_inv
from ..utils import dm_rbf_expectation, dm_rbf_variance, pure2dm


class KDMRegressModel(nn.Module):
    """KDM regression model.

    `forward(x)` returns the output KDM rho_y; use `predict_reg(x)` to get
    the predictive mean and variance. Training loops compute the NLL via
    `utils.dm_rbf_loglik(y_true, rho_y, sigma_y)`.
    """

    def __init__(
        self,
        encoded_size: int,
        dim_y: int,
        encoder: nn.Module,
        n_comp: int,
        sigma_x: float = 0.1,
        min_sigma_x: float = 1e-3,
        sigma_y: float = 0.1,
        min_sigma_y: float = 1e-3,
        x_train: bool = True,
        y_train: bool = True,
        w_train: bool = True,
        sigma_x_trainable: bool = True,
        sigma_y_trainable: bool = True,
    ):
        super().__init__()
        self.encoded_size = encoded_size
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.encoder = encoder
        self.min_sigma_y = min_sigma_y
        self.kernel = RBFKernelLayer(
            sigma=sigma_x, dim=encoded_size,
            trainable=sigma_x_trainable, min_sigma=min_sigma_x,
        )
        self.kdm = KDMLayer(
            kernel=self.kernel,
            dim_x=encoded_size,
            dim_y=dim_y,
            n_comp=n_comp,
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
        )
        sigma_y_t = torch.as_tensor(float(sigma_y))
        if sigma_y_t <= min_sigma_y:
            raise ValueError(
                f"sigma_y ({float(sigma_y_t):g}) must be > min_sigma_y ({min_sigma_y:g})"
            )
        raw = _softplus_inv(sigma_y_t - min_sigma_y)
        self.raw_sigma_y = nn.Parameter(raw, requires_grad=sigma_y_trainable)

    @property
    def sigma_y(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_y) + self.min_sigma_y

    @sigma_y.setter
    def sigma_y(self, value) -> None:
        with torch.no_grad():
            value_t = torch.as_tensor(
                value, dtype=self.raw_sigma_y.dtype, device=self.raw_sigma_y.device
            )
            if (value_t <= self.min_sigma_y).any():
                raise ValueError(
                    f"Assigned sigma_y must be > min_sigma_y ({self.min_sigma_y:g})"
                )
            self.raw_sigma_y.copy_(_softplus_inv(value_t - self.min_sigma_y))

    def forward(self, x):
        encoded = self.encoder(x)
        rho_x = pure2dm(encoded)
        return self.kdm(rho_x)

    @torch.no_grad()
    def predict_reg(self, x):
        was_training = self.training
        self.eval()
        try:
            rho_y = self.forward(x)
            y_exp = dm_rbf_expectation(rho_y)
            y_var = dm_rbf_variance(rho_y, self.sigma_y)
        finally:
            self.train(was_training)
        return y_exp, y_var

    def get_sigmas(self):
        return float(self.kernel.sigma.detach()), float(self.sigma_y.detach())
