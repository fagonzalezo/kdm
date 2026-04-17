import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDMLayer(nn.Module):
    """Kernel Density Matrix Layer.

    Maps an input KDM to an output KDM via a learned joint KDM.

    Input shape:
        (batch_size, n_comp_in, dim_x + 1)
        where [:, :, 0] are the input component weights and [:, :, 1:] are
        the input component vectors.
    Output shape:
        (batch_size, n_comp, dim_y + 1)
        with the same weight/vector layout.

    Arguments:
        kernel: a Kernel module with forward(A, B) and log_weight().
        dim_x: dimensionality of the input state.
        dim_y: dimensionality of the output state.
        n_comp: number of components representing the joint KDM.
        x_train / y_train / w_train: whether to train each component tensor.

    Regularization and the generative log-likelihood are not applied inside
    forward; compute them explicitly in the training loop using `log_marginal`
    and parameter-wise L1 helpers in `kdm.losses`.
    """

    def __init__(
        self,
        kernel,
        dim_x: int,
        dim_y: int,
        x_train: bool = True,
        y_train: bool = True,
        w_train: bool = True,
        n_comp: int = 0,
    ):
        super().__init__()
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.eps = 1e-12

        c_x = torch.randn(n_comp, dim_x) * 0.05  # matches keras.initializers.random_normal default
        c_y = torch.full((n_comp, dim_y), math.sqrt(1.0 / dim_y))
        c_w = torch.full((n_comp,), 1.0 / n_comp)
        self.c_x = nn.Parameter(c_x, requires_grad=x_train)
        self.c_y = nn.Parameter(c_y, requires_grad=y_train)
        self.c_w = nn.Parameter(c_w, requires_grad=w_train)

    def _normalized_comp_w(self) -> torch.Tensor:
        comp_w = self.c_w.abs()
        return comp_w / comp_w.sum().clamp(min=self.eps)

    def _compute_mixture(self, rho_in):
        """Shared intermediate used by forward and log_marginal.

        Returns:
            in_w:  (bs, n_comp_in)
            out_w: (bs, n_comp_in, n_comp) — unnormalized joint weights
        """
        in_w = rho_in[:, :, 0]
        in_v = rho_in[:, :, 1:]
        comp_w = self._normalized_comp_w()
        out_vw = self.kernel(in_v, self.c_x)                     # (bs, n_comp_in, n_comp)
        out_w = comp_w.view(1, 1, -1) * out_vw.square()          # (bs, n_comp_in, n_comp)
        return in_w, out_w

    def forward(self, rho_in):
        in_w, out_w = self._compute_mixture(rho_in)
        out_w = out_w.clamp(min=self.eps)
        out_w = out_w / out_w.sum(dim=2, keepdim=True)           # (bs, n_comp_in, n_comp)
        out_w = torch.einsum('...i,...ij->...j', in_w, out_w)    # (bs, n_comp)
        out_w = out_w.unsqueeze(-1)                              # (bs, n_comp, 1)
        out_y = self.c_y.unsqueeze(0).expand(out_w.shape[0], -1, -1)
        return torch.cat((out_w, out_y), dim=2)

    def log_marginal(self, rho_in):
        """Log-likelihood of the input under the layer's marginal KDM over x.

        Used as the `generative` term when training conditional generative
        models. Returns a tensor of shape (batch_size,) that the caller
        typically means-reduces into the loss.
        """
        in_w, out_w = self._compute_mixture(rho_in)
        proj = torch.einsum('...i,...ij->...', in_w, out_w)      # (bs,) — summed over n_comp
        return torch.log(proj + self.eps) + self.kernel.log_weight()
