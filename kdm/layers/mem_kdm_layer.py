import torch
import torch.nn as nn

from ..utils import comp2dm


class MemKDMLayer(nn.Module):
    """Memory-based Kernel Density Matrix Layer.

    Maps an input KDM `rho_in` to an output KDM over a per-sample memory
    (`neighbors`, `labels`). Mirrors `KDMLayer.forward` but with no learned
    parameters — the support set is supplied per batch.

    Input:
        (rho_in, neighbors, labels)
            rho_in:    (batch_size, n_comp_in, dim_x + 1)
            neighbors: (batch_size, n_comp,    dim_x)
            labels:    (batch_size, n_comp,    dim_y)
    Output:
        (batch_size, n_comp, dim_y + 1) — standard KDM layout:
            [:, :, 0]  = weights
            [:, :, 1:] = component vectors

    `dim_x`, `dim_y`, `n_comp` are informational (shapes are inferred at
    call time); they are kept for API symmetry with `KDMLayer`.
    """

    def __init__(self, kernel, dim_x: int, dim_y: int, n_comp: int):
        super().__init__()
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.eps = 1e-12

    def _compute_mixture(self, rho_in, neighbors):
        """Shared intermediate used by forward and log_marginal.

        Returns:
            in_w:  (bs, n_comp_in)
            out_w: (bs, n_comp_in, n_comp) — unnormalized joint weights
        """
        in_w = rho_in[:, :, 0]                        # (bs, n_comp_in)
        in_v = rho_in[:, :, 1:]                       # (bs, n_comp_in, dim_x)
        out_vw = self.kernel(in_v, neighbors)          # (bs, n_comp_in, n_comp)
        out_w = out_vw.square()                        # (bs, n_comp_in, n_comp)
        return in_w, out_w

    def forward(self, inputs):
        rho_in, neighbors, labels = inputs
        in_w, out_w = self._compute_mixture(rho_in, neighbors)
        out_w = out_w.clamp(min=self.eps)
        out_w = out_w / out_w.sum(dim=2, keepdim=True)              # normalize over n_comp
        out_w = torch.einsum('...i,...ij->...j', in_w, out_w)       # (bs, n_comp)
        return comp2dm(out_w, labels)

    def log_marginal(self, rho_in, neighbors):
        """Log-likelihood of `rho_in` under the per-sample marginal KDM over
        `neighbors`. Memory weights are treated as uniform (`1/n_comp`),
        matching `KDMLayer.log_marginal` where `comp_w` sums to 1. Returns
        shape `(bs,)`; caller typically `.mean()`-reduces into the loss.
        """
        in_w, out_w = self._compute_mixture(rho_in, neighbors)
        n_comp = out_w.shape[-1]
        proj = torch.einsum('...i,...ij->...', in_w, out_w) / n_comp  # (bs,)
        return torch.log(proj + self.eps) + self.kernel.log_weight()
