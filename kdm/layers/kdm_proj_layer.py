import torch
import torch.nn as nn


class KDMProjLayer(nn.Module):
    """Kernel Density Matrix projection layer.

    Projects an input vector onto the layer's KDM support points using the
    supplied kernel; returns a scalar weight per batch element that can be
    interpreted (up to the kernel's log_weight normalization) as an
    unnormalized density estimate.

    Input shape:
        (batch_size, dim_x)
    Output shape:
        (batch_size,)

    Arguments:
        kernel: a Kernel module with forward(A, B) and log_weight().
        dim_x: dimensionality of the input state.
        x_train: whether to train the support components.
        w_train: whether to train the component weights.
        n_comp: number of components representing the layer KDM.
    """

    def __init__(
        self,
        kernel,
        dim_x: int,
        x_train: bool = True,
        w_train: bool = True,
        n_comp: int = 0,
    ):
        super().__init__()
        self.kernel = kernel
        self.dim_x = dim_x
        self.x_train = x_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.eps = 1e-7  # matches keras.config.epsilon()

        c_x = torch.randn(n_comp, dim_x) * 0.05  # matches keras.initializers.random_normal default
        c_w = torch.full((n_comp,), 1.0 / n_comp)
        self.c_x = nn.Parameter(c_x, requires_grad=x_train)
        self.c_w = nn.Parameter(c_w, requires_grad=w_train)

    def forward(self, inputs):
        comp_w = self.c_w.abs() + self.eps
        comp_w = comp_w / comp_w.sum()
        in_v = inputs.unsqueeze(1)  # (bs, 1, dim_x)
        out_vw = self.kernel(in_v, self.c_x) ** 2  # (bs, 1, n_comp)
        return torch.einsum('...j,...ij->...', comp_w, out_vw)  # (bs,)
