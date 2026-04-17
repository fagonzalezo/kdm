import torch
import torch.nn as nn


class MemKDMLayer(nn.Module):
    """Memory-based Kernel Density Matrix Layer.

    Receives a sample together with its retrieved neighbors and their
    labels, and returns a KDM over the neighbors weighted by the kernel
    similarity between the sample and each neighbor.

    Input:
        [samples, neighbors, labels]
            samples:   (batch_size, dim_x)
            neighbors: (batch_size, n_comp, dim_x)
            labels:    (batch_size, n_comp, dim_y)
    Output:
        (batch_size, n_comp, dim_y + 1) — standard KDM layout:
            [:, :, 0]  = weights
            [:, :, 1:] = component vectors
    """

    def __init__(self, kernel, dim_x: int, dim_y: int, n_comp: int):
        super().__init__()
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.eps = 1e-7  # matches keras.config.epsilon()

    def forward(self, inputs):
        in_v, c_x, c_y = inputs
        in_v = in_v.unsqueeze(1)                       # (bs, 1, dim_x)
        out_vw = self.kernel(in_v, c_x)                # (bs, 1, n_comp)
        out_w = (out_vw ** 2).clamp(min=self.eps)[:, 0, :]  # (bs, n_comp)
        out_w = out_w / out_w.sum(dim=1, keepdim=True)
        out_w = out_w.unsqueeze(-1)                    # (bs, n_comp, 1)
        return torch.cat((out_w, c_y), dim=2)
