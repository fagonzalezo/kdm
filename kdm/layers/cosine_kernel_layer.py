import torch
import torch.nn.functional as F

from ..base import Kernel


class CosineKernelLayer(Kernel):
    """Cosine-similarity kernel: dot product of L2-normalized vectors."""

    def forward(self, A, B):
        """
        Arguments:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Returns:
            K: tensor of shape (bs, n, m)
        """
        A = F.normalize(A, p=2, dim=-1, eps=1e-12)
        B = F.normalize(B, p=2, dim=-1, eps=1e-12)
        return torch.einsum("...nd,md->...nm", A, B)
