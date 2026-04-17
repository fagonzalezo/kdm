from ..base import Kernel


class CompTransKernelLayer(Kernel):
    """Composes a transformation and a kernel to create a new kernel.

    Arguments:
        transform: a callable (nn.Module) f that transforms inputs before
                   passing them to the kernel: f: (bs, d) -> (bs, D).
        kernel: kernel applied to transformed inputs:
                k: (bs, n, D) x (m, D) -> (bs, n, m).
    """

    def __init__(self, transform, kernel):
        super().__init__()
        self.transform = transform
        self.kernel = kernel

    def forward(self, A, B):
        """
        Arguments:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Returns:
            K: tensor of shape (bs, n, m)
        """
        bs, n, d = A.shape
        A_flat = A.reshape(bs * n, d)
        A_trans = self.transform(A_flat)
        dim_out = A_trans.shape[-1]
        A_out = A_trans.reshape(bs, n, dim_out)
        B_out = self.transform(B)
        return self.kernel(A_out, B_out)

    def log_weight(self):
        return self.kernel.log_weight()
