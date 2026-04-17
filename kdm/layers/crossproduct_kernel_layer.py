from ..base import Kernel


class CrossProductKernelLayer(Kernel):
    """Cross-product kernel: splits inputs into two parts and multiplies
    two child kernels applied to each part.

    Arguments:
        dim1: dimension of the first part of the input vector.
        kernel1: kernel for the first part: (bs, n, dim1) x (m, dim1) -> (bs, n, m).
        kernel2: kernel for the second part: (bs, n, d - dim1) x (m, d - dim1) -> (bs, n, m).
    """

    def __init__(self, dim1, kernel1, kernel2):
        super().__init__()
        self.dim1 = dim1
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, A, B):
        """
        Arguments:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Returns:
            K: tensor of shape (bs, n, m)
        """
        A1 = A[:, :, :self.dim1]
        A2 = A[:, :, self.dim1:]
        B1 = B[:, :self.dim1]
        B2 = B[:, self.dim1:]
        return self.kernel1(A1, B1) * self.kernel2(A2, B2)

    def log_weight(self):
        return self.kernel1.log_weight() + self.kernel2.log_weight()
