import keras

class CrossProductKernelLayer(keras.layers.Layer):

    def __init__(self, dim1, kernel1, kernel2, **kwargs):
        '''
        Create a layer that calculates the cross product kernel of two input
        kernels. The input vector are divided into two parts, the first of dimension 
        dim1 and the second of dimension d - dim1. Each input kernel is applied to 
        one of the parts of the input. 
        Arguments:
            dim1: the dimension of the first part of the input vector
            kernel1: a kernel function
                    k1:(bs, n, dim1)x(m, dim1) -> (bs, n, m)
            kernel2: a kernel function
                    k2:(bs, n, d - dim1)x(m, d - dim1) -> (bs, n, m)
        '''
        super().__init__(**kwargs)
        self.dim1 = dim1
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        A1 = A[:, :, :self.dim1]
        A2 = A[:, :, self.dim1:]
        B1 = B[:, :self.dim1]
        B2 = B[:, self.dim1:]
        return self.kernel1(A1, B1) * self.kernel2(A2, B2)
    
    def log_weight(self):
        return self.kernel1.log_weight() + self.kernel2.log_weight()