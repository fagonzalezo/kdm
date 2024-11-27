import keras
import numpy as np

class RBFKernelLayer(keras.layers.Layer):
    def __init__(self, sigma, dim, trainable=True, min_sigma=1e-3, **kwargs):
        '''
        Builds a layer that calculates the rbf kernel between two set of vectors
        Arguments:
            sigma: RBF scale parameter.
        '''
        super().__init__(**kwargs)
        self.sigma = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(value=sigma),
            trainable=trainable)
        self.dim = dim
        self.min_sigma = min_sigma

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        shape_A = keras.ops.shape(A)
        shape_B = keras.ops.shape(B)
        A_norm = keras.ops.sum(A ** 2, axis=-1)[..., np.newaxis]
        B_norm = keras.ops.sum(B ** 2, axis=-1)[np.newaxis, np.newaxis, :]
        A_reshaped = keras.ops.reshape(A, [-1, shape_A[2]])
        AB = keras.ops.matmul(A_reshaped, keras.ops.transpose(B)) 
        AB = keras.ops.reshape(AB, [shape_A[0], shape_A[1], shape_B[0]])
        dist2 = A_norm + B_norm - 2. * AB
        dist2 = keras.ops.clip(dist2, 0., np.inf)
        sigma = keras.ops.clip(self.sigma, self.min_sigma, np.inf)
        K = keras.ops.exp(-dist2 / (2. * sigma ** 2.)) 
        return K
    
    def log_weight(self):
        sigma = keras.ops.clip(self.sigma, self.min_sigma, np.inf)
        return - self.dim * keras.ops.log(sigma + 1e-12) - self.dim * np.log(4 * np.pi) 

class MemRBFKernelLayer(RBFKernelLayer):
    def __init__(self, sigma, dim, trainable=True, min_sigma=1e-3, **kwargs):
        '''
        Builds a layer that calculates the rbf kernel between two set of vectors
        Arguments:
            sigma: RBF scale parameter. 
        '''
        super().__init__(sigma, dim, trainable, min_sigma, **kwargs)

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (bs, m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        A_norm = keras.ops.sum(A ** 2, axis=-1)[..., np.newaxis]  # shape (bs, n, 1)
        B_norm = keras.ops.sum(B ** 2, axis=-1)[:, np.newaxis, :]  # shape (bs, 1, m)
        AB = keras.ops.matmul(A, keras.ops.transpose(B, [0, 2, 1])) 
        dist2 = A_norm + B_norm - 2. * AB # shape (bs, n, m)
        dist2 = keras.ops.clip(dist2, 0., np.inf)
        sigma = keras.ops.clip(self.sigma, self.min_sigma, np.inf)
        K = keras.ops.exp(-dist2 / (2. * sigma ** 2.)) # type: ignore
        return K