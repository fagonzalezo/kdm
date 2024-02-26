import keras

class CosineKernelLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        '''
        Builds a layer that calculates the cosine kernel between two set of vectors
        '''
        super().__init__(**kwargs)
        self.eps = 1e-6

    def call(self, A, B):
        '''
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        '''
        A = keras.utils.normalize(A, order=2, axis=-1)
        B = keras.utils.normalize(B, order=2, axis=-1)
        K = keras.ops.einsum("...nd,md->...nm", A, B)
        return K
    
    def log_weight(self):
        return 0
