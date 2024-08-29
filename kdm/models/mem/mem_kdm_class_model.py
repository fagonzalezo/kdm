import keras
import faiss
import numpy as np
from ...layers import MemRBFKernelLayer, MemKDMLayer
from ...utils import dm2discrete
import tensorflow as tf

class MemKDMClassModel(keras.Model):
    '''
    Memory-based KDM model for classification
    '''
    def __init__(self,
                 encoded_size,
                 dim_y,
                 n_comp,
                 sigma=0.1,
                 **kargs):
        super().__init__(**kargs)
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.n_comp = n_comp
        self.kernel = MemRBFKernelLayer(sigma=sigma,
                                         dim=encoded_size,
                                         trainable=True)
        self.mkdm = MemKDMLayer(kernel=self.kernel,
                                       dim_x=encoded_size,
                                       dim_y=dim_y,
                                       n_comp=n_comp)

    def call(self, input):
        x_enc, x_neigh, y_neigh = input
        y_neigh = keras.ops.one_hot(y_neigh, self.dim_y)
        rho_y = self.mkdm((x_enc, x_neigh, y_neigh))
        probs = dm2discrete(rho_y)
        return probs