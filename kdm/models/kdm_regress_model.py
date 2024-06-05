import keras
import numpy as np
from ..layers import KDMLayer, RBFKernelLayer
from ..utils import pure2dm, dm2discrete, dm2comp
from sklearn.metrics import pairwise_distances

class KDMRegressModel(keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 encoder, 
                 n_comp, 
                 sigma=0.1,
                 w_train=True,
                 **kwargs):
        super().__init__(**kwargs) 
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma=sigma, 
                                         dim=encoded_size, 
                                         trainable=True)
        self.kdm = KDMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp,
                                       w_train=w_train)
    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y = self.kdm(rho_x)
        w, v = dm2comp(rho_y)
        y = keras.ops.einsum('...j,...ji->...i', w, v)
        return y

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        encoded_x = self.encoder(samples_x)
        if init_sigma:
            np_encoded_x = keras.ops.convert_to_numpy(encoded_x)
            distances = pairwise_distances(np_encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.kdm.c_x.assign(encoded_x)
        self.kdm.c_y.assign(samples_y)
        self.kdm.c_w.assign(keras.ops.ones((self.n_comp,)) / self.n_comp)

