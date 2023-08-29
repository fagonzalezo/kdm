import keras_core as keras
from ..layers import RBFKernelLayer, KDMProjLayer
import numpy as np
from sklearn.metrics import pairwise_distances

class KDMDenEstModel(keras.Model):
    def __init__(self,
                 dim_x,
                 sigma,
                 n_comp,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma, dim=dim_x)
        self.kdmproj = KDMProjLayer(self.kernel,
                                dim_x=dim_x,
                                n_comp=n_comp)
        self.eps = keras.config.epsilon()

    def call(self, inputs):
        log_probs = (keras.ops.log(self.kdmproj(inputs) + self.eps)
                     + self.kernel.log_weight())
        self.add_loss(-keras.ops.mean(log_probs))
        return log_probs
    
    def init_components(self, samples_x, init_sigma=False, sigma_mult=1):
        if init_sigma:
            distances = pairwise_distances(samples_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.kdmproj.c_x.assign(samples_x)
        self.kdmproj.c_w.assign(keras.ops.ones((self.n_comp,)) / self.n_comp)
