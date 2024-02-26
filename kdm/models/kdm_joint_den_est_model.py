import keras
from ..layers import RBFKernelLayer, KDMProjLayer, \
                     CosineKernelLayer, CrossProductKernelLayer
import numpy as np
from sklearn.metrics import pairwise_distances


class KDMJointDenEstModel(keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 sigma,
                 n_comp,
                 trainable_sigma=True,
                 min_sigma=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.kernel_x = RBFKernelLayer(sigma, dim=dim_x, 
                                       trainable=trainable_sigma,
                                       min_sigma=min_sigma)
        self.kernel_y = CosineKernelLayer()
        self.kernel = CrossProductKernelLayer(dim1=dim_x, kernel1=self.kernel_x, kernel2=self.kernel_y)
        self.kdmproj = KDMProjLayer(self.kernel,
                                dim_x=dim_x + dim_y,
                                n_comp=n_comp)
        self.eps = keras.config.epsilon()

    def call(self, inputs):
        log_probs = (keras.ops.log(self.kdmproj(inputs) + self.eps)
                     + self.kernel.log_weight())
        self.add_loss(-keras.ops.mean(log_probs))
        return log_probs
    
    def init_components(self, samples_xy,
                        sigma):
        self.kernel_x.sigma.assign(sigma)
        self.kdmproj.c_x.assign(samples_xy)
        self.kdmproj.c_w.assign(keras.ops.ones((self.n_comp,)) / self.n_comp)
