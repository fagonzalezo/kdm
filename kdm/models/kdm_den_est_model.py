import keras
from ..layers import RBFKernelLayer, KDMProjLayer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow_probability as tfp

class KDMDenEstModel(keras.Model):
    def __init__(self,
                 dim_x,
                 sigma,
                 n_comp,
                 trainable_sigma=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_x = dim_x
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma, trainable=trainable_sigma, dim=dim_x)
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
            nn_model = NearestNeighbors(n_neighbors=3)
            nn_model.fit(samples_x)
            distances, _ = nn_model.kneighbors(samples_x)
            sigma = np.mean(distances[:, 2]) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.kdmproj.c_x.assign(samples_x)
        self.kdmproj.c_w.assign(keras.ops.ones((self.n_comp,)) / self.n_comp)

    def get_distrib(self):
        comp_w = keras.ops.abs(self.kdmproj.c_w) + self.eps
        comp_w = comp_w / keras.ops.sum(comp_w)
        gm = tfp.distributions.MixtureSameFamily(
            reparameterize=True,
            mixture_distribution=tfp.distributions.Categorical(
                                    probs=comp_w),
            components_distribution=tfp.distributions.Independent( 
                tfp.distributions.Normal(
                    loc=self.kdmproj.c_x,  # component 2
                    scale=self.kernel.sigma / np.sqrt(2.)),
                    reinterpreted_batch_ndims=1))
        return gm
