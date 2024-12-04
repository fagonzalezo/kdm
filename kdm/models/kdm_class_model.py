import keras
import numpy as np
from ..layers import KDMLayer, RBFKernelLayer
from ..utils import pure2dm, dm2discrete
from sklearn.metrics import pairwise_distances

class KDMClassModel(keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 encoder, 
                 n_comp, 
                 sigma=0.1,
                 sigma_trainable=True,
                 w_train=True,
                 generative=0.,
                 **kwargs):
        super().__init__(**kwargs) 
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.generative = generative
        self.encoder = encoder
        if generative > 0.:
            encoder.trainable = False
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma=sigma, 
                                         dim=encoded_size, 
                                         trainable=sigma_trainable)
        self.kdm = KDMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp,
                                       w_train=w_train,
                                       generative=generative)
    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y =self.kdm(rho_x)
        probs = dm2discrete(rho_y)
        return probs

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        encoded_x = self.encoder.predict(samples_x)
        if init_sigma:
            np_encoded_x = keras.ops.convert_to_numpy(encoded_x)
            distances = pairwise_distances(np_encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.kdm.c_x.assign(encoded_x)
        self.kdm.c_y.assign(samples_y)
        self.kdm.c_w.assign(keras.ops.ones((self.n_comp,)) / self.n_comp)

