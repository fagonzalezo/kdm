import keras
import numpy as np
from ..layers import KDMLayer, RBFKernelLayer
from ..utils import pure2dm, dm_rbf_loglik, gauss_entropy_lb, dm2comp
from sklearn.neighbors import NearestNeighbors


class KDMRegressModel(keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 encoder, 
                 n_comp, 
                 sigma_x=0.1,
                 sigma_y=0.1,
                 w_train=True,
                 generative=0.,
                 entropy_reg_x=0.,
                 **kwargs):
        super().__init__(**kwargs) 
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        if generative > 0.:
            encoder.trainable = False
        self.entropy_reg_x = entropy_reg_x
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma=sigma_x, 
                                         dim=encoded_size, 
                                         trainable=True)
        self.kdm = KDMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp,
                                       w_train=w_train,
                                       generative=generative)
        self.sigma_y = self.add_weight(
            shape=(),
            initializer=keras.initializers.constant(sigma_y),
            trainable=True,
            name="sigma_y")

    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y = self.kdm(rho_x)
        return rho_y
    
    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        encoded_x = self.encoder(samples_x)
        if init_sigma:
            np_encoded_x = keras.ops.convert_to_numpy(encoded_x)
            nn_model = NearestNeighbors(n_neighbors=3)
            nn_model.fit(np_encoded_x)
            distances, _ = nn_model.kneighbors(np_encoded_x)
            sigma = np.mean(distances[:, 2]) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.kdm.c_x.assign(encoded_x)
        self.kdm.c_y.assign(samples_y)
        self.kdm.c_w.assign(keras.ops.ones((self.n_comp,)) / self.n_comp)

    def loglik(self, y_true, y_pred):
        sigma = keras.ops.clip(self.sigma_y, self.kdm.kernel.min_sigma, np.inf) / np.sqrt(2)
        return -keras.ops.mean(dm_rbf_loglik(y_true, y_pred, sigma))

    def loglik_lb_1(self, y_true, y_pred):
        sigma = self.sigma_y / np.sqrt(2)
        d = keras.ops.shape(y_true)[-1]
        w, v = dm2comp(y_pred) # Shape: (bs, n), (bs, n, d)
        dist = keras.ops.sum((y_true[:, np.newaxis, :] - v) ** 2, axis=-1) # Shape: (bs, n)
        log_likelihood = keras.ops.einsum('...i,...i->...', w, 
                                     -dist / (2 * sigma ** 2))
        coeff = d * keras.ops.log(sigma + 1e-12) + d * np.log(4 * np.pi)
        log_likelihood = log_likelihood - coeff   
        return - keras.ops.mean(log_likelihood)

    def loglik_lb_2(self, y_true, y_pred):
        sigma = self.sigma_y / np.sqrt(2)
        d = keras.ops.shape(y_true)[-1]
        w, v = dm2comp(y_pred) # Shape: (bs, n), (bs, n, d)
        expectation = keras.ops.einsum('...i,...ij->...j', w, v)
        dist = keras.ops.sum((y_true - expectation) ** 2, axis=-1) # Shape: (bs, 1)
        log_likelihood = - dist / (2 * sigma ** 2) 
        coeff = d * keras.ops.log(sigma + 1e-12) + d * np.log(4 * np.pi)
        log_likelihood = log_likelihood - coeff   
        return - keras.ops.mean(log_likelihood)

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        loss = self.loglik(y, y_pred)
        if len(self.losses) > 0:
            loss += keras.ops.sum(self.losses)
        if self.entropy_reg_x > 0.:
            loss -= self.entropy_reg_x * gauss_entropy_lb(self.encoded_size, 
                                                        self.kernel.sigma / np.sqrt(2))
        #self.loss_tracker.update_state(loss)
        return loss