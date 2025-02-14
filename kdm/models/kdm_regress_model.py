import keras
import numpy as np
from ..layers import KDMLayer, RBFKernelLayer
from ..utils import pure2dm, dm_rbf_loglik, gauss_entropy_lb, dm_rbf_expectation, dm_rbf_variance
from sklearn.neighbors import NearestNeighbors


class KDMRegressModel(keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 encoder, 
                 n_comp, 
                 sigma_x=0.1,
                 min_sigma_x=1e-3,
                 sigma_y=0.1,
                 min_sigma_y=1e-3,
                 x_train=True,
                 y_train=True,
                 w_train=True,
                 generative=0.,
                 entropy_reg_x=0.,
                 sigma_x_trainable=True,
                 sigma_y_trainable=True,
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
                                         trainable=sigma_x_trainable,
                                         min_sigma=min_sigma_x)
        self.kdm = KDMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp,
                                       x_train=x_train,
                                       y_train=y_train,
                                       w_train=w_train,
                                       generative=generative)
        self.sigma_y = self.add_weight(
            shape=(),
            initializer=keras.initializers.constant(sigma_y),
            trainable=sigma_y_trainable,
            name="sigma_y")
        self.min_sigma_y = min_sigma_y

    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y = self.kdm(rho_x)
        self.sigma_y.assign(keras.ops.clip(self.sigma_y, self.min_sigma_y, np.inf))
        return rho_y
    
    def predict_reg(self, input, **kwargs):
        rho_y = self.predict(input, **kwargs)
        y_exp = keras.ops.convert_to_numpy(dm_rbf_expectation(rho_y))
        y_var = keras.ops.convert_to_numpy(dm_rbf_variance(rho_y, self.sigma_y))
        return y_exp, y_var
    
    def get_sigmas(self):
        sigma_y = keras.ops.convert_to_numpy(self.sigma_y)
        sigma_x = keras.ops.convert_to_numpy(self.kernel.sigma)
        return sigma_x, sigma_y
    
    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        encoded_x = self.encoder.predict(samples_x)
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
        return -keras.ops.mean(dm_rbf_loglik(y_true, y_pred,self.sigma_y))

    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        loss = self.loglik(y, y_pred)
        if len(self.losses) > 0:
            loss += keras.ops.sum(self.losses)
        if self.entropy_reg_x > 0.:
            loss -= self.entropy_reg_x * gauss_entropy_lb(self.encoded_size, 
                                                        self.kernel.sigma / np.sqrt(2))
        #self.loss_tracker.update_state(loss)
        return loss