import keras
import numpy as np
from ..layers import KDMLayer, RBFKernelLayer, CosineKernelLayer
from ..utils import pure2dm, dm2discrete, cartesian_product
from sklearn.metrics import pairwise_distances


class KDMSequentialJointClassModel(keras.Model):
    def __init__(self,
                 encoded_size,
                 dim_y,
                 encoder,
                 n_comp,
                 sigma=0.1,
                 sequences=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_y = dim_y
        #self.encoded_size = encoded_size
        #self.encoder = encoder
        self.n_comp = n_comp
        input = KDMLayer(kernel=
                         RBFKernelLayer(sigma=sigma,
                                               dim=encoded_size,
                                               trainable=True),
                         dim_x=encoded_size,
                         dim_y=dim_y,
                         n_comp=n_comp)
        model_sequences = [keras.Sequential(
            [input]
        )]
        for seq in sequences:
            model_sequence = []
            if not isinstance(seq, list) and seq['type'] == 'merge':
                model_sequences.append('merge')
            else:
                for layer in seq:
                    model_sequence.append(
                        KDMLayer(kernel=layer['kernel'],
                                 dim_x=layer['dim_x'],
                                 dim_y=layer['dim_y'],
                                 n_comp=layer['n_comp']
                                 )
                    )
                model_sequences.append(keras.Sequential(
                    model_sequence
                )
                )

        self.model = model_sequences

    def call(self, input):
        encoded = keras.layers.Identity()(input)
        rho_x = encoded
        rho_x = pure2dm(rho_x)
        rho_y = rho_x
        ans = []
        idx = 0
        for seq in self.model:
            idx += 1
            if seq == 'merge':
                merged_prbs = cartesian_product(ans)
                rho_x = merged_prbs
                rho_x = pure2dm(rho_x)
                rho_y = merged_prbs
                probs = rho_y
                ans = []
            else:
                rho_y = seq(rho_x)
                probs = dm2discrete(rho_y)
                ans.append(probs)
        return probs

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1, index=0, super_index=0):
        encoded_x = keras.layers.Identity()(samples_x)
        if init_sigma:
            np_encoded_x = keras.ops.convert_to_numpy(encoded_x)
            distances = pairwise_distances(np_encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.model[super_index].layers[index].kernel.sigma.assign(sigma)
        self.model[super_index].layers[index].c_x.assign(encoded_x)
        self.model[super_index].layers[index].c_y.assign(samples_y)
        self.model[super_index].layers[index].c_w.assign(
            keras.ops.ones((self.model[super_index].layers[index].n_comp,)) / self.model[super_index].layers[index].n_comp)
