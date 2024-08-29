import keras
import numpy as np

class MemKDMLayer(keras.layers.Layer):
    """Memory based Kernel Density Matrix Layer
    Receives as input a sample along with its nearest neighbors, which
    are retrieved from a vector memory.
    Returns a resulting KDM.
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        n_comp: int. Number of neighbors 
    Input shape:
        [samples, neighbors, labels]
            samples:(batch_size, dim_x)
            neighbors:(batch_size, n_comp, dim_x)
            labels:(batch_size, n_comp, dim_y)
    Output shape:
        (batch_size, n_comp, dim_y)
        The weights of the output KDM for sample i are at [i, :, 0], 
        and the components are at [i, :, 1:dim_y + 1].
    """
    def __init__(
            self,
            kernel,
            dim_x: int,
            dim_y: int,
            n_comp: int, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.eps = keras.config.epsilon()

    def call(self, inputs): 
        in_v, c_x, c_y = inputs 
        in_v = keras.ops.expand_dims(in_v, axis=1) # shape (bs, 1, dim_x)
        out_vw = self.kernel(in_v, c_x)  # shape (bs, 1, n_comp)
        out_w = keras.ops.square(out_vw) # shape (bs, 1, n_comp)
        out_w = keras.ops.maximum(out_w, self.eps)[:, 0, :] # shape (bs, n_comp) 
        # normalize out_w to sum to 1
        out_w = out_w / keras.ops.sum(out_w, axis=1, keepdims=True)
        out_w = keras.ops.expand_dims(out_w, axis=-1) # shape (b, n_comp, 1)
        out = keras.ops.concatenate((out_w, c_y), 2)
        return out

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "w_train": self.w_train,
            "l1_x": self.l1_x,
            "l1_y": self.l1_y,
            "l1_act": self.l1_act,
        }
        base_config = super().get_config()
        return {**base_config, **config}
