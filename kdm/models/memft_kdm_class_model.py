import keras
import faiss
import numpy as np
from ..layers import MemRBFKernelLayer, MemKDMLayer
from ..utils import pure2dm, dm2discrete
import torch

class MemFTKDMClassModel(keras.Model):
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 samples_x,
                 samples_y,
                 encoder, 
                 n_comp,
                 sigma=0.1, 
                 **kargs):
        super().__init__(**kargs) 
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.samples_x = samples_x
        self.samples_y = samples_y
        self.kernel = MemRBFKernelLayer(sigma=sigma, 
                                         dim=encoded_size, 
                                         trainable=True)
        self.mkdm = MemKDMLayer(kernel=self.kernel, 
                                       dim_x=encoded_size,
                                       dim_y=dim_y, 
                                       n_comp=n_comp)
        
    def call(self, input):
        x, neighbors = input
        x_enc = self.encoder(x)
        neighbors = keras.ops.cast(neighbors, "int32")
        x_neigh = keras.ops.take(self.samples_x, neighbors, axis=0)
        x_neigh = keras.ops.reshape(x_neigh, (-1, x.shape[-1]))
        x_neigh = self.encoder(x_neigh)
        x_neigh = keras.ops.reshape(x_neigh, (-1, self.n_comp, self.encoded_size))
        y_neigh = keras.ops.take(self.samples_y, neighbors, axis=0)
        rho_y = self.mkdm([x_enc, x_neigh, y_neigh])
        probs = dm2discrete(rho_y)
        return probs

    
class MemftKdmClassWrapper:
    def __init__(self, 
                 encoded_size, 
                 dim_y, 
                 samples_x,
                 samples_y,
                 encoder, 
                 n_comp,
                 sigma=0.1,
                 **kargs):
        samples_x = keras.ops.convert_to_tensor(samples_x, dtype="float32")
        samples_y = keras.ops.convert_to_tensor(samples_y, dtype="float32")
        self.model = MemFTKDMClassModel(
                 encoded_size, 
                 dim_y, 
                 samples_x,
                 samples_y,
                 encoder, 
                 n_comp,
                 sigma=sigma,
                 **kargs)
        self.index = faiss.IndexFlatL2(encoded_size)
        enc_samples_x = keras.ops.convert_to_tensor(self.model.encoder.predict(samples_x), dtype="float32")
        self.index.add(enc_samples_x)

    def predict(self, X, batch_size=32):
        y_preds = []
        for i in range(0, X.shape[0], batch_size):
            x_batch = keras.ops.convert_to_tensor(X[i:i+batch_size], dtype="float32")
            enc_batch = self.model.encoder(x_batch).detach().numpy()
            _, I = self.index.search(enc_batch, self.model.n_comp)
            neighbors = keras.ops.convert_to_tensor(I, dtype="int32")
            y_pred = self.model((x_batch, neighbors), training=True)
            y_preds.append(keras.ops.convert_to_numpy(y_pred))
        return np.concatenate(y_preds, axis=0)


    def fit(self,batch_size=32, epochs=100, optimizer=None, recalc_index_every=10):
        '''
        Implements a Pytorch training loop
        '''
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            optimizer = optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            if epoch % recalc_index_every == 1:
                self.index.reset()
                enc_samples_x = keras.ops.convert_to_tensor(
                    self.model.encoder.predict(self.model.samples_x), dtype="float32")
                self.index.add(enc_samples_x)
            p = np.random.permutation(self.model.samples_x.shape[0])
            idx = keras.ops.convert_to_tensor(p, dtype="int32")
            running_loss = 0
            count = 0
            for i in range(0, idx.shape[0], batch_size):
                x_batch = self.model.samples_x[idx[i:i+batch_size]]
                y_batch = self.model.samples_y[idx[i:i+batch_size]]
                enc_batch = self.model.encoder(x_batch).detach().numpy()
                _, I = self.index.search(enc_batch, self.model.n_comp + 1)
                neighbors = keras.ops.convert_to_tensor(I[:, 1:], dtype="int32")
                optimizer.zero_grad()
                y_pred = self.model((x_batch, neighbors), training=True)
                #print(y_pred)
                loss = loss_fn(y_pred, y_batch)
                running_loss += loss.detach().numpy()
                count += 1
                loss.backward()
                optimizer.step()
            print(epoch, running_loss/count)
        return self.model

    def score(self, X, y):
        return self.model.score(X, y)
