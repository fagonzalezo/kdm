import keras
import torch
import math
import faiss
import numpy as np
from ..models import MemKDMClassModel
from ..utils import pure2dm, dm2discrete

class MemKDMClassModelWrapper:
    def __init__(self,
                 encoded_size,
                 dim_y,
                 samples_x,
                 samples_y,
                 encoder,
                 n_comp,
                 sigma=0.1,
                 **kargs):
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.samples_y = samples_y
        samples_x = torch.tensor(samples_x, dtype=torch.float32, device='cpu')
        dataset = torch.utils.data.TensorDataset(samples_x)
        nlist = 100
        self.index = faiss.IndexFlatL2(encoded_size)  
        #self.index = faiss.IndexHNSWFlat(encoded_size, 32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        self.samples_x_enc = np.zeros((samples_x.shape[0], encoded_size))
        i = 0
        for x_batch in dataloader:
            x_batch = x_batch[0]
            enc_batch = self.encoder(x_batch)
            enc_batch = keras.ops.convert_to_numpy(enc_batch)
            self.index.add(enc_batch)
            self.samples_x_enc[i:i+enc_batch.shape[0]] = enc_batch
            i += enc_batch.shape[0]
        self.model = MemKDMClassModel(
                 encoded_size,
                 dim_y,
                 n_comp,
                 sigma=sigma,
                 **kargs)

    def predict(self, X, batch_size=32):
        y_preds = []
        X = torch.tensor(X, dtype=torch.float32, device='cpu')
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for x_batch in dataloader:
            x_batch = x_batch[0]
            x_enc = self.encoder(x_batch)
            _, I = self.index.search(x_enc, self.n_comp)
            x_neigh = keras.ops.take(self.samples_x_enc, I, axis=0)
            y_neigh = keras.ops.take(self.samples_y, I, axis=0)
            y_pred = self.model((x_enc, x_neigh[:x_enc.shape[0], ...], y_neigh))
            y_preds.append(keras.ops.convert_to_numpy(y_pred))
        return np.concatenate(y_preds, axis=0)
    
    def init_sigma(self, mult=0.1, n_samples=100):
        '''
        Initialize the sigma parameter of the RBF kernel
        using the average distance between the nearest neighbors
        of a set of random samples
        '''
        n_samples = min(n_samples, self.samples_x_enc.shape[0])
        rng = np.random.default_rng()
        samples_x = rng.choice(self.samples_x_enc, 
                                n_samples, 
                                replace=False)
        dists, I = self.index.search(samples_x, self.n_comp + 1)
        x_neigh = np.take(self.samples_x_enc, I, axis=0)
        dists_1 = np.linalg.norm(samples_x[:, None, :] - x_neigh, axis=-1)
        #sigma = np.mean(np.sqrt(dists[:, 1:])) * mult
        sigma = np.mean(dists_1[:, 1:]) * mult
        self.model.kernel.sigma.assign(sigma)
        return sigma
                          
    def compile(self, optimizer, loss, metrics=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, batch_size=32, epochs=1, verbose=1):
        dataset = CustomDataset(self.samples_x_enc, 
                                self.samples_y, 
                                self.index, 
                                self.n_comp,
                                batch_size=batch_size)
        return self.model.fit(dataset, epochs=epochs, verbose=verbose)

class CustomDataset(keras.utils.PyDataset):
    def __init__(self, samples_x_enc, 
                 samples_y, index, 
                 n_comp, batch_size, 
                 **kwargs):
        super().__init__(**kwargs)
        self.samples_x_enc = samples_x_enc
        self.samples_y = samples_y
        self.index = index
        self.n_comp = n_comp
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.samples_y) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.samples_y))
        x_enc = self.samples_x_enc[low:high]
        _, I = self.index.search(x_enc, self.n_comp + 1)
        x_neigh = np.take(self.samples_x_enc, I, axis=0)
        y_neigh = np.take(self.samples_y, I, axis=0)
        return (x_enc, x_neigh[:, 1:], y_neigh[:,1:]), self.samples_y[low:high]
