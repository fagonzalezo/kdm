import keras
import math
import faiss
import numpy as np
from ..mem import MemKDMClassModel
from ...utils import dm2comp

class MemKDMClassModelWrapper:
    def __init__(self,
                 encoded_size,
                 dim_y,
                 samples_x,
                 samples_y,
                 encoder,
                 n_comp,
                 index_type='Flat',
                 sigma=0.1,
                 **kargs):
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.samples_y = samples_y
        dataloader = TestDataset(samples_x, batch_size=32)
        self.samples_x_enc = np.zeros((samples_x.shape[0], encoded_size), dtype=np.float32)
        i = 0
        for batch in range(len(dataloader)):
            x_batch = dataloader[batch]
            enc_batch = self.encoder(x_batch)
            enc_batch = keras.ops.convert_to_numpy(enc_batch)
            self.samples_x_enc[i:i+enc_batch.shape[0]] = enc_batch
            i += enc_batch.shape[0]
        self.index = faiss.index_factory(encoded_size, index_type)
        print('Building index...')
        self.index.train(self.samples_x_enc)
        print('Adding samples to index...')
        self.index.add(self.samples_x_enc)
        self.model = MemKDMClassModel(
                 encoded_size,
                 dim_y,
                 n_comp,
                 sigma=sigma,
                 **kargs)

    def predict(self, X, batch_size=32):
        y_preds = []
        dataloader = TestDataset(X, batch_size=batch_size)
        for batch in range(len(dataloader)):
            x_batch = dataloader[batch]
            x_enc = self.encoder(x_batch)
            x_enc = keras.ops.convert_to_numpy(x_enc)
            _, I = self.index.search(x_enc, self.n_comp)
            x_neigh = keras.ops.take(self.samples_x_enc, I, axis=0)
            y_neigh = keras.ops.take(self.samples_y, I, axis=0)
            y_pred = self.model((x_enc, x_neigh, y_neigh))
            y_preds.append(keras.ops.convert_to_numpy(y_pred))
        return np.concatenate(y_preds, axis=0)
    
    def predict_explain(self, x, n_neighbors):
        x_enc = self.encoder(x[None, ...])
        x_enc = keras.ops.convert_to_numpy(x_enc)
        _, I = self.index.search(x_enc, self.n_comp)
        x_neigh = keras.ops.take(self.samples_x_enc, I, axis=0)
        y_neigh = keras.ops.take(self.samples_y, I, axis=0)
        y_neigh_ohe = keras.ops.one_hot(y_neigh, self.dim_y)
        rho_y = self.model.mkdm((x_enc, x_neigh, y_neigh_ohe))
        w, _ = dm2comp(rho_y)
        w = keras.ops.convert_to_numpy(w)
        idx = np.argsort(w, axis=1)[:, ::-1]
        idx = idx[:, :n_neighbors]
        return (np.take_along_axis(I, idx, axis=1), 
                np.take_along_axis(w, idx, axis=1))


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
        _, I = self.index.search(samples_x, self.n_comp + 1)
        x_neigh = np.take(self.samples_x_enc, I, axis=0)
        dists_1 = np.linalg.norm(samples_x[:, None, :] - x_neigh, axis=-1)
        sigma = np.median(dists_1[:, 1:]) * mult
        self.model.kernel.sigma.assign(sigma)
        return sigma
                          
    def compile(self, optimizer, loss, metrics=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X=None, y=None, batch_size=32, epochs=1, verbose=1, callbacks=None):
        '''
        Fit the model using the given data. If X is None, it will use the samples
        used to build the index. Otherwise it will use the given data. 
        '''
        if X is None:
            dataset = TrainDataset(mkdm_model=self,
                                   batch_size=batch_size,
                                   use_index_samples=True)
            return self.model.fit(dataset, epochs=epochs, verbose=verbose, callbacks=callbacks)
        else:
            assert y is not None and len(X) == len(y), 'X and y must have the same length'
            dataset = TrainDataset(mkdm_model=self,
                                   samples_x=X,
                                   samples_y=y, 
                                   batch_size=batch_size,
                                   use_index_samples=False)
            return self.model.fit(dataset, epochs=epochs, verbose=verbose, callbacks=callbacks)
        
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
            
class TrainDataset(keras.utils.PyDataset):
    def __init__(self,  
                 mkdm_model,
                 samples_x=None, 
                 samples_y=None,
                 batch_size=32,
                 use_index_samples=True,
                 **kwargs):
        super().__init__(**kwargs)
        if use_index_samples:
            self.samples_y = mkdm_model.samples_y
        else:
            assert samples_x is not None and samples_y is not None, 'samples_x and samples_y must be provided'
            self.samples_x = samples_x
            self.samples_y = samples_y            
        self.batch_size = batch_size
        self.mkdm_model = mkdm_model
        self.use_index_samples = use_index_samples


    def __len__(self):
        return math.ceil(len(self.samples_y) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.samples_y))
        if self.use_index_samples:
            x_enc = self.mkdm_model.samples_x_enc[low:high]
            n_comp = self.mkdm_model.n_comp + 1
        else:
            x_enc = self.mkdm_model.encoder(self.samples_x[low:high])
            n_comp = self.mkdm_model.n_comp
        x_enc = keras.ops.convert_to_numpy(x_enc)
        _, I = self.mkdm_model.index.search(x_enc, n_comp)
        x_neigh = np.take(self.mkdm_model.samples_x_enc, I, axis=0)
        y_neigh = np.take(self.mkdm_model.samples_y, I, axis=0)
        n_comp = self.mkdm_model.n_comp
        return (x_enc, x_neigh[:, -n_comp:], y_neigh[:, -n_comp:]), self.samples_y[low:high]

class TestDataset(keras.utils.PyDataset):
    def __init__(self,  
                 samples_x,
                 batch_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.samples_x = samples_x
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.samples_x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, len(self.samples_x))
        return self.samples_x[low:high]
