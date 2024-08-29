import keras
import numpy as np



def dm2comp(dm):
    '''
    Extract vectors and weights from a factorized density matrix representation
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    '''
    return dm[:, :, 0], dm[:, :, 1:]


def comp2dm(w, v):
    '''
    Construct a factorized density matrix from vectors and weights
    Arguments:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    Returns:
     dm: tensor of shape (bs, n, d + 1)
    '''
    return keras.ops.concatenate((w[:, :, np.newaxis], v), axis=2)

def samples2dm(samples):
    '''
    Construct a factorized density matrix from a batch of samples
    each sample will have the same weight. Samples that are all 
    zero will be ignored.
    Arguments:
        samples: tensor of shape (bs, n, d)
    Returns:
        dm: tensor of shape (bs, n, d + 1)
    '''
    w = keras.ops.any(samples, axis=-1)
    w = w / keras.ops.sum(w, axis=-1, keepdims=True)
    return comp2dm(w, samples)

def pure2dm(psi):
    '''
    Construct a factorized density matrix to represent a pure state
    Arguments:
     psi: tensor of shape (bs, d)
    Returns:
     dm: tensor of shape (bs, 1, d + 1)
    '''
    ones = keras.ops.ones_like(psi[:, 0:1])
    dm = keras.ops.concatenate((ones[:,np.newaxis, :],
                    psi[:,np.newaxis, :]),
                   axis=2)
    return dm


def dm2discrete(dm):
    '''
    Creates a discrete distribution from the components of a density matrix
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     prob: vector of probabilities (bs, d)
    '''
    w, v = dm2comp(dm)
    w = w / keras.ops.sum(w, axis=-1, keepdims=True)
    v = keras.utils.normalize(v, order=2, axis=-1)
    probs = keras.ops.einsum('...j,...ji->...i', w, v ** 2)
    probs = keras.ops.clip(probs, 0., 1.)
    return probs

def cartesian_product(x):
    # x is a list of two tensors
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        a, b = x
        a = keras.ops.expand_dims(a, axis=-1)  # Shape: (batch_size, num_classes, 1)
        b = keras.ops.expand_dims(b, axis=1)   # Shape: (batch_size, 1, num_classes)
        return keras.ops.reshape(a * b, (keras.ops.shape(a)[0], -1))
    else:
        a, b = x[:2]
        a = keras.ops.expand_dims(a, axis=-1)  # Shape: (batch_size, num_classes_a, 1)
        b = keras.ops.expand_dims(b, axis=1)   # Shape: (batch_size, 1, num_classes_b)
        ab = keras.ops.reshape(a * b, (keras.ops.shape(a)[0], -1))  # Shape: (batch_size, num_classes_a * num_classes_b)

        for i in range(2, len(x)):
            ab = keras.ops.expand_dims(ab, axis=-1)  # Shape: (batch_size, num_classes_ab, 1)
            c = keras.ops.expand_dims(x[i], axis=1)  # Shape: (batch_size, 1, num_classes_c)
            ab = keras.ops.reshape(ab * c, (keras.ops.shape(ab)[0], -1))  # Shape: (batch_size, num_classes_ab * num_classes_c)

        return ab


def pure_dm_overlap(x, dm, kernel):
    '''
    Calculates the overlap of a state  \phi(x) with a density 
    matrix in a RKHS defined by a kernel
    Arguments:
      x: tensor of shape (bs, d)
     dm: tensor of shape (bs, n, d + 1)
     kernel: kernel function 
              k: (bs, d) x (bs, n, d) -> (bs, n)
    Returns:
     overlap: tensor with shape (bs, )
    '''
    w, v = dm2comp(dm)
    overlap = keras.ops.einsum('...i,...i->...', w, kernel(x, v) ** 2)
    return overlap