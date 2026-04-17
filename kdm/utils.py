import math
import numpy as np
import torch


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
    return torch.cat((w.unsqueeze(-1), v), dim=2)


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
    nonzero = (samples != 0).any(dim=-1).to(samples.dtype)
    w = nonzero / nonzero.sum(dim=-1, keepdim=True)
    return comp2dm(w, samples)


def pure2dm(psi):
    '''
    Construct a factorized density matrix to represent a pure state
    Arguments:
     psi: tensor of shape (bs, d)
    Returns:
     dm: tensor of shape (bs, 1, d + 1)
    '''
    ones = torch.ones_like(psi[:, 0:1])
    dm = torch.cat((ones.unsqueeze(1), psi.unsqueeze(1)), dim=2)
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
    w = w / w.sum(dim=-1, keepdim=True)
    v = torch.nn.functional.normalize(v, p=2, dim=-1, eps=1e-12)
    probs = torch.einsum('...j,...ji->...i', w, v ** 2)
    probs = probs.clamp(0., 1.)
    return probs


def dm_rbf_loglik(x, dm, sigma):
    '''
    Calculates the log likelihood of a set of points x given a density
    matrix in a RKHS defined by a RBF kernel
    Arguments:
      x: tensor of shape (bs, d)
     dm: tensor of shape (bs, n, d + 1)
    sigma: scalar
    Returns:
        log_likelihood: tensor with shape (bs, )
    '''
    d = x.shape[-1]
    w, v = dm2comp(dm)  # Shape: (bs, n), (bs, n, d)
    dist = ((x.unsqueeze(1) - v) ** 2).sum(dim=-1)  # Shape: (bs, n)
    log_likelihood = torch.log(torch.einsum('...i,...i->...', w,
                                            torch.exp(-dist / (2 * sigma ** 2)) ** 2)
                               + 1e-12)
    coeff = d * torch.log(sigma + 1e-12) + d * math.log(math.pi) / 2
    log_likelihood = log_likelihood - coeff
    return log_likelihood


def dm_rbf_expectation(dm):
    '''
    Calculates the expectation of a density matrix in a RKHS defined by a RBF kernel
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    sigma: scalar
    Returns:
        expectation: tensor with shape (bs, d)
    '''
    w, v = dm2comp(dm)  # Shape: (bs, n), (bs, n, d)
    expectation = torch.einsum('...i,...ij->...j', w, v)
    return expectation


def dm_rbf_variance(dm, sigma):
    '''
    Calculates the sum of the variances along each dimension (the trace of the covariance)
    of a GMM-like density matrix in a RKHS defined by an RBF kernel.
    Each component of the mixture is assumed to have covariance sigma^2 * I.

    Arguments:
        dm: tensor of shape (bs, n, d + 1)
        sigma: scalar
    Returns:
        variance_trace: tensor of shape (bs,)
            The sum of variances along each dimension for each batch element.
    '''
    sigma = sigma / math.sqrt(2)

    w, v = dm2comp(dm)  # w: (bs, n), v: (bs, n, d)
    d = v.shape[-1]

    squared_norms = (v ** 2).sum(dim=-1)  # (bs, n)
    weighted_squared_norms = torch.einsum('...i,...i->...', w, squared_norms)  # (bs,)

    weighted_means = torch.einsum('...i,...ij->...j', w, v)  # (bs, d)
    squared_means = (weighted_means ** 2).sum(dim=-1)  # (bs,)

    between_component_variance = weighted_squared_norms - squared_means

    variance_trace = between_component_variance + d * (sigma ** 2)

    return variance_trace


def gauss_entropy_lb(d, sigma):
    '''
    Calculates Jensen's inequality-based lower bound on the entropy of a
    Gaussian mixture, given that each component is a d-dimensional Gaussian
    with covariance sigma^2 I. This bound does not depend on the mixture
    parameters (weights, means) and only depends on d and sigma.

    Arguments:
        d: int or scalar, the dimensionality of the Gaussian
        sigma: scalar, the sigma for each Gaussian component

    Returns:
        entropy_lb: scalar (or tensor), the entropy lower bound.
    '''
    entropy_lb = (d / 2.0) * (1.0 + torch.log(2.0 * math.pi * (sigma ** 2)))
    return entropy_lb


def cartesian_product(x):
    # x is a list of two tensors
    if len(x) == 1:
        return x[0]
    elif len(x) == 2:
        a, b = x
        a = a.unsqueeze(-1)  # Shape: (batch_size, num_classes, 1)
        b = b.unsqueeze(1)   # Shape: (batch_size, 1, num_classes)
        return (a * b).reshape(a.shape[0], -1)
    else:
        a, b = x[:2]
        a = a.unsqueeze(-1)  # Shape: (batch_size, num_classes_a, 1)
        b = b.unsqueeze(1)   # Shape: (batch_size, 1, num_classes_b)
        ab = (a * b).reshape(a.shape[0], -1)  # Shape: (batch_size, num_classes_a * num_classes_b)

        for i in range(2, len(x)):
            ab = ab.unsqueeze(-1)            # Shape: (batch_size, num_classes_ab, 1)
            c = x[i].unsqueeze(1)            # Shape: (batch_size, 1, num_classes_c)
            ab = (ab * c).reshape(ab.shape[0], -1)

        return ab


def pure_dm_overlap(x, dm, kernel):
    '''
    Calculates the overlap of a state \\phi(x) with a density
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
    overlap = torch.einsum('...i,...i->...', w, kernel(x, v) ** 2)
    return overlap
