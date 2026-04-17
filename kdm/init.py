"""Data-driven initializers for KDM layers.

These replace the per-model `init_components` methods from the Keras
version. The logic is identical — sigma is set from the mean distance to
the k-th nearest neighbor in the encoded training set, and the component
tensors are overwritten with the encoded samples — but lives in one place
and operates on already-instantiated layers.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

__all__ = ["init_kdm_layer", "init_kdm_proj_layer"]


def _sigma_from_knn(points: np.ndarray, sigma_mult: float = 1.0) -> float:
    nn_model = NearestNeighbors(n_neighbors=3)
    nn_model.fit(points)
    distances, _ = nn_model.kneighbors(points)
    return float(np.mean(distances[:, 2]) * sigma_mult)


def init_kdm_layer(
    layer,
    encoded_x,
    samples_y,
    init_sigma: bool = False,
    sigma_mult: float = 1.0,
) -> None:
    """Initialize a KDMLayer's c_x, c_y, c_w from data.

    Arguments:
        layer: a KDMLayer (its n_comp must equal the number of rows in
               encoded_x and samples_y).
        encoded_x: (n_comp, dim_x) array / tensor of support points.
        samples_y: (n_comp, dim_y) array / tensor of target components.
        init_sigma: if True, set layer.kernel.sigma from the mean distance
                    to the 2nd-nearest neighbor of encoded_x.
        sigma_mult: scalar multiplier applied to the computed sigma.
    """
    encoded_x_t = torch.as_tensor(encoded_x)
    samples_y_t = torch.as_tensor(samples_y)
    if init_sigma:
        points = encoded_x_t.detach().cpu().numpy()
        layer.kernel.sigma = _sigma_from_knn(points, sigma_mult)
    with torch.no_grad():
        layer.c_x.copy_(encoded_x_t.to(layer.c_x))
        layer.c_y.copy_(samples_y_t.to(layer.c_y))
        layer.c_w.fill_(1.0 / layer.n_comp)


def init_kdm_proj_layer(
    layer,
    samples_x,
    init_sigma: bool = False,
    sigma_mult: float = 1.0,
) -> None:
    """Initialize a KDMProjLayer's c_x and c_w from data."""
    samples_x_t = torch.as_tensor(samples_x)
    if init_sigma:
        points = samples_x_t.detach().cpu().numpy()
        layer.kernel.sigma = _sigma_from_knn(points, sigma_mult)
    with torch.no_grad():
        layer.c_x.copy_(samples_x_t.to(layer.c_x))
        layer.c_w.fill_(1.0 / layer.n_comp)
