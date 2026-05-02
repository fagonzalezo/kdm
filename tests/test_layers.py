"""
Functional tests for kdm/layers/ (PyTorch).

Each test validates a mathematical property or behavioral invariant using
inline data only — no external fixture files required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kdm.layers import (
    CosineKernelLayer,
    CrossProductKernelLayer,
    KDMLayer,
    KDMProjLayer,
    MemKDMLayer,
    MemRBFKernelLayer,
    RBFKernelLayer,
)
from kdm.utils import pure2dm

RNG = np.random.default_rng(2024)

DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["float64", "float32"]


def _rand(shape, dtype):
    return torch.as_tensor(RNG.standard_normal(shape), dtype=dtype)


def _cast_layer(layer: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    # Re-assign sigma after cast so softplus_inv runs at the new dtype.
    rbf_sigmas = [
        (m, float(m.sigma.detach()))
        for m in layer.modules()
        if isinstance(m, RBFKernelLayer)
    ]
    layer = layer.double() if dtype == torch.float64 else layer.float()
    for m, s in rbf_sigmas:
        m.sigma = s
    return layer


# ---------------------------------------------------------------------------
# RBFKernelLayer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel_shape(dtype):
    layer = _cast_layer(RBFKernelLayer(sigma=0.5, dim=3), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    K = layer(A, B)
    assert K.shape == (4, 5, 7)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel_self_similarity(dtype):
    # K(x, x) = exp(0) = 1 for every x.
    layer = _cast_layer(RBFKernelLayer(sigma=0.5, dim=3), dtype)
    A = _rand((1, 5, 3), dtype)
    K = layer(A, A[0])  # square kernel matrix
    diag = K[0].diagonal()
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(diag, torch.ones_like(diag), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel_symmetry(dtype):
    # K(A, B) == K(B, A).T when A and B contain the same vectors (square case).
    layer = _cast_layer(RBFKernelLayer(sigma=0.5, dim=3), dtype)
    A = _rand((1, 5, 3), dtype)
    K = layer(A, A[0])
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(K[0], K[0].T, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel_values_in_range(dtype):
    layer = _cast_layer(RBFKernelLayer(sigma=0.5, dim=3), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    K = layer(A, B)
    assert torch.all(K > 0)
    assert torch.all(K <= 1.0 + 1e-6)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel_log_weight_formula(dtype):
    # Verify the formula: log_weight = −d·log(σ+ε) − d·log(π)/2
    # Use the layer's actual sigma (post-softplus) to avoid roundtrip precision issues.
    d = 3
    layer = _cast_layer(RBFKernelLayer(sigma=0.7, dim=d), dtype)
    sigma_actual = layer.sigma.detach()
    result = layer.log_weight()
    expected = -d * torch.log(sigma_actual + 1e-12) - d * math.log(math.pi) / 2.0
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(result, expected, atol=atol)


def test_rbf_kernel_sigma_property():
    layer = RBFKernelLayer(sigma=1.0, dim=3)
    layer.sigma = 0.5
    assert abs(layer.sigma.detach().item() - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# CosineKernelLayer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cosine_kernel_shape(dtype):
    layer = _cast_layer(CosineKernelLayer(), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    K = layer(A, B)
    assert K.shape == (4, 5, 7)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cosine_kernel_range(dtype):
    layer = _cast_layer(CosineKernelLayer(), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    K = layer(A, B)
    assert torch.all(K >= -1.0 - 1e-6)
    assert torch.all(K <= 1.0 + 1e-6)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cosine_kernel_self_similarity(dtype):
    layer = _cast_layer(CosineKernelLayer(), dtype)
    A = _rand((1, 5, 3), dtype)
    K = layer(A, A[0])
    diag = K[0].diagonal()
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(diag, torch.ones_like(diag), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cosine_kernel_scale_invariant(dtype):
    layer = _cast_layer(CosineKernelLayer(), dtype)
    A = _rand((4, 5, 3), dtype)
    B = _rand((7, 3), dtype)
    K1 = layer(A, B)
    K2 = layer(2.0 * A, B)
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(K1, K2, atol=atol)


# ---------------------------------------------------------------------------
# CrossProductKernelLayer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_crossproduct_kernel_shape(dtype):
    k1 = RBFKernelLayer(sigma=0.5, dim=2)
    k2 = RBFKernelLayer(sigma=0.8, dim=3)
    layer = _cast_layer(CrossProductKernelLayer(dim1=2, kernel1=k1, kernel2=k2), dtype)
    A = _rand((4, 5, 5), dtype)
    B = _rand((7, 5), dtype)
    K = layer(A, B)
    assert K.shape == (4, 5, 7)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_crossproduct_kernel_factorization(dtype):
    dim1 = 2
    k1 = _cast_layer(RBFKernelLayer(sigma=0.5, dim=dim1), dtype)
    k2 = _cast_layer(RBFKernelLayer(sigma=0.8, dim=3), dtype)
    layer = _cast_layer(
        CrossProductKernelLayer(dim1=dim1, kernel1=k1, kernel2=k2), dtype
    )
    A = _rand((4, 5, 5), dtype)
    B = _rand((7, 5), dtype)
    K_cp = layer(A, B)
    K1 = k1(A[:, :, :dim1], B[:, :dim1])
    K2 = k2(A[:, :, dim1:], B[:, dim1:])
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(K_cp, K1 * K2, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_crossproduct_kernel_log_weight(dtype):
    k1 = RBFKernelLayer(sigma=0.5, dim=2)
    k2 = RBFKernelLayer(sigma=0.8, dim=3)
    layer = _cast_layer(CrossProductKernelLayer(dim1=2, kernel1=k1, kernel2=k2), dtype)
    k1c = list(layer.modules())[1]
    k2c = list(layer.modules())[2]
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(
        layer.log_weight(), k1c.log_weight() + k2c.log_weight(), atol=atol
    )


# ---------------------------------------------------------------------------
# KDMProjLayer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_proj_layer_shape(dtype):
    kernel = RBFKernelLayer(sigma=0.5, dim=3)
    layer = _cast_layer(KDMProjLayer(kernel=kernel, dim_x=3, n_comp=7), dtype)
    x = _rand((4, 3), dtype)
    out = layer(x)
    assert out.shape == (4,)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_proj_layer_positive(dtype):
    kernel = RBFKernelLayer(sigma=0.5, dim=3)
    layer = _cast_layer(KDMProjLayer(kernel=kernel, dim_x=3, n_comp=7), dtype)
    x = _rand((4, 3), dtype)
    out = layer(x)
    assert torch.all(out > 0)


def test_kdm_proj_layer_gradient_flows():
    kernel = RBFKernelLayer(sigma=0.5, dim=3)
    layer = _cast_layer(KDMProjLayer(kernel=kernel, dim_x=3, n_comp=7), torch.float64)
    x = _rand((4, 3), torch.float64).requires_grad_(True)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# KDMLayer
# ---------------------------------------------------------------------------

def _make_kdm_layer(dtype, dim_x=3, dim_y=6, n_comp=7, sigma=0.5):
    kernel = RBFKernelLayer(sigma=sigma, dim=dim_x)
    layer = KDMLayer(kernel=kernel, dim_x=dim_x, dim_y=dim_y, n_comp=n_comp)
    return _cast_layer(layer, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_output_shape(dtype):
    layer = _make_kdm_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    rho_out = layer(rho_in)
    assert rho_out.shape == (4, 7, 7)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_weights_sum_to_one(dtype):
    layer = _make_kdm_layer(dtype)
    # pure2dm gives a valid KDM with input weights summing to 1.
    rho_in = pure2dm(_rand((4, 3), dtype))
    rho_out = layer(rho_in)
    weights_sum = rho_out[:, :, 0].sum(dim=-1)
    atol = 1e-5
    assert torch.allclose(weights_sum, torch.ones(4, dtype=dtype), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_weights_nonneg(dtype):
    layer = _make_kdm_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    rho_out = layer(rho_in)
    assert torch.all(rho_out[:, :, 0] >= 0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_log_marginal_shape(dtype):
    layer = _make_kdm_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    log_probs = layer.log_marginal(rho_in)
    assert log_probs.shape == (4,)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_log_marginal_finite(dtype):
    layer = _make_kdm_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    log_probs = layer.log_marginal(rho_in)
    assert torch.isfinite(log_probs).all()


def test_kdm_layer_gradient_flows():
    layer = _make_kdm_layer(torch.float64)
    rho_in = pure2dm(_rand((4, 3), torch.float64))
    log_probs = layer.log_marginal(rho_in)
    log_probs.sum().backward()
    assert layer.c_x.grad is not None
    assert layer.kernel.raw_sigma.grad is not None
    assert torch.isfinite(layer.c_x.grad).all()


# ---------------------------------------------------------------------------
# MemKDMLayer
# ---------------------------------------------------------------------------

def _make_mem_layer(dtype, dim_x=3, dim_y=6, n_comp=8, sigma=0.5):
    kernel = MemRBFKernelLayer(sigma=sigma, dim=dim_x)
    layer = MemKDMLayer(kernel=kernel, dim_x=dim_x, dim_y=dim_y, n_comp=n_comp)
    return _cast_layer(layer, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mem_kdm_layer_shape(dtype):
    layer = _make_mem_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    neighbors = _rand((4, 8, 3), dtype)
    labels = _rand((4, 8, 6), dtype)
    out = layer((rho_in, neighbors, labels))
    assert out.shape == (4, 8, 7)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mem_kdm_layer_weights_valid(dtype):
    layer = _make_mem_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    neighbors = _rand((4, 8, 3), dtype)
    labels = _rand((4, 8, 6), dtype)
    out = layer((rho_in, neighbors, labels))
    weights = out[:, :, 0]
    assert torch.all(weights >= 0)
    atol = 1e-5
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, dtype=dtype), atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mem_kdm_layer_labels_passthrough(dtype):
    # The vector slice of the output KDM is exactly the input labels.
    layer = _make_mem_layer(dtype)
    rho_in = pure2dm(_rand((4, 3), dtype))
    neighbors = _rand((4, 8, 3), dtype)
    labels = _rand((4, 8, 6), dtype)
    out = layer((rho_in, neighbors, labels))
    assert torch.equal(out[:, :, 1:], labels)


def test_mem_kdm_layer_gradient_flows():
    layer = _make_mem_layer(torch.float64)
    sample = _rand((4, 3), torch.float64).requires_grad_(True)
    rho_in = pure2dm(sample)
    neighbors = _rand((4, 8, 3), torch.float64)
    labels = _rand((4, 8, 6), torch.float64)
    out = layer((rho_in, neighbors, labels))
    out.sum().backward()
    assert sample.grad is not None
    assert torch.isfinite(sample.grad).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mem_kdm_layer_mixed_input(dtype):
    # Verify that a multi-component input KDM (n_comp_in > 1) is handled
    # correctly: output shape is unchanged and weights still sum to 1.
    layer = _make_mem_layer(dtype, n_comp=8)
    n_comp_in = 3
    raw_w = _rand((4, n_comp_in), dtype).abs() + 0.1
    in_w = raw_w / raw_w.sum(dim=1, keepdim=True)           # (4, n_comp_in), sums to 1
    in_v = _rand((4, n_comp_in, 3), dtype)
    rho_in = torch.cat((in_w.unsqueeze(-1), in_v), dim=2)   # (4, n_comp_in, dim_x+1)
    neighbors = _rand((4, 8, 3), dtype)
    labels = _rand((4, 8, 6), dtype)
    out = layer((rho_in, neighbors, labels))
    assert out.shape == (4, 8, 7)
    atol = 1e-5
    assert torch.allclose(out[:, :, 0].sum(dim=-1), torch.ones(4, dtype=dtype), atol=atol)


def test_mem_kdm_layer_log_marginal():
    layer = _make_mem_layer(torch.float64)
    sample = _rand((4, 3), torch.float64).requires_grad_(True)
    rho_in = pure2dm(sample)
    neighbors = _rand((4, 8, 3), torch.float64)
    lm = layer.log_marginal(rho_in, neighbors)
    assert lm.shape == (4,)
    assert torch.isfinite(lm).all()
    lm.sum().backward()
    assert sample.grad is not None
    assert torch.isfinite(sample.grad).all()
