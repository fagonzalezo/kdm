"""
Functional tests for kdm/utils.py (PyTorch).

Each test validates a mathematical property or behavioral invariant of the
utility functions, using inline data only — no external fixture files required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kdm import utils

RNG = np.random.default_rng(42)

DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["float64", "float32"]


def _t(arr, dtype=torch.float64):
    return torch.as_tensor(np.asarray(arr), dtype=dtype)


def _rand(shape, dtype):
    return torch.as_tensor(RNG.standard_normal(shape), dtype=dtype)


def _valid_dm(bs, n, d, dtype):
    """Build a KDM with valid (non-negative, normalized) weights."""
    w = torch.softmax(_rand((bs, n), dtype), dim=-1)
    v = _rand((bs, n, d), dtype)
    return torch.cat([w.unsqueeze(-1), v], dim=2)


# ---------------------------------------------------------------------------
# dm2comp / comp2dm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm2comp_roundtrip(dtype):
    dm = _rand((4, 5, 4), dtype)
    w, v = utils.dm2comp(dm)
    assert w.shape == (4, 5)
    assert v.shape == (4, 5, 3)
    dm2 = utils.comp2dm(w, v)
    assert dm2.shape == dm.shape
    assert torch.equal(dm, dm2)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_comp2dm_structure(dtype):
    w = _rand((4, 5), dtype)
    v = _rand((4, 5, 3), dtype)
    dm = utils.comp2dm(w, v)
    assert torch.equal(dm[:, :, 0], w)
    assert torch.equal(dm[:, :, 1:], v)


# ---------------------------------------------------------------------------
# pure2dm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_pure2dm_shape_and_weight(dtype):
    psi = _rand((4, 3), dtype)
    dm = utils.pure2dm(psi)
    assert dm.shape == (4, 1, 4)
    assert torch.all(dm[:, 0, 0] == 1.0)
    assert torch.equal(dm[:, 0, 1:], psi)


# ---------------------------------------------------------------------------
# dm2discrete
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm2discrete_valid_distribution(dtype):
    dm = _valid_dm(4, 5, 3, dtype)
    probs = utils.dm2discrete(dm)
    assert probs.shape == (4, 3)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4, dtype=dtype), atol=1e-5)


def test_dm2discrete_pure_state():
    # A pure state on the k-th standard basis vector should concentrate on k.
    d = 4
    k = 2
    psi = torch.zeros(1, d)
    psi[0, k] = 1.0
    dm = utils.pure2dm(psi)
    probs = utils.dm2discrete(dm)
    assert probs.argmax(dim=-1).item() == k


# ---------------------------------------------------------------------------
# dm_rbf_loglik
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_loglik_shape_and_dtype(dtype):
    x = _rand((4, 3), dtype)
    dm = _valid_dm(4, 5, 3, dtype)
    sigma = torch.tensor(0.7, dtype=dtype)
    out = utils.dm_rbf_loglik(x, dm, sigma)
    assert out.shape == (4,)
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_loglik_monotone(dtype):
    # Loglik should be higher when query matches the support than when far away.
    mu = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    dm = utils.pure2dm(mu)  # single-component DM at mu
    sigma = torch.tensor(1.0, dtype=dtype)
    loglik_near = utils.dm_rbf_loglik(mu, dm, sigma)
    far = mu + 100.0
    loglik_far = utils.dm_rbf_loglik(far, dm, sigma)
    assert loglik_near.item() > loglik_far.item()


def test_dm_rbf_loglik_gradient_flows():
    psi = torch.randn(3, 4, requires_grad=True, dtype=torch.float64)
    dm = utils.pure2dm(psi)
    x = torch.randn(3, 4, dtype=torch.float64)
    sigma = torch.tensor(1.0, dtype=torch.float64)
    out = utils.dm_rbf_loglik(x, dm, sigma)
    out.sum().backward()
    assert psi.grad is not None
    assert torch.isfinite(psi.grad).all()


# ---------------------------------------------------------------------------
# dm_rbf_expectation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_expectation_single_component(dtype):
    psi = _rand((4, 3), dtype)
    dm = utils.pure2dm(psi)
    exp = utils.dm_rbf_expectation(dm)
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(exp, psi, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_expectation_weighted_mean(dtype):
    v1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)
    v2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype)
    w = torch.tensor([[0.3, 0.7]], dtype=dtype)
    v = torch.cat([v1.unsqueeze(1), v2.unsqueeze(1)], dim=1)
    dm = utils.comp2dm(w, v)
    exp = utils.dm_rbf_expectation(dm)
    expected = (0.3 * v1 + 0.7 * v2)
    atol = 1e-12 if dtype == torch.float64 else 1e-6
    assert torch.allclose(exp, expected, atol=atol)


# ---------------------------------------------------------------------------
# dm_rbf_variance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_variance_nonnegative(dtype):
    # Only guaranteed non-negative for valid KDMs (non-negative weights summing to 1).
    dm = _valid_dm(4, 5, 3, dtype)
    sigma = torch.tensor(0.7, dtype=dtype)
    var = utils.dm_rbf_variance(dm, sigma)
    assert torch.all(var >= 0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_variance_single_component(dtype):
    # For a pure state, between-component variance is 0.
    # Total = d * (sigma / sqrt(2))^2 = d * sigma^2 / 2
    d = 4
    sigma_val = 0.7
    psi = _rand((3, d), dtype)
    dm = utils.pure2dm(psi)
    sigma = torch.tensor(sigma_val, dtype=dtype)
    var = utils.dm_rbf_variance(dm, sigma)
    expected = d * (sigma_val ** 2) / 2.0
    atol = 1e-10 if dtype == torch.float64 else 1e-5
    assert torch.allclose(var, torch.full_like(var, expected), atol=atol)


# ---------------------------------------------------------------------------
# gauss_entropy_lb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_gauss_entropy_lb_formula(dtype):
    d, sigma_val = 4, 0.7
    sigma = torch.tensor(sigma_val, dtype=dtype)
    result = utils.gauss_entropy_lb(d, sigma)
    expected = (d / 2.0) * (1.0 + math.log(2.0 * math.pi * sigma_val ** 2))
    atol = 1e-10 if dtype == torch.float64 else 1e-5
    assert abs(result.item() - expected) < atol


def test_gauss_entropy_lb_monotone():
    d = 4
    lb_small = utils.gauss_entropy_lb(d, torch.tensor(0.5, dtype=torch.float64))
    lb_large = utils.gauss_entropy_lb(d, torch.tensor(2.0, dtype=torch.float64))
    assert lb_large > lb_small


# ---------------------------------------------------------------------------
# cartesian_product
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_identity(dtype):
    a = _rand((4, 3), dtype)
    out = utils.cartesian_product([a])
    assert torch.equal(out, a)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_two(dtype):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    b = torch.tensor([[0.5, 0.5, 1.0], [1.0, 2.0, 3.0]], dtype=dtype)
    out = utils.cartesian_product([a, b])
    assert out.shape == (2, 6)
    # Each element [i, j*3+k] == a[i,j] * b[i,k]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                atol = 1e-12 if dtype == torch.float64 else 1e-6
                assert abs(out[i, j * 3 + k].item() - (a[i, j] * b[i, k]).item()) < atol


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_three_shape(dtype):
    a = _rand((3, 2), dtype)
    b = _rand((3, 3), dtype)
    c = _rand((3, 4), dtype)
    out = utils.cartesian_product([a, b, c])
    assert out.shape == (3, 24)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_probability_vectors(dtype):
    # cartesian product of probability vectors is a probability vector
    bs = 3
    for sizes in [(4,), (4, 5), (4, 5, 3)]:
        tensors = []
        for s in sizes:
            t = torch.abs(_rand((bs, s), dtype)) + 0.1
            t = t / t.sum(dim=-1, keepdim=True)
            tensors.append(t)
        out = utils.cartesian_product(tensors)
        atol = 1e-10 if dtype == torch.float64 else 1e-5
        assert torch.allclose(out.sum(dim=-1), torch.ones(bs, dtype=dtype), atol=atol)


# ---------------------------------------------------------------------------
# samples2dm
# ---------------------------------------------------------------------------

def _numpy_samples2dm(samples: np.ndarray) -> np.ndarray:
    nonzero = np.any(samples != 0, axis=-1).astype(samples.dtype)
    w = nonzero / nonzero.sum(axis=-1, keepdims=True)
    return np.concatenate([w[..., None], samples], axis=-1)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_samples2dm_nonzero_weights(dtype):
    # Construct samples where some rows are all zero.
    samples = torch.zeros(2, 4, 3, dtype=dtype)
    samples[0, 0] = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    samples[0, 2] = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    samples[1, 1] = torch.tensor([0.5, 0.5, 0.0], dtype=dtype)
    samples[1, 3] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)

    dm = utils.samples2dm(samples)
    w = dm[:, :, 0]
    # Zero rows get zero weight
    assert w[0, 1].item() == 0.0 and w[0, 3].item() == 0.0
    assert w[1, 0].item() == 0.0 and w[1, 2].item() == 0.0
    # Remaining weights are equal and positive
    assert abs(w[0, 0].item() - w[0, 2].item()) < 1e-6
    assert abs(w[1, 1].item() - w[1, 3].item()) < 1e-6
    # Weights sum to 1
    atol = 1e-10 if dtype == torch.float64 else 1e-6
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, dtype=dtype), atol=atol)
    # Vector slice is unchanged
    assert torch.equal(dm[:, :, 1:], samples)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_samples2dm_matches_numpy(dtype):
    # Validate against the corrected numpy formula.
    # The original Keras version had a bool/int dtype bug; the torch port fixes it.
    samples_np = RNG.standard_normal((4, 5, 3))
    # Force some rows to zero
    samples_np[0, 2] = 0.0
    samples_np[1, 4] = 0.0
    samples_np[2, 0] = 0.0
    expected = _numpy_samples2dm(samples_np)
    out = utils.samples2dm(_t(samples_np, dtype))
    tol = 1e-10 if dtype == torch.float64 else 1e-6
    np.testing.assert_allclose(
        out.detach().cpu().numpy(),
        expected.astype(out.numpy().dtype),
        rtol=tol, atol=tol,
    )
