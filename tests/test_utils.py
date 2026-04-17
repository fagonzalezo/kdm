"""
Equivalence tests for kdm/utils.py (PyTorch port).

Reference outputs were produced from the Keras 3 version on the keras-legacy
branch; see tests/generate_fixtures.py. Fixtures are stored as float64 to
sidestep latent dtype bugs in the original Keras utilities (they do not
affect notebook users, who always ran with float32 tensors that happened to
avoid the bool/int / float64 mixing points).

The PyTorch port is tested here in float64 for tight numerical agreement
(tol=1e-10) and separately in float32 (tol=1e-6) to guard against any
cross-dtype regressions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kdm"))

import utils  # noqa: E402  (the PyTorch kdm/utils.py, loaded bare)

FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "utils_references.npz"


@pytest.fixture(scope="module")
def refs() -> dict[str, np.ndarray]:
    if not FIXTURES_PATH.exists():
        pytest.fail(
            f"Missing fixture file {FIXTURES_PATH}. Regenerate with:\n"
            f"    conda run -n tf2 python tests/generate_fixtures.py"
        )
    with np.load(FIXTURES_PATH) as f:
        return {k: f[k] for k in f.files}


def _t(arr: np.ndarray, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype)


def _assert_close(torch_out, keras_ref, *, dtype: torch.dtype) -> None:
    tol = {torch.float64: 1e-10, torch.float32: 1e-6}[dtype]
    actual = torch_out.detach().cpu().numpy()
    expected = np.asarray(keras_ref, dtype=actual.dtype)
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["float64", "float32"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm2comp(refs, dtype):
    dm = _t(refs["input_dm"], dtype)
    w, v = utils.dm2comp(dm)
    _assert_close(w, refs["out_dm2comp_w"], dtype=dtype)
    _assert_close(v, refs["out_dm2comp_v"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_comp2dm(refs, dtype):
    out = utils.comp2dm(_t(refs["input_w"], dtype), _t(refs["input_v"], dtype))
    _assert_close(out, refs["out_comp2dm"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_pure2dm(refs, dtype):
    out = utils.pure2dm(_t(refs["input_psi"], dtype))
    _assert_close(out, refs["out_pure2dm"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm2discrete(refs, dtype):
    out = utils.dm2discrete(_t(refs["input_dm"], dtype))
    _assert_close(out, refs["out_dm2discrete"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_loglik(refs, dtype):
    out = utils.dm_rbf_loglik(
        _t(refs["input_x_query"], dtype),
        _t(refs["input_dm"], dtype),
        _t(refs["input_sigma"], dtype),
    )
    _assert_close(out, refs["out_dm_rbf_loglik"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_expectation(refs, dtype):
    out = utils.dm_rbf_expectation(_t(refs["input_dm"], dtype))
    _assert_close(out, refs["out_dm_rbf_expectation"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_dm_rbf_variance(refs, dtype):
    out = utils.dm_rbf_variance(
        _t(refs["input_dm"], dtype),
        _t(refs["input_sigma"], dtype),
    )
    _assert_close(out, refs["out_dm_rbf_variance"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_gauss_entropy_lb(refs, dtype):
    # d is a Python int in the reference; pass the same.
    d = int(refs["input_v"].shape[-1])
    out = utils.gauss_entropy_lb(d, _t(refs["input_sigma"], dtype))
    _assert_close(out, refs["out_gauss_entropy_lb"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_1(refs, dtype):
    out = utils.cartesian_product([_t(refs["input_cart_a"], dtype)])
    _assert_close(out, refs["out_cartesian_product_1"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_2(refs, dtype):
    out = utils.cartesian_product(
        [_t(refs["input_cart_a"], dtype), _t(refs["input_cart_b"], dtype)]
    )
    _assert_close(out, refs["out_cartesian_product_2"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cartesian_product_3(refs, dtype):
    out = utils.cartesian_product(
        [
            _t(refs["input_cart_a"], dtype),
            _t(refs["input_cart_b"], dtype),
            _t(refs["input_cart_c"], dtype),
        ]
    )
    _assert_close(out, refs["out_cartesian_product_3"], dtype=dtype)


# ---------------------------------------------------------------------------
# samples2dm: the Keras version has a bool/int dtype bug that made it unusable
# on TF backend (never called from any notebook or model). The torch port
# fixes it; verified here against a pure-numpy reference.
# ---------------------------------------------------------------------------


def _numpy_samples2dm(samples: np.ndarray) -> np.ndarray:
    nonzero = np.any(samples != 0, axis=-1).astype(samples.dtype)
    w = nonzero / nonzero.sum(axis=-1, keepdims=True)
    return np.concatenate([w[..., None], samples], axis=-1)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_samples2dm_against_numpy(refs, dtype):
    samples_np = refs["input_samples_with_zeros"]
    expected = _numpy_samples2dm(samples_np)
    out = utils.samples2dm(_t(samples_np, dtype))
    _assert_close(out, expected, dtype=dtype)
