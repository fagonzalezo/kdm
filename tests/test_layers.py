"""
Equivalence tests for the ported layers in kdm/layers/ (PyTorch).

Reference outputs were produced from the Keras 3 version on the keras-legacy
branch; see tests/generate_layer_fixtures.py. Each test materializes the
trainable weights from the same numpy arrays used by the fixture generator,
so any discrepancy reflects a genuine math difference (not init drift).

The softplus reparameterization on RBFKernelLayer means direct assignment to
`kernel.sigma = value` internally inverts softplus; the sigma property still
returns the requested value to within float round-off, so numerical
equivalence is preserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kdm.layers import (
    CosineKernelLayer,
    CrossProductKernelLayer,
    KDMLayer,
    KDMProjLayer,
    MemKDMLayer,
    MemRBFKernelLayer,
    RBFKernelLayer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "layer_references.npz"


@pytest.fixture(scope="module")
def refs() -> dict[str, np.ndarray]:
    if not FIXTURES_PATH.exists():
        pytest.fail(
            f"Missing fixture file {FIXTURES_PATH}. Regenerate with:\n"
            f"    conda run -n tf2 python tests/generate_layer_fixtures.py"
        )
    with np.load(FIXTURES_PATH) as f:
        return {k: f[k] for k in f.files}


def _t(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=dtype)


def _assert_close(actual: torch.Tensor, expected: np.ndarray, *, dtype: torch.dtype) -> None:
    tol = {torch.float64: 1e-10, torch.float32: 1e-6}[dtype]
    act = actual.detach().cpu().numpy()
    exp = np.asarray(expected, dtype=act.dtype)
    np.testing.assert_allclose(act, exp, rtol=tol, atol=tol)


def _cast_layer(layer: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    # Capture sigmas before casting: raw_sigma was computed via softplus_inv at
    # the layer's original (float32) precision; simply casting to float64 would
    # leave a float32-precision value in a float64 tensor. Re-assigning sigma
    # after cast re-runs softplus_inv at the new dtype.
    rbf_sigmas = [
        (m, float(m.sigma.detach()))
        for m in layer.modules()
        if isinstance(m, RBFKernelLayer)
    ]
    layer = layer.double() if dtype == torch.float64 else layer.float()
    for m, s in rbf_sigmas:
        m.sigma = s
    return layer


DTYPES = [torch.float64, torch.float32]
DTYPE_IDS = ["float64", "float32"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rbf_kernel(refs, dtype):
    sigma = float(refs["input_sigma"])
    d = int(refs["input_A"].shape[-1])
    layer = _cast_layer(RBFKernelLayer(sigma=sigma, dim=d), dtype)
    K = layer(_t(refs["input_A"], dtype), _t(refs["input_B"], dtype))
    _assert_close(K, refs["out_rbf_K"], dtype=dtype)
    _assert_close(layer.log_weight(), refs["out_rbf_log_weight"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_cosine_kernel(refs, dtype):
    layer = _cast_layer(CosineKernelLayer(), dtype)
    K = layer(_t(refs["input_A"], dtype), _t(refs["input_B"], dtype))
    _assert_close(K, refs["out_cos_K"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_crossproduct_kernel(refs, dtype):
    dim1 = int(refs["input_cp_dim1"])
    d = int(refs["input_A"].shape[-1])
    sigma1 = float(refs["input_cp_sigma1"])
    sigma2 = float(refs["input_cp_sigma2"])
    k1 = RBFKernelLayer(sigma=sigma1, dim=dim1)
    k2 = RBFKernelLayer(sigma=sigma2, dim=d - dim1)
    layer = _cast_layer(
        CrossProductKernelLayer(dim1=dim1, kernel1=k1, kernel2=k2), dtype
    )
    K = layer(_t(refs["input_A"], dtype), _t(refs["input_B"], dtype))
    _assert_close(K, refs["out_cp_K"], dtype=dtype)
    _assert_close(layer.log_weight(), refs["out_cp_log_weight"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_proj_layer(refs, dtype):
    d = int(refs["input_proj_input"].shape[-1])
    n_comp = int(refs["input_c_x"].shape[0])
    sigma = float(refs["input_sigma"])
    kernel = RBFKernelLayer(sigma=sigma, dim=d)
    layer = _cast_layer(
        KDMProjLayer(kernel=kernel, dim_x=d, n_comp=n_comp), dtype
    )
    with torch.no_grad():
        layer.c_x.copy_(_t(refs["input_c_x"], dtype))
        layer.c_w.copy_(_t(refs["input_c_w"], dtype))
    out = layer(_t(refs["input_proj_input"], dtype))
    _assert_close(out, refs["out_proj"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_forward(refs, dtype):
    d = int(refs["input_c_x"].shape[-1])
    dim_y = int(refs["input_c_y"].shape[-1])
    n_comp = int(refs["input_c_x"].shape[0])
    sigma = float(refs["input_sigma"])
    kernel = RBFKernelLayer(sigma=sigma, dim=d)
    layer = _cast_layer(
        KDMLayer(kernel=kernel, dim_x=d, dim_y=dim_y, n_comp=n_comp), dtype
    )
    with torch.no_grad():
        layer.c_x.copy_(_t(refs["input_c_x"], dtype))
        layer.c_y.copy_(_t(refs["input_c_y"], dtype))
        layer.c_w.copy_(_t(refs["input_c_w"], dtype))
    rho_out = layer(_t(refs["input_rho_in"], dtype))
    _assert_close(rho_out, refs["out_kdm_rho"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_kdm_layer_log_marginal(refs, dtype):
    d = int(refs["input_c_x"].shape[-1])
    dim_y = int(refs["input_c_y"].shape[-1])
    n_comp = int(refs["input_c_x"].shape[0])
    sigma = float(refs["input_sigma"])
    kernel = RBFKernelLayer(sigma=sigma, dim=d)
    layer = _cast_layer(
        KDMLayer(kernel=kernel, dim_x=d, dim_y=dim_y, n_comp=n_comp), dtype
    )
    with torch.no_grad():
        layer.c_x.copy_(_t(refs["input_c_x"], dtype))
        layer.c_y.copy_(_t(refs["input_c_y"], dtype))
        layer.c_w.copy_(_t(refs["input_c_w"], dtype))
    log_probs = layer.log_marginal(_t(refs["input_rho_in"], dtype))
    _assert_close(log_probs, refs["out_kdm_log_marginal"], dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mem_kdm_layer(refs, dtype):
    d = int(refs["input_mem_sample"].shape[-1])
    dim_y = int(refs["input_mem_labels"].shape[-1])
    mem_n = int(refs["input_mem_neighbors"].shape[1])
    mem_sigma = float(refs["input_mem_sigma"])
    kernel = MemRBFKernelLayer(sigma=mem_sigma, dim=d)
    layer = _cast_layer(
        MemKDMLayer(kernel=kernel, dim_x=d, dim_y=dim_y, n_comp=mem_n), dtype
    )
    out = layer(
        (
            _t(refs["input_mem_sample"], dtype),
            _t(refs["input_mem_neighbors"], dtype),
            _t(refs["input_mem_labels"], dtype),
        )
    )
    _assert_close(out, refs["out_mem_rho"], dtype=dtype)
