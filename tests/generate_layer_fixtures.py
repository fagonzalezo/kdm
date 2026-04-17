"""
Generate Keras reference fixtures for layer equivalence tests.

Run once in an env that has keras + tensorflow installed (e.g. conda env `tf2`):

    conda run -n tf2 python tests/generate_layer_fixtures.py

Produces tests/fixtures/layer_references.npz. Fixed-seed inputs and all
trainable weights are materialized from numpy, so the torch test side can
load the same arrays and initialize equivalent layers.

Legacy layer sources are loaded dynamically from the `keras-legacy` branch
via `git show`, so no duplicate reference file is kept in the repo.
"""
from __future__ import annotations

import subprocess
import types
from pathlib import Path

import numpy as np

import keras

keras.config.set_floatx("float64")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
FIXTURES_PATH = FIXTURES_DIR / "layer_references.npz"


def load_legacy_module(rel_path: str, name: str) -> types.ModuleType:
    src = subprocess.check_output(
        ["git", "show", f"keras-legacy:{rel_path}"],
        cwd=REPO_ROOT,
    ).decode()
    mod = types.ModuleType(name)
    mod.__file__ = f"<keras-legacy:{rel_path}>"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def to_numpy(x):
    return keras.ops.convert_to_numpy(x)


def main() -> None:
    rbf_mod = load_legacy_module("kdm/layers/rbf_kernel_layer.py", "kref_rbf")
    cos_mod = load_legacy_module("kdm/layers/cosine_kernel_layer.py", "kref_cos")
    cp_mod = load_legacy_module("kdm/layers/crossproduct_kernel_layer.py", "kref_cp")
    kdm_mod = load_legacy_module("kdm/layers/kdm_layer.py", "kref_kdm")
    proj_mod = load_legacy_module("kdm/layers/kdm_proj_layer.py", "kref_proj")
    mem_mod = load_legacy_module("kdm/layers/mem_kdm_layer.py", "kref_mem")

    rng = np.random.default_rng(2024)

    bs, n_in, d, n_comp, dim_y = 4, 5, 3, 7, 6

    A_full = rng.standard_normal((bs, n_in, d)).astype(np.float64)
    B_full = rng.standard_normal((n_comp, d)).astype(np.float64)
    sigma = np.float64(0.7)

    c_x = rng.standard_normal((n_comp, d)).astype(np.float64) * 0.25
    c_y = rng.standard_normal((n_comp, dim_y)).astype(np.float64) * 0.3
    c_w = rng.uniform(0.1, 1.0, (n_comp,)).astype(np.float64)

    rho_in_w = rng.uniform(0.1, 1.0, (bs, n_in)).astype(np.float64)
    rho_in_w = rho_in_w / rho_in_w.sum(axis=1, keepdims=True)
    rho_in = np.concatenate([rho_in_w[..., None], A_full], axis=-1)

    mem_n = 8
    mem_sample = rng.standard_normal((bs, d)).astype(np.float64)
    mem_neighbors = rng.standard_normal((bs, mem_n, d)).astype(np.float64)
    mem_labels = rng.standard_normal((bs, mem_n, dim_y)).astype(np.float64)
    mem_sigma = np.float64(0.9)

    dim1 = 2
    sigma1 = np.float64(0.5)
    sigma2 = np.float64(1.1)

    proj_input = rng.standard_normal((bs, d)).astype(np.float64)

    fixtures: dict[str, np.ndarray] = {
        "input_A": A_full,
        "input_B": B_full,
        "input_sigma": sigma,
        "input_c_x": c_x,
        "input_c_y": c_y,
        "input_c_w": c_w,
        "input_rho_in": rho_in,
        "input_mem_sample": mem_sample,
        "input_mem_neighbors": mem_neighbors,
        "input_mem_labels": mem_labels,
        "input_mem_sigma": mem_sigma,
        "input_cp_dim1": np.int64(dim1),
        "input_cp_sigma1": sigma1,
        "input_cp_sigma2": sigma2,
        "input_proj_input": proj_input,
    }

    A_t = keras.ops.convert_to_tensor(A_full)
    B_t = keras.ops.convert_to_tensor(B_full)

    # --- RBFKernelLayer ---
    rbf = rbf_mod.RBFKernelLayer(sigma=float(sigma), dim=d)
    K_rbf = rbf(A_t, B_t)
    fixtures["out_rbf_K"] = to_numpy(K_rbf)
    fixtures["out_rbf_log_weight"] = to_numpy(rbf.log_weight())

    # --- CosineKernelLayer ---
    cos = cos_mod.CosineKernelLayer()
    K_cos = cos(A_t, B_t)
    fixtures["out_cos_K"] = to_numpy(K_cos)

    # --- CrossProductKernelLayer (RBF x RBF) ---
    rbf1 = rbf_mod.RBFKernelLayer(sigma=float(sigma1), dim=dim1)
    rbf2 = rbf_mod.RBFKernelLayer(sigma=float(sigma2), dim=d - dim1)
    cp = cp_mod.CrossProductKernelLayer(dim1=dim1, kernel1=rbf1, kernel2=rbf2)
    K_cp = cp(A_t, B_t)
    fixtures["out_cp_K"] = to_numpy(K_cp)
    fixtures["out_cp_log_weight"] = to_numpy(cp.log_weight())

    # --- KDMProjLayer ---
    rbf_for_proj = rbf_mod.RBFKernelLayer(sigma=float(sigma), dim=d)
    proj = proj_mod.KDMProjLayer(kernel=rbf_for_proj, dim_x=d, n_comp=n_comp)
    proj.c_x.assign(keras.ops.convert_to_tensor(c_x))
    proj.c_w.assign(keras.ops.convert_to_tensor(c_w))
    proj_out = proj(keras.ops.convert_to_tensor(proj_input))
    fixtures["out_proj"] = to_numpy(proj_out)

    # --- KDMLayer.forward ---
    # NOTE: legacy KDMLayer mutates c_w via .assign() in call() (normalizes it).
    # That means any second call would act on the already-normalized c_w. We
    # compute only the first-call output, and also the "log_marginal-equivalent"
    # quantity by setting generative=1 on a separate instance and capturing
    # the added loss (or by recomputing manually after a fresh instance).
    rbf_for_kdm = rbf_mod.RBFKernelLayer(sigma=float(sigma), dim=d)
    kdm = kdm_mod.KDMLayer(kernel=rbf_for_kdm, dim_x=d, dim_y=dim_y, n_comp=n_comp)
    kdm.c_x.assign(keras.ops.convert_to_tensor(c_x))
    kdm.c_y.assign(keras.ops.convert_to_tensor(c_y))
    kdm.c_w.assign(keras.ops.convert_to_tensor(c_w))
    rho_out = kdm(keras.ops.convert_to_tensor(rho_in))
    fixtures["out_kdm_rho"] = to_numpy(rho_out)

    # log_marginal: recompute from scratch against an unmutated c_w.
    # We mirror the legacy formula exactly using freshly assigned weights.
    rbf_for_lm = rbf_mod.RBFKernelLayer(sigma=float(sigma), dim=d)
    kdm_lm = kdm_mod.KDMLayer(
        kernel=rbf_for_lm, dim_x=d, dim_y=dim_y, n_comp=n_comp, generative=1.0
    )
    kdm_lm.c_x.assign(keras.ops.convert_to_tensor(c_x))
    kdm_lm.c_y.assign(keras.ops.convert_to_tensor(c_y))
    kdm_lm.c_w.assign(keras.ops.convert_to_tensor(c_w))

    # Replicate the legacy per-sample log-prob (before the mean/-scale in add_loss).
    # Legacy does: c_w.assign(normalize(abs(c_w))); then out_w = c_w * out_vw**2;
    # proj = einsum(in_w, out_w); log_probs = log(proj + eps) + kernel.log_weight()
    eps = 1e-12
    comp_w_np = np.abs(c_w)
    comp_w_np = comp_w_np / comp_w_np.sum()
    in_w_np = rho_in[:, :, 0]
    in_v_np = rho_in[:, :, 1:]
    rbf_for_lm_eval = rbf_mod.RBFKernelLayer(sigma=float(sigma), dim=d)
    out_vw_np = to_numpy(
        rbf_for_lm_eval(
            keras.ops.convert_to_tensor(in_v_np),
            keras.ops.convert_to_tensor(c_x),
        )
    )
    out_w_np = comp_w_np[None, None, :] * out_vw_np ** 2
    proj_np = np.einsum("...i,...ij->...", in_w_np, out_w_np)
    log_w = to_numpy(rbf_for_lm_eval.log_weight())
    fixtures["out_kdm_log_marginal"] = np.log(proj_np + eps) + log_w

    # --- MemKDMLayer ---
    mem_rbf = rbf_mod.MemRBFKernelLayer(sigma=float(mem_sigma), dim=d)
    mem_layer = mem_mod.MemKDMLayer(
        kernel=mem_rbf, dim_x=d, dim_y=dim_y, n_comp=mem_n
    )
    mem_out = mem_layer(
        [
            keras.ops.convert_to_tensor(mem_sample),
            keras.ops.convert_to_tensor(mem_neighbors),
            keras.ops.convert_to_tensor(mem_labels),
        ]
    )
    fixtures["out_mem_rho"] = to_numpy(mem_out)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES_PATH, **fixtures)
    print(f"Wrote {len(fixtures)} arrays to {FIXTURES_PATH.relative_to(REPO_ROOT)}")
    for k in sorted(fixtures):
        arr = fixtures[k]
        print(f"  {k:30s} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
