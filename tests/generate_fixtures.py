"""
Generate Keras reference fixtures for utils equivalence tests.

Run once in an env that has keras + tensorflow installed (e.g. conda env `tf2`):

    conda run -n tf2 python tests/generate_fixtures.py

Produces tests/fixtures/utils_references.npz with fixed-seed inputs and the
corresponding Keras outputs. Fixtures are committed to the repo so the torch
test suite can run without keras installed.

The Keras version of kdm/utils.py is loaded dynamically from the
`keras-legacy` branch via `git show`, so no duplicate reference file is kept
in the repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

import keras

keras.config.set_floatx("float64")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
FIXTURES_PATH = FIXTURES_DIR / "utils_references.npz"


def load_keras_utils_from_legacy() -> types.ModuleType:
    """Dynamically load kdm/utils.py from the keras-legacy branch."""
    src = subprocess.check_output(
        ["git", "show", "keras-legacy:kdm/utils.py"],
        cwd=REPO_ROOT,
    ).decode()
    mod = types.ModuleType("kdm_utils_legacy")
    mod.__file__ = "<keras-legacy:kdm/utils.py>"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def to_numpy(x):
    return keras.ops.convert_to_numpy(x)


def main() -> None:
    kref = load_keras_utils_from_legacy()

    rng = np.random.default_rng(42)

    bs, n, d = 4, 5, 3

    dm = rng.standard_normal((bs, n, d + 1)).astype(np.float64)
    w = rng.uniform(0.0, 1.0, (bs, n)).astype(np.float64)
    v = rng.standard_normal((bs, n, d)).astype(np.float64)
    psi = rng.standard_normal((bs, d)).astype(np.float64)
    x_query = rng.standard_normal((bs, d)).astype(np.float64)
    sigma = np.float64(0.7)

    samples = rng.standard_normal((bs, n, d)).astype(np.float64)
    samples[0, 2] = 0.0
    samples[1, 4] = 0.0
    samples[2, 0] = 0.0

    cart_a = rng.uniform(0.0, 1.0, (bs, 2)).astype(np.float64)
    cart_b = rng.uniform(0.0, 1.0, (bs, 3)).astype(np.float64)
    cart_c = rng.uniform(0.0, 1.0, (bs, 4)).astype(np.float64)

    fixtures: dict[str, np.ndarray] = {
        "input_dm": dm,
        "input_w": w,
        "input_v": v,
        "input_psi": psi,
        "input_x_query": x_query,
        "input_sigma": sigma,
        "input_samples_with_zeros": samples,
        "input_cart_a": cart_a,
        "input_cart_b": cart_b,
        "input_cart_c": cart_c,
    }

    out_w, out_v = kref.dm2comp(dm)
    fixtures["out_dm2comp_w"] = to_numpy(out_w)
    fixtures["out_dm2comp_v"] = to_numpy(out_v)

    fixtures["out_comp2dm"] = to_numpy(kref.comp2dm(w, v))
    # NOTE: kref.samples2dm is skipped. The Keras version has a latent
    # bool/int dtype bug on the TF backend (see `ops.any` returning bool,
    # then dividing by a sum-of-bools). It is never called from the rest
    # of the codebase. The torch port fixes the bug by casting to float
    # before the division; it is tested against an inline numpy reference
    # in tests/test_utils.py.
    fixtures["out_pure2dm"] = to_numpy(kref.pure2dm(psi))
    fixtures["out_dm2discrete"] = to_numpy(kref.dm2discrete(dm))

    fixtures["out_dm_rbf_loglik"] = to_numpy(
        kref.dm_rbf_loglik(x_query, dm, sigma)
    )
    fixtures["out_dm_rbf_expectation"] = to_numpy(kref.dm_rbf_expectation(dm))
    fixtures["out_dm_rbf_variance"] = to_numpy(kref.dm_rbf_variance(dm, sigma))

    fixtures["out_gauss_entropy_lb"] = to_numpy(kref.gauss_entropy_lb(d, sigma))

    fixtures["out_cartesian_product_1"] = to_numpy(kref.cartesian_product([cart_a]))
    fixtures["out_cartesian_product_2"] = to_numpy(
        kref.cartesian_product([cart_a, cart_b])
    )
    fixtures["out_cartesian_product_3"] = to_numpy(
        kref.cartesian_product([cart_a, cart_b, cart_c])
    )

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES_PATH, **fixtures)
    print(f"Wrote {len(fixtures)} arrays to {FIXTURES_PATH.relative_to(REPO_ROOT)}")
    for k in sorted(fixtures):
        arr = fixtures[k]
        print(f"  {k:35s} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
