import math
from typing import Iterator, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from ...utils import dm2comp, pure2dm
from ..mem import MemKDMClassModel


class MemKDMClassModelWrapper:
    """Wraps a `MemKDMClassModel` with an encoder, a faiss index over the
    training set, and batching helpers.

    Training is notebook-owned: the notebook constructs an optimizer over
    `wrapper.model.parameters()` and iterates batches from `iter_batches`.
    """

    def __init__(
        self,
        encoded_size: int,
        dim_y: int,
        samples_x,
        samples_y,
        encoder: torch.nn.Module,
        n_comp: int,
        index_type: str = "Flat",
        sigma: float = 0.1,
    ):
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.samples_y = np.asarray(samples_y)
        self.samples_x_enc = self._encode_all(samples_x).astype(np.float32)
        self.index = faiss.index_factory(encoded_size, index_type)
        self.index.train(self.samples_x_enc)
        self.index.add(self.samples_x_enc)
        self.model = MemKDMClassModel(
            encoded_size=encoded_size,
            dim_y=dim_y,
            n_comp=n_comp,
            sigma=sigma,
        )

    # ------------------------------------------------------------------ utils

    @torch.no_grad()
    def _encode_all(self, x, batch_size: int = 32) -> np.ndarray:
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            x_t = torch.as_tensor(x)
            out = np.zeros((x_t.shape[0], self.encoded_size), dtype=np.float32)
            for i in range(0, x_t.shape[0], batch_size):
                batch = x_t[i : i + batch_size]
                enc = self.encoder(batch).detach().cpu().numpy()
                out[i : i + enc.shape[0]] = enc
        finally:
            self.encoder.train(was_training)
        return out

    # ------------------------------------------------------------------ sigma

    def init_sigma(self, mult: float = 0.1, n_samples: int = 100) -> float:
        """Initialize sigma from the median inter-sample distance among a
        random subset of the training encodings (1st-nearest-neighbor
        distance, matching the legacy heuristic).
        """
        n_samples = min(n_samples, self.samples_x_enc.shape[0])
        rng = np.random.default_rng()
        samples = rng.choice(self.samples_x_enc, n_samples, replace=False)
        _, I = self.index.search(samples, self.n_comp + 1)
        neigh = np.take(self.samples_x_enc, I, axis=0)
        dists = np.linalg.norm(samples[:, None, :] - neigh, axis=-1)
        sigma = float(np.median(dists[:, 1:]) * mult)
        self.model.kernel.sigma = sigma
        return sigma

    # ------------------------------------------------------------------ predict

    @torch.no_grad()
    def predict(self, X, batch_size: int = 32) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        try:
            X_enc = self._encode_all(X, batch_size=batch_size)
            preds = []
            for i in range(0, X_enc.shape[0], batch_size):
                x_enc_np = X_enc[i : i + batch_size]
                _, I = self.index.search(x_enc_np, self.n_comp)
                x_neigh_np = np.take(self.samples_x_enc, I, axis=0)
                y_neigh_np = np.take(self.samples_y, I, axis=0)
                probs = self.model(
                    (
                        torch.as_tensor(x_enc_np),
                        torch.as_tensor(x_neigh_np),
                        torch.as_tensor(y_neigh_np),
                    )
                )
                preds.append(probs.detach().cpu().numpy())
        finally:
            self.model.train(was_training)
        return np.concatenate(preds, axis=0)

    @torch.no_grad()
    def predict_explain(self, x, n_neighbors: int):
        """Return the top-`n_neighbors` contributing memory indices and their
        posterior weights for a single input sample.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            x_enc_np = self._encode_all(x[None, ...])
            _, I = self.index.search(x_enc_np, self.n_comp)
            x_neigh_np = np.take(self.samples_x_enc, I, axis=0)
            y_neigh_np = np.take(self.samples_y, I, axis=0)
            y_neigh_ohe = F.one_hot(
                torch.as_tensor(y_neigh_np).long(), num_classes=self.dim_y
            ).to(torch.as_tensor(x_enc_np).dtype)
            rho_x = pure2dm(torch.as_tensor(x_enc_np))
            rho_y = self.model.mkdm(
                (
                    rho_x,
                    torch.as_tensor(x_neigh_np),
                    y_neigh_ohe,
                )
            )
            w, _ = dm2comp(rho_y)
            w = w.detach().cpu().numpy()
        finally:
            self.model.train(was_training)
        idx = np.argsort(w, axis=1)[:, ::-1][:, :n_neighbors]
        return np.take_along_axis(I, idx, axis=1), np.take_along_axis(w, idx, axis=1)

    # ------------------------------------------------------------------ batching

    def iter_batches(
        self,
        X=None,
        y=None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Iterator:
        """Yield `((x_enc, x_neigh, y_neigh), y_true)` torch-tensor batches.

        When `X` is None, iterates over the index's own samples (the query's
        own entry is dropped from its neighbor set by taking the last
        `n_comp` of `n_comp + 1` returned neighbors).
        """
        use_index_samples = X is None
        if use_index_samples:
            n = self.samples_x_enc.shape[0]
            y_all = self.samples_y
        else:
            assert y is not None and len(X) == len(y), "X and y must match length"
            n = len(X)
            y_all = np.asarray(y)

        order = np.random.permutation(n) if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            if use_index_samples:
                x_enc_np = self.samples_x_enc[idx]
                search_k = self.n_comp + 1
            else:
                x_enc_np = self._encode_all(X[idx], batch_size=batch_size)
                search_k = self.n_comp
            _, I = self.index.search(x_enc_np, search_k)
            x_neigh_np = np.take(self.samples_x_enc, I, axis=0)[:, -self.n_comp :]
            y_neigh_np = np.take(self.samples_y, I, axis=0)[:, -self.n_comp :]
            yield (
                (
                    torch.as_tensor(x_enc_np),
                    torch.as_tensor(x_neigh_np),
                    torch.as_tensor(y_neigh_np),
                ),
                torch.as_tensor(y_all[idx]),
            )

    def n_batches(self, X=None, batch_size: int = 32) -> int:
        n = self.samples_x_enc.shape[0] if X is None else len(X)
        return math.ceil(n / batch_size)
