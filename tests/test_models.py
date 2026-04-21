"""
Smoke tests for kdm/models/ — instantiate, forward pass, shape and validity checks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kdm.models import KDMClassModel


def _simple_encoder(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(in_dim, out_dim)


def test_kdm_class_model_forward():
    in_dim, encoded_size, n_classes, n_comp = 8, 4, 3, 10
    encoder = _simple_encoder(in_dim, encoded_size)
    model = KDMClassModel(
        encoded_size=encoded_size,
        dim_y=n_classes,
        encoder=encoder,
        n_comp=n_comp,
        sigma=0.5,
    )
    x = torch.randn(5, in_dim)
    probs = model(x)
    assert probs.shape == (5, n_classes)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_kdm_class_model_gradient_flows():
    in_dim, encoded_size, n_classes, n_comp = 8, 4, 3, 10
    encoder = _simple_encoder(in_dim, encoded_size)
    model = KDMClassModel(
        encoded_size=encoded_size,
        dim_y=n_classes,
        encoder=encoder,
        n_comp=n_comp,
        sigma=0.5,
    )
    x = torch.randn(5, in_dim)
    probs = model(x)
    loss = -torch.log(probs + 1e-7).mean()
    loss.backward()
    # Both encoder and KDM layer should receive gradients.
    enc_param = next(encoder.parameters())
    assert enc_param.grad is not None
    assert torch.isfinite(enc_param.grad).all()
    assert model.kdm.c_x.grad is not None
    assert torch.isfinite(model.kdm.c_x.grad).all()
