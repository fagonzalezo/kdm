---
name: kdm
description: >
  Guide users in applying the kdm-torch library (Kernel Density Matrices) to
  machine learning tasks. Make sure to use this skill whenever the user asks
  how to build a KDM model, use kdm-torch, or apply KDMs to classification,
  regression, density estimation, uncertainty quantification, or generative
  modeling. Trigger also when the user mentions kernel density matrices,
  KDMClassModel, KDMRegressModel, KDMDenEstModel, KDMLayer, init_kdm_layer,
  log_marginal, input log-likelihood, or OOD detection with KDMs.
---

# KDM skill

You are helping the user apply the `kdm-torch` library. Follow the workflow
below and use the task-specific snapshots to produce correct, runnable code.
For a **full runnable template** for any task, read the corresponding file in
`references/`.

---

## Universal 4-step workflow

Every KDM task follows this sequence — never skip step 3.

1. **Build encoder** — `nn.Sequential` / custom `nn.Module`. For raw vectors,
   `nn.Identity()` works.
2. **Instantiate model** — choose the right model class for the task (table
   below).
3. **Initialize from data** — call `init_kdm_layer` or `init_kdm_proj_layer`
   with a random subset of training encodings. **This is critical: random
   initialization does not converge.**
4. **Write an explicit training loop** — there is no `.fit()`. Compose the
   loss yourself from the primitives listed under each task.

### Model class quick-reference

| Task | Model class | Init helper |
|---|---|---|
| Classification | `KDMClassModel` | `init_kdm_layer(model.kdm, enc, y_onehot, init_sigma=True)` |
| Regression | `KDMRegressModel` | `init_kdm_layer(model.kdm, enc, y, init_sigma=True)` |
| Density estimation | `KDMDenEstModel` | `init_kdm_proj_layer(model.kdmproj, x_sub, init_sigma=True)` |
| Uncertainty (OOD) | `KDMClassModel` | same as classification |
| Generative co-training | `KDMClassModel` | same as classification |

---

## Task snapshots

### Classification

```python
import torch, torch.nn as nn, torch.nn.functional as F
from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer

encoder = nn.Linear(input_dim, 32)
model = KDMClassModel(encoded_size=32, dim_y=n_classes,
                      encoder=encoder, n_comp=200, sigma=0.5)

# Init — use a random subset of size n_comp
idx = torch.randperm(len(X_train))[:200]
model.eval()
with torch.no_grad():
    enc_sub = encoder(X_train[idx])
init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot[idx], init_sigma=True)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for xb, yb in dataloader:
    probs = model(xb)                                          # (bs, n_classes)
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yb)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

→ Full template: `references/classification.md`

---

### Regression

```python
from kdm.models import KDMRegressModel
from kdm.init import init_kdm_layer
from kdm.utils import dm_rbf_loglik

encoder = nn.Linear(input_dim, 32)
model = KDMRegressModel(encoded_size=32, dim_y=1, encoder=encoder, n_comp=200)

idx = torch.randperm(len(X_train))[:200]
with torch.no_grad():
    enc_sub = encoder(X_train[idx])
init_kdm_layer(model.kdm, enc_sub.detach(), y_train[idx], init_sigma=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for xb, yb in dataloader:
    rho_y = model(xb)
    loss = -dm_rbf_loglik(yb, rho_y, model.sigma_y).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

mean, variance = model.predict_reg(X_test)   # uncertainty-aware predictions
```

→ Full template: `references/regression.md`

---

### Density estimation

```python
from kdm.models import KDMDenEstModel
from kdm.init import init_kdm_proj_layer

model = KDMDenEstModel(dim_x=input_dim, sigma=0.5, n_comp=500)

idx = torch.randperm(len(X_train))[:500]
init_kdm_proj_layer(model.kdmproj, X_train[idx], init_sigma=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for xb in dataloader:
    loss = -model(xb).mean()                 # model(x) returns log-density
    optimizer.zero_grad(); loss.backward(); optimizer.step()

distrib = model.get_distrib()                # torch.distributions.MixtureSameFamily
log_p  = distrib.log_prob(X_test)
```

→ Full template: `references/density_estimation.md`

---

### Uncertainty quantification

KDMs naturally assign high entropy to out-of-distribution inputs because the
kernel weights decay as inputs move away from prototype support points. No
temperature scaling or post-hoc calibration is needed.

Two complementary OOD scores are available:

**Entropy** — label-level uncertainty ("is the model unsure which class?"):
```python
from kdm.utils import pure2dm

model.eval()
with torch.no_grad():
    probs = model(x)                                          # (bs, n_classes)

entropy    = -(probs * torch.log(probs.clamp_min(1e-7))).sum(dim=-1)
confidence = probs.max(dim=-1).values
# High entropy / low confidence → uncertain or OOD input
```

**Input log-likelihood** — input-level novelty ("has the model seen inputs like this?"):
```python
with torch.no_grad():
    rho_x   = pure2dm(model.encoder(x))
    log_p_x = model.kdm.log_marginal(rho_x)   # (bs,) — higher = more in-distribution
# Use -log_p_x as OOD score; catches overconfident-yet-OOD cases that entropy misses
```

These can diverge: a rotated image may still be confidently classified (low
entropy) while being far from any training prototype (low log P(x)). Use both
for robust OOD detection.

→ Full template (calibration plots, OOD benchmarks, AUROC): `references/uncertainty.md`

---

### Generative co-training

Train discriminatively first, then fine-tune by adding a generative term that
maximises the marginal log-likelihood of inputs under the joint KDM.

```python
from kdm.utils import pure2dm

# Phase 1 — discriminative (standard classification loop above)
# Phase 2 — generative co-training
model.encoder.requires_grad_(False)          # freeze encoder
opt_gen = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)
alpha = 0.1
for xb, yb in dataloader:
    probs   = model(xb)
    disc    = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yb)
    rho_x   = pure2dm(model.encoder(xb))
    gen     = -model.kdm.log_marginal(rho_x).mean()
    loss    = disc + alpha * gen
    opt_gen.zero_grad(); loss.backward(); opt_gen.step()
```

→ Full template (conditional generation, image synthesis): `references/generative.md`

---

## Loss toolkit

| Symbol | Where it comes from | When to use |
|---|---|---|
| `F.nll_loss(log(probs), y)` | `torch.nn.functional` | Classification |
| `dm_rbf_loglik(y, rho_y, sigma_y)` | `kdm.utils` | Regression NLL |
| `-model(x).mean()` | `KDMDenEstModel.forward` | Density estimation |
| `-model.kdm.log_marginal(rho_x).mean()` | `KDMLayer.log_marginal` | Generative term |
| `l1_norm(model.kdm.c_w)` | `kdm.losses` | Optional weight regularization |

---

## Common pitfalls

- **Skipping init** — `init_kdm_layer` / `init_kdm_proj_layer` must be called
  before the first optimizer step. Omitting it gives NaN or near-zero gradients.
- **Epsilon in log** — always use `.clamp_min(1e-7)` or `+ 1e-7` before `log`
  to match the library's numerical convention.
- **Setting sigma directly** — `layer.sigma = v` works correctly (it inverts
  softplus internally). Do not write to `raw_sigma` by hand.
- **n_comp must equal subset size** — the number of components passed to the
  model constructor must match the number of rows in the data passed to the
  init helper.

---

## Building custom models with KDMLayer directly

When the built-in model classes don't fit the task, use `KDMLayer` and
`KDMProjLayer` directly. See `references/kdm_layer.md` for:

- KDM tensor format `(bs, n_comp, dim+1)` and the `pure2dm` / `dm2comp` /
  `comp2dm` utilities
- `KDMLayer` constructor arguments, trainable parameters, `forward`, and
  `log_marginal` signatures
- `KDMProjLayer` for density estimation
- All kernel options: `RBFKernelLayer`, `CosineKernelLayer`,
  `CrossProductKernelLayer`, `CompTransKernelLayer`
- Initialization with `init_kdm_layer` / `init_kdm_proj_layer`
- A complete custom two-headed model example
