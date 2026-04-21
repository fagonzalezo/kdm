# KDMLayer — Custom Model Guide

Reference for building custom models directly on top of `KDMLayer` and
`KDMProjLayer`, including tensor formats, kernel options, and initialization.

---

## KDM tensor format

All KDM tensors use a packed layout: the first slice along the last axis is
the component weight, and the remaining slices are the component vector.

```
dm:  (batch_size, n_comp, dim + 1)
       └── [:, :, 0]   — component weights
       └── [:, :, 1:]  — component vectors  (the support points)
```

**Utility functions** in `kdm.utils` for working with this format:

```python
from kdm.utils import pure2dm, dm2comp, comp2dm, dm2discrete

# Wrap a batch of encoder outputs as pure-state KDMs (n_comp=1, weight=1)
rho_x = pure2dm(encoded)          # (bs, d) → (bs, 1, d+1)

# Decompose a KDM into weights and vectors
w, v = dm2comp(dm)                 # w: (bs, n), v: (bs, n, d)

# Reconstruct a KDM from weights and vectors
dm = comp2dm(w, v)                 # (bs, n) + (bs, n, d) → (bs, n, d+1)

# Convert output KDM to class probabilities
probs = dm2discrete(rho_y)         # (bs, n, d+1) → (bs, d)
```

---

## KDMLayer

Maps an input KDM to an output KDM via a learned joint KDM.

```python
from kdm.layers import KDMLayer, RBFKernelLayer

kernel = RBFKernelLayer(sigma=0.5, dim=encoded_size)

kdm = KDMLayer(
    kernel=kernel,
    dim_x=encoded_size,   # dimensionality of input vectors
    dim_y=output_dim,     # dimensionality of output vectors
    n_comp=200,           # number of joint support components
    x_train=True,         # train input support points c_x
    y_train=True,         # train output support points c_y
    w_train=True,         # train component weights c_w
)
```

**Trainable parameters:**

| Parameter | Shape | Role |
|---|---|---|
| `kdm.c_x` | `(n_comp, dim_x)` | Input support points |
| `kdm.c_y` | `(n_comp, dim_y)` | Output support points |
| `kdm.c_w` | `(n_comp,)` | Component weights (abs-normalized at runtime) |

**`forward(rho_in)`** — inference:

```
rho_in:  (batch_size, n_comp_in, dim_x + 1)   ← input KDM
rho_out: (batch_size, n_comp,    dim_y + 1)   ← output KDM
```

The weight column of `rho_out` holds the conditional mixture weights; the
vector columns are filled with `c_y` (broadcast over the batch).

**`log_marginal(rho_in)`** — generative log-likelihood:

```
rho_in:  (batch_size, n_comp_in, dim_x + 1)
returns: (batch_size,)   ← per-sample log P(x) under the input marginal
```

Use this to add a generative term to the loss: `-w * kdm.log_marginal(rho_x).mean()`.

---

## KDMProjLayer

Projects raw input vectors onto a learned KDM; used for density estimation.
Takes flat vectors (not packed KDMs) as input.

```python
from kdm.layers import KDMProjLayer, RBFKernelLayer

kernel = RBFKernelLayer(sigma=0.5, dim=input_dim)

proj = KDMProjLayer(
    kernel=kernel,
    dim_x=input_dim,
    n_comp=500,
    x_train=True,
    w_train=True,
)
```

**`forward(x)`:**

```
x:       (batch_size, dim_x)
returns: (batch_size,)   ← unnormalized overlap with the learned KDM
```

Log-density: `torch.log(proj(x) + 1e-7) + kernel.log_weight()`.

---

## Kernels

All kernels are `nn.Module` subclasses implementing the `Kernel` ABC:

```
forward(A, B) → K
    A: (batch_size, n, d)
    B: (m, d)
    K: (batch_size, n, m)

log_weight() → scalar   ← log normalization constant for generative log-likelihood
```

### RBFKernelLayer

Gaussian kernel `exp(-‖a − b‖² / (2σ²))` with a trainable, strictly positive
bandwidth.

```python
from kdm.layers import RBFKernelLayer

kernel = RBFKernelLayer(
    sigma=0.5,           # initial bandwidth (must be > min_sigma)
    dim=encoded_size,    # input dimensionality (used by log_weight)
    trainable=True,      # whether sigma is a learned parameter
    min_sigma=1e-3,      # structural lower bound via softplus reparameterization
)
```

`sigma` is parameterized as `softplus(raw_sigma) + min_sigma`. Use the
property setter for assignment — **do not write to `raw_sigma` directly**:

```python
kernel.sigma = 0.3     # correct: inverts softplus internally
```

`log_weight()` returns `-dim * log(σ) - dim * log(π) / 2`, which is the
log normalization constant of the RBF kernel as a density over ℝ^d.

### CosineKernelLayer

Cosine similarity `(a/‖a‖) · (b/‖b‖)`.  `log_weight()` returns `0`.
Use when inputs are already L2-normalized (e.g. one-hot labels).

```python
from kdm.layers import CosineKernelLayer
kernel = CosineKernelLayer()
```

### CrossProductKernelLayer

Splits the input vector at `dim1` and applies two child kernels to each part,
then multiplies the results. `log_weight()` sums the children's log weights.

```python
from kdm.layers import CrossProductKernelLayer, RBFKernelLayer, CosineKernelLayer

kernel = CrossProductKernelLayer(
    dim1=continuous_dim,                               # split point
    kernel1=RBFKernelLayer(sigma=0.5, dim=continuous_dim),
    kernel2=CosineKernelLayer(),
)
```

Typical use: joint density over concatenated `[x_continuous, y_onehot]`.

### CompTransKernelLayer

Applies a learned transformation before the kernel. Useful when the encoder
is shared but the kernel space needs its own projection.

```python
from kdm.layers import CompTransKernelLayer, RBFKernelLayer

transform = nn.Linear(input_dim, projected_dim)
kernel = CompTransKernelLayer(
    transform=transform,
    kernel=RBFKernelLayer(sigma=0.5, dim=projected_dim),
)
```

`log_weight()` delegates to the inner kernel.

---

## Initialization

Always call an init helper before the first optimizer step.

```python
from kdm.init import init_kdm_layer, init_kdm_proj_layer

# For KDMLayer
init_kdm_layer(
    layer=model.kdm,
    encoded_x=enc_sub,      # (n_comp, dim_x) — encoded training subset
    samples_y=y_sub,        # (n_comp, dim_y) — target subset (one-hot or raw)
    init_sigma=True,        # set sigma from mean 2nd-nearest-neighbor distance
    sigma_mult=1.0,         # optional multiplier on the computed sigma
)

# For KDMProjLayer
init_kdm_proj_layer(
    layer=model.kdmproj,
    samples_x=x_sub,        # (n_comp, dim_x) — training subset
    init_sigma=True,
)
```

The subset size must equal `n_comp`. A random permutation is the standard
approach:

```python
idx = torch.randperm(len(X_train))[:n_comp]
model.eval()
with torch.no_grad():
    enc_sub = encoder(X_train[idx])
init_kdm_layer(model.kdm, enc_sub.detach(), y_sub[idx], init_sigma=True)
model.train()
```

---

## Building a custom model

Minimal example of a custom two-headed model (shared encoder, two KDM
outputs) that illustrates direct use of `KDMLayer`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from kdm.layers import KDMLayer, RBFKernelLayer
from kdm.utils import pure2dm, dm2discrete, dm_rbf_loglik
from kdm.init import init_kdm_layer


class MyDualHeadModel(nn.Module):
    def __init__(self, input_dim, encoded_size, n_classes, n_comp, sigma=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, encoded_size),
        )
        kernel = RBFKernelLayer(sigma=sigma, dim=encoded_size)

        # Classification head: output vectors live in one-hot label space
        self.kdm_cls = KDMLayer(
            kernel=kernel, dim_x=encoded_size, dim_y=n_classes, n_comp=n_comp
        )
        # Regression head: output vectors are scalar targets
        kernel_reg = RBFKernelLayer(sigma=sigma, dim=encoded_size)
        self.kdm_reg = KDMLayer(
            kernel=kernel_reg, dim_x=encoded_size, dim_y=1, n_comp=n_comp
        )
        self.sigma_y = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        rho_x = pure2dm(self.encoder(x))       # (bs, 1, encoded_size+1)
        rho_cls = self.kdm_cls(rho_x)          # (bs, n_comp, n_classes+1)
        rho_reg = self.kdm_reg(rho_x)          # (bs, n_comp, 2)
        probs = dm2discrete(rho_cls)            # (bs, n_classes)
        return probs, rho_reg

    def loss(self, x, y_cls, y_reg):
        probs, rho_reg = self.forward(x)
        cls_loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), y_cls)
        reg_loss = -dm_rbf_loglik(y_reg, rho_reg, self.sigma_y).mean()
        return cls_loss + reg_loss


# Initialization
model = MyDualHeadModel(input_dim=20, encoded_size=32, n_classes=3, n_comp=200)
idx = torch.randperm(len(X_train))[:200]
model.eval()
with torch.no_grad():
    enc_sub = model.encoder(X_train[idx])
y_onehot = F.one_hot(y_cls_train[idx].long(), 3).float()
init_kdm_layer(model.kdm_cls, enc_sub.detach(), y_onehot, init_sigma=True)
init_kdm_layer(model.kdm_reg, enc_sub.detach(), y_reg_train[idx], init_sigma=True)
model.train()
```
