# KDM Density Estimation — Full Template

Complete runnable scaffold for non-parametric density estimation with
`KDMDenEstModel`. After training, the model exposes a
`torch.distributions.MixtureSameFamily` for sampling and log-prob queries.

---

## Imports

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kdm.models import KDMDenEstModel
from kdm.init import init_kdm_proj_layer
```

---

## Model

No external encoder — `KDMDenEstModel` operates directly on raw inputs.

```python
input_dim = X_train.shape[1]
n_comp    = 500   # must match subset size used in init

model = KDMDenEstModel(
    dim_x=input_dim,
    sigma=0.5,
    n_comp=n_comp,
    trainable_sigma=True,
)
```

---

## Initialization (critical)

```python
idx = torch.randperm(len(X_train))[:n_comp]
init_kdm_proj_layer(
    model.kdmproj,
    samples_x=X_train[idx],
    init_sigma=True,
)
```

---

## Training loop

```python
dataset    = TensorDataset(X_train)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for (xb,) in dataloader:
        log_p = model(xb)          # per-sample log-density
        loss  = -log_p.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}  NLL={total_loss/len(dataloader):.4f}"
              f"  sigma={float(model.kernel.sigma.detach()):.4f}")
```

---

## Inference

```python
model.eval()
with torch.no_grad():
    log_p_test = model(X_test)             # (n_test,) log-densities

# Full distribution object (MixtureSameFamily)
distrib = model.get_distrib()
samples = distrib.sample((1000,))          # (1000, input_dim)
log_p   = distrib.log_prob(X_test)        # (n_test,)
```

---

## Joint density estimation (mixed continuous + discrete inputs)

For joint densities over `(x_continuous, x_discrete)` use
`KDMJointDenEstModel` with a `CrossProductKernelLayer` (RBF × Cosine).

```python
from kdm.models import KDMJointDenEstModel
from kdm.init import init_kdm_proj_layer

# x_cont: continuous features, x_disc: one-hot discrete features
dim_cont = x_cont_train.shape[1]
dim_disc = x_disc_train.shape[1]
n_comp   = 500

model = KDMJointDenEstModel(
    dim_cont=dim_cont,
    dim_disc=dim_disc,
    sigma=0.5,
    n_comp=n_comp,
)

idx = torch.randperm(len(x_cont_train))[:n_comp]
x_sub = torch.cat([x_cont_train[idx], x_disc_train[idx]], dim=-1)
init_kdm_proj_layer(model.kdmproj, x_sub, init_sigma=True)

# Training loop — concatenate inputs before passing to model
for x_cont_b, x_disc_b in dataloader:
    xb   = torch.cat([x_cont_b, x_disc_b], dim=-1)
    loss = -model(xb).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```
