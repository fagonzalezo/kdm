# KDM Regression — Full Template

Complete runnable scaffold for probabilistic regression with `KDMRegressModel`.
Produces a predictive distribution (mean + variance) for each input.

---

## Imports

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kdm.models import KDMRegressModel
from kdm.init import init_kdm_layer
from kdm.utils import dm_rbf_loglik
```

---

## Encoder

```python
input_dim    = X_train.shape[1]
encoded_size = 32

encoder = nn.Sequential(
    nn.Linear(input_dim, 64), nn.Tanh(),
    nn.Linear(64, encoded_size),
)
```

---

## Model

```python
n_comp = 200   # must match subset size used in init
dim_y  = y_train.shape[1] if y_train.ndim > 1 else 1

model = KDMRegressModel(
    encoded_size=encoded_size,
    dim_y=dim_y,
    encoder=encoder,
    n_comp=n_comp,
    sigma_x=0.5,
    sigma_y=0.5,
    sigma_x_trainable=True,
    sigma_y_trainable=True,
)
```

---

## Initialization (critical)

```python
y_init = y_train if y_train.ndim > 1 else y_train.unsqueeze(-1)

model.eval()
idx = torch.randperm(len(X_train))[:n_comp]
with torch.no_grad():
    enc_sub = encoder(X_train[idx])

init_kdm_layer(
    model.kdm,
    encoded_x=enc_sub.detach(),
    samples_y=y_init[idx],
    init_sigma=True,
)
model.train()
```

---

## Training loop

```python
y_2d       = y_train if y_train.ndim > 1 else y_train.unsqueeze(-1)
dataset    = TensorDataset(X_train, y_2d)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        rho_y = model(xb)
        loss  = -dm_rbf_loglik(yb, rho_y, model.sigma_y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        sx, sy = model.get_sigmas()
        print(f"Epoch {epoch+1}  loss={total_loss/len(dataloader):.4f}"
              f"  sigma_x={sx:.4f}  sigma_y={sy:.4f}")
```

---

## Inference

```python
model.eval()
mean, variance = model.predict_reg(X_test)   # shapes: (bs, dim_y), (bs,)
std            = variance.sqrt()

# Point prediction and 95% interval (approximate, Gaussian assumption)
lower = mean.squeeze() - 1.96 * std
upper = mean.squeeze() + 1.96 * std
```

---

## Optional: generative co-training

Generative co-training can improve calibration on sparse data. Freeze the
encoder and add the marginal log-likelihood term to the loss.

```python
from kdm.utils import pure2dm

model.encoder.requires_grad_(False)
opt_gen = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)
alpha = 0.1

for xb, yb in dataloader:
    rho_y  = model(xb)
    disc   = -dm_rbf_loglik(yb, rho_y, model.sigma_y).mean()
    rho_x  = pure2dm(model.encoder(xb))
    gen    = -model.kdm.log_marginal(rho_x).mean()
    loss   = disc + alpha * gen
    opt_gen.zero_grad(); loss.backward(); opt_gen.step()
```
