# KDM Classification — Full Template

Complete runnable scaffold for supervised classification with `KDMClassModel`.
Covers shallow (MLP) and deep (CNN) encoders, initialization, training, and
evaluation.

---

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
from kdm.losses import l1_norm
```

---

## Encoder

For tabular / low-dimensional data:

```python
input_dim  = X_train.shape[1]
encoded_size = 32
n_classes  = int(y_train.max().item()) + 1

encoder = nn.Sequential(
    nn.Linear(input_dim, 64), nn.ReLU(),
    nn.Linear(64, encoded_size),
)
```

For images (e.g. MNIST 1×28×28):

```python
encoded_size = 128

encoder = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, encoded_size),
)
```

---

## Model

```python
n_comp = 200   # must match subset size used in init

model = KDMClassModel(
    encoded_size=encoded_size,
    dim_y=n_classes,
    encoder=encoder,
    n_comp=n_comp,
    sigma=0.5,
    sigma_trainable=True,
)
```

---

## Initialization (critical)

```python
# One-hot encode labels for init
y_onehot = F.one_hot(y_train.long(), n_classes).float()

model.eval()
idx = torch.randperm(len(X_train))[:n_comp]
with torch.no_grad():
    enc_sub = encoder(X_train[idx])

init_kdm_layer(
    model.kdm,
    encoded_x=enc_sub.detach(),
    samples_y=y_onehot[idx],
    init_sigma=True,
    sigma_mult=1.0,
)
model.train()
```

---

## Training loop

```python
dataset    = TensorDataset(X_train, y_train.long())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        probs = model(xb)                                      # (bs, n_classes)
        nll   = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yb)
        reg   = 1e-3 * l1_norm(model.kdm.c_w.unsqueeze(0))    # optional
        loss  = nll + reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epochs}  loss={total_loss/len(dataloader):.4f}")
```

---

## Evaluation

```python
model.eval()
with torch.no_grad():
    probs = model(X_test)                                      # (bs, n_classes)
    preds = probs.argmax(dim=-1)
    acc   = (preds == y_test.long()).float().mean().item()
print(f"Test accuracy: {acc:.4f}")
```

---

## Optional: CNN encoder warmup

For image tasks it helps to pre-train the encoder with a standard softmax
cross-entropy loss before handing it to the KDM layer.

```python
warmup_head = nn.Linear(encoded_size, n_classes)
warmup_opt  = torch.optim.Adam(
    list(encoder.parameters()) + list(warmup_head.parameters()), lr=1e-3
)
for xb, yb in dataloader:
    logits = warmup_head(encoder(xb))
    loss   = F.cross_entropy(logits, yb)
    warmup_opt.zero_grad(); loss.backward(); warmup_opt.step()

# Re-initialize KDM components after warmup
model.eval()
with torch.no_grad():
    enc_sub = encoder(X_train[idx])
init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot[idx], init_sigma=True)
model.train()
```
