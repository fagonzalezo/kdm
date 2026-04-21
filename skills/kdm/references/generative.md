# KDM Generative Co-training — Full Template

Two-phase workflow:
1. **Phase 1** — Discriminative training (standard classification).
2. **Phase 2** — Generative co-training: freeze the encoder and add a marginal
   log-likelihood term that encourages the KDM to model the input distribution.

For conditional image generation, a decoder (e.g., a CNN autoencoder) is
trained separately and then used to map latent samples back to pixel space.

---

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
from kdm.utils import pure2dm, dm2comp
```

---

## Phase 1: Discriminative training

```python
input_dim    = X_train.shape[1]
encoded_size = 64
n_classes    = int(y_train.max().item()) + 1
n_comp       = 200

encoder = nn.Sequential(
    nn.Linear(input_dim, 128), nn.ReLU(),
    nn.Linear(128, encoded_size),
)
model = KDMClassModel(
    encoded_size=encoded_size, dim_y=n_classes,
    encoder=encoder, n_comp=n_comp, sigma=0.5,
)

y_onehot = F.one_hot(y_train.long(), n_classes).float()
idx      = torch.randperm(len(X_train))[:n_comp]
model.eval()
with torch.no_grad():
    enc_sub = encoder(X_train[idx])
init_kdm_layer(model.kdm, enc_sub.detach(), y_onehot[idx], init_sigma=True)
model.train()

dataset    = TensorDataset(X_train, y_train.long())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
opt_disc   = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for xb, yb in dataloader:
        probs = model(xb)
        loss  = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yb)
        opt_disc.zero_grad(); loss.backward(); opt_disc.step()
```

---

## Phase 2: Generative co-training

```python
model.encoder.requires_grad_(False)       # freeze encoder

opt_gen = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)
alpha = 0.1    # weight of generative term; tune between 0.01 – 1.0

for epoch in range(30):
    model.train()
    for xb, yb in dataloader:
        probs  = model(xb)
        disc   = F.nll_loss(torch.log(probs.clamp_min(1e-7)), yb)
        rho_x  = pure2dm(model.encoder(xb))
        gen    = -model.kdm.log_marginal(rho_x).mean()
        loss   = disc + alpha * gen
        opt_gen.zero_grad(); loss.backward(); opt_gen.step()
```

---

## Conditional generation via latent sampling

After co-training, the KDM joint distribution can be used to sample latent
encodings conditioned on a class label.

```python
@torch.no_grad()
def sample_class_latents(model, class_idx, n_samples=16):
    """Sample latent encodings for a given class from the KDM joint."""
    w, v = dm2comp(model.kdm.c_y.unsqueeze(0))   # (1, n_comp, dim_y)
    w  = w.squeeze(0)                             # (n_comp,)
    v  = v.squeeze(0)                             # (n_comp, dim_y)

    # Component class probabilities
    v_norm = F.normalize(v, p=2, dim=-1)
    class_probs = v_norm[:, class_idx] ** 2       # (n_comp,)

    # Weights proportional to joint: w_i * p(class | component_i)
    joint_w = w.abs() * class_probs
    joint_w = joint_w / joint_w.sum()

    # Sample component indices
    comp_idx = torch.multinomial(joint_w, n_samples, replacement=True)
    # Return corresponding input support points
    return model.kdm.c_x[comp_idx]               # (n_samples, encoded_size)
```

---

## Optional: decode latents to images

Train a decoder to map encoded representations back to pixel space, then pass
sampled latents through it.

```python
# Autoencoder decoder (adapt architecture to your encoder)
decoder = nn.Sequential(
    nn.Linear(encoded_size, 128), nn.ReLU(),
    nn.Linear(128, input_dim), nn.Sigmoid(),
)

# Train decoder to reconstruct training images from encoder outputs
opt_dec = torch.optim.Adam(decoder.parameters(), lr=1e-3)
for epoch in range(50):
    for xb, _ in dataloader:
        with torch.no_grad():
            latent = model.encoder(xb)
        recon = decoder(latent)
        loss  = F.mse_loss(recon, xb)
        opt_dec.zero_grad(); loss.backward(); opt_dec.step()

# Generate images for class 3
latents = sample_class_latents(model, class_idx=3, n_samples=16)
images  = decoder(latents)                         # (16, input_dim)
```
