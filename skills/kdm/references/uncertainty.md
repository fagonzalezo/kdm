# KDM Uncertainty Quantification — Full Template

KDMs are naturally calibrated: the kernel weights decay as inputs move away
from the training support, so entropy increases automatically for OOD samples.
No temperature scaling or post-hoc calibration is needed.

This template covers in-distribution calibration metrics and OOD detection
experiments (e.g., MNIST vs. rotated MNIST).

---

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
```

---

## Model setup

Use a standard `KDMClassModel` (see `references/classification.md` for full
training). The uncertainty analysis happens at inference time.

```python
# Assumes a trained `model: KDMClassModel`
model.eval()
```

---

## Uncertainty metrics

```python
@torch.no_grad()
def predict_with_uncertainty(model, x):
    probs = model(x)                                       # (bs, n_classes)
    pred  = probs.argmax(dim=-1)
    # Predictive entropy — max at log(n_classes) for uniform distribution
    entropy    = -(probs * torch.log(probs.clamp_min(1e-7))).sum(dim=-1)
    confidence = probs.max(dim=-1).values
    return pred, probs, entropy, confidence
```

---

## In-distribution calibration

Expected calibration error (ECE) — lower is better.

```python
def compute_ece(probs, labels, n_bins=10):
    confidences, predictions = probs.max(dim=-1)
    accuracies  = predictions.eq(labels)
    ece         = torch.zeros(1)
    bin_edges   = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            acc_bin  = accuracies[mask].float().mean()
            conf_bin = confidences[mask].mean()
            ece     += mask.float().mean() * (conf_bin - acc_bin).abs()
    return ece.item()

with torch.no_grad():
    probs_test = model(X_test)
ece = compute_ece(probs_test, y_test)
print(f"ECE: {ece:.4f}")
```

---

## OOD detection

Entropy increases for inputs far from the training distribution. Compare
in-distribution entropy vs. OOD entropy.

```python
@torch.no_grad()
def ood_entropy(model, x_in, x_ood):
    _, _, ent_in,  _ = predict_with_uncertainty(model, x_in)
    _, _, ent_ood, _ = predict_with_uncertainty(model, x_ood)
    print(f"In-dist  entropy: {ent_in.mean():.4f} ± {ent_in.std():.4f}")
    print(f"OOD      entropy: {ent_ood.mean():.4f} ± {ent_ood.std():.4f}")
    return ent_in, ent_ood

# Example: MNIST test set vs. rotated MNIST
ent_in, ent_ood = ood_entropy(model, X_test_in, X_test_ood)
```

AUROC for OOD detection (using entropy as the anomaly score):

```python
from sklearn.metrics import roc_auc_score
import numpy as np

scores = torch.cat([ent_in, ent_ood]).numpy()
labels = np.array([0] * len(ent_in) + [1] * len(ent_ood))   # 1 = OOD
auroc  = roc_auc_score(labels, scores)
print(f"OOD AUROC (entropy): {auroc:.4f}")
```

---

## Input log-likelihood OOD detection

Entropy captures *label-level* uncertainty ("is the model unsure which class?").
The KDM also provides *input-level* novelty via `log_marginal`: it measures how
well an input fits the density learned over the training distribution. A model
can be overconfident (low entropy) yet OOD (low log P(x)) — this catches cases
that entropy misses.

```python
from kdm.utils import pure2dm

@torch.no_grad()
def compute_log_density(model, X, batch_size=512):
    """log P(x) for each sample — higher = more in-distribution."""
    model.eval()
    dev = next(model.parameters()).device
    scores = []
    for i in range(0, len(X), batch_size):
        xb    = X[i:i+batch_size].to(dev)
        rho_x = pure2dm(model.encoder(xb))
        scores.append(model.kdm.log_marginal(rho_x))
    return torch.cat(scores).cpu()

log_p_in  = compute_log_density(model, X_test_in)
log_p_ood = compute_log_density(model, X_test_ood)

print(f"In-dist  log P(x): {log_p_in.mean():.4f} ± {log_p_in.std():.4f}")
print(f"OOD      log P(x): {log_p_ood.mean():.4f} ± {log_p_ood.std():.4f}")
```

Calibrate a detection threshold at a chosen false-positive rate:

```python
import numpy as np

threshold = np.percentile(log_p_in.numpy(), 5)   # 5% FPR on in-distribution
print(f"OOD threshold (5% FPR): log P(x) < {threshold:.3f}")
```

AUROC using `-log P(x)` as the anomaly score:

```python
scores = torch.cat([-log_p_in, -log_p_ood]).numpy()
labels = np.array([0] * len(log_p_in) + [1] * len(log_p_ood))   # 1 = OOD
auroc  = roc_auc_score(labels, scores)
print(f"OOD AUROC (input log-likelihood): {auroc:.4f}")
```

---

## Reliability diagram

```python
import matplotlib.pyplot as plt

def reliability_diagram(probs, labels, n_bins=10):
    confidences, predictions = probs.max(dim=-1)
    accuracies  = predictions.eq(labels).float()
    bin_edges   = torch.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc.append(accuracies[mask].mean().item())
            bin_conf.append(confidences[mask].mean().item())
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, label="KDM")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.legend(); plt.tight_layout(); plt.show()

with torch.no_grad():
    probs_test = model(X_test)
reliability_diagram(probs_test, y_test)
```
