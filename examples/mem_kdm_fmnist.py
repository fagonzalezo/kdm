"""Fashion-MNIST memory-based KDM classifier.

Pipeline
--------
1. Train a small residual CNN encoder with supervised contrastive loss (SupCon)
   to produce a 20-d L2-normalized space with good kNN geometry.
2. Freeze the encoder; build a MemKDMClassModelWrapper over the full training set.
3. Fit the single learnable parameter (sigma) via 1-D golden-section search on
   validation NLL — neighbour encodings are precomputed once, so each evaluation
   is a cheap tensor op with no encoder or faiss queries.

Usage
-----
    conda run -n pytorch python examples/mem_kdm_fmnist.py
    conda run -n pytorch python examples/mem_kdm_fmnist.py \\
        --encoded-size 20 --n-comp 200 --epochs 12 --sigma-method golden
    conda run -n pytorch python examples/mem_kdm_fmnist.py --sigma-method both
"""

import argparse
import math
import sys
import time

import faiss  # import before torch on macOS to avoid OpenMP conflict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from kdm.models.mem import MemKDMClassModelWrapper

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class _L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)


def make_encoder(encoded_size: int = 20) -> nn.Module:
    """Small residual CNN: (1,28,28) -> encoded_size-d unit vector.

    Architecture:
        Conv(1→32, 3x3) → BN → ReLU
        ResBlock(32→32)
        ResBlock(32→64, stride=2)   28→14
        ResBlock(64→64, stride=2)   14→7
        GlobalAvgPool → Linear(64, encoded_size) → L2-normalize
    """
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        _ResBlock(32, 32),
        _ResBlock(32, 64, stride=2),
        _ResBlock(64, 64, stride=2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, encoded_size),
        _L2Norm(),
    )


# ---------------------------------------------------------------------------
# Supervised contrastive loss
# ---------------------------------------------------------------------------

def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SupCon loss (Khosla et al. 2020).

    z: (2B, d)  — two views per image stacked: cat([z1, z2], dim=0)
    y: (2B,)    — class labels, repeated for both views
    """
    n = z.shape[0]
    sim = torch.mm(z, z.T) / temperature           # (2B, 2B)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()                    # numerical stability

    mask_self = torch.eye(n, dtype=torch.bool, device=z.device)
    mask_pos  = (y.unsqueeze(0) == y.unsqueeze(1)) & ~mask_self

    exp_sim = torch.exp(sim)
    exp_sum = exp_sim.masked_fill(mask_self, 0).sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(exp_sum + 1e-8)

    n_pos = mask_pos.sum(dim=1).clamp(min=1)
    return (-(log_prob * mask_pos).sum(dim=1) / n_pos).mean()


# ---------------------------------------------------------------------------
# Two-view dataset wrapper
# ---------------------------------------------------------------------------

class TwoViewDataset(Dataset):
    """Returns (view1, view2, label) for contrastive training."""

    def __init__(self, base_dataset, augment):
        self.ds = base_dataset
        self.aug = augment

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return self.aug(img), self.aug(img), label


# ---------------------------------------------------------------------------
# Encoder training
# ---------------------------------------------------------------------------

def train_encoder(
    encoder: nn.Module,
    train_ds,
    epochs: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Train the encoder with supervised contrastive loss."""
    augment = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    loader = DataLoader(
        TwoViewDataset(train_ds, augment),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),   # MPS does not support pin_memory
        drop_last=True,
    )
    encoder.to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_batches_per_epoch = len(loader)
    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss = 0.0
        for batch_idx, (v1, v2, labels) in enumerate(loader, 1):
            v1, v2, labels = v1.to(device), v2.to(device), labels.to(device)
            z = torch.cat([encoder(v1), encoder(v2)], dim=0)
            y = torch.cat([labels, labels], dim=0)
            loss = supcon_loss(z, y, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if batch_idx % max(1, n_batches_per_epoch // 4) == 0 or batch_idx == n_batches_per_epoch:
                print(f"    [{batch_idx:3d}/{n_batches_per_epoch}]  loss={total_loss / batch_idx:.4f}")
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"  epoch {epoch:2d}/{epochs}  avg_loss={total_loss / n_batches_per_epoch:.4f}  lr={lr_now:.2e}")

    encoder.cpu()
    return encoder


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(encoder: nn.Module, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Encode a tensor dataset in batches; returns float32 numpy array."""
    encoder.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(encoder(X[i : i + batch_size]).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# kNN evaluation (sanity check)
# ---------------------------------------------------------------------------

def eval_knn(
    enc_tr: np.ndarray,
    y_tr: np.ndarray,
    enc_te: np.ndarray,
    y_te: np.ndarray,
    k: int = 10,
) -> float:
    """kNN accuracy using a faiss flat L2 index."""
    index = faiss.IndexFlatL2(enc_tr.shape[1])
    index.add(enc_tr)
    _, I = index.search(enc_te, k)          # (N, k)
    nn_labels = y_tr[I]                      # (N, k)
    n_classes = int(y_tr.max()) + 1
    preds = np.array([
        np.bincount(row, minlength=n_classes).argmax() for row in nn_labels
    ])
    return float((preds == y_te).mean())


# ---------------------------------------------------------------------------
# Sigma optimisation: golden-section search
# ---------------------------------------------------------------------------

def _nll_for_sigma(
    sigma: float,
    x_enc: torch.Tensor,      # (N, d)
    x_neigh: torch.Tensor,    # (N, k, d)  — precomputed
    y_neigh: torch.Tensor,    # (N, k)     — int labels
    y_true: torch.Tensor,     # (N,)
    dim_y: int,
    batch_size: int = 512,
) -> float:
    """Mean NLL for a given sigma using cached neighbors (no encoder / faiss)."""
    total, n = 0.0, 0
    for i in range(0, x_enc.shape[0], batch_size):
        xe = x_enc[i : i + batch_size]
        xn = x_neigh[i : i + batch_size]
        yn = y_neigh[i : i + batch_size]
        yt = y_true[i : i + batch_size]

        # RBF weights: exp(-||xi - xn_j||^2 / (2 sigma^2)), then softmax-normalised
        sq_dist = ((xe.unsqueeze(1) - xn) ** 2).sum(-1)        # (B, k)
        log_k   = -sq_dist / (2.0 * sigma ** 2)
        w = (log_k - torch.logsumexp(log_k, dim=1, keepdim=True)).exp()  # (B, k)

        yn_ohe = F.one_hot(yn.long(), num_classes=dim_y).float()  # (B, k, C)
        probs  = (w.unsqueeze(-1) * yn_ohe).sum(1).clamp_min(1e-7)  # (B, C)
        total += F.nll_loss(torch.log(probs), yt.long(), reduction="sum").item()
        n += xe.shape[0]
    return total / n


def fit_sigma_golden(
    wrapper: MemKDMClassModelWrapper,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_iter: int = 30,
) -> float:
    """Golden-section search for the optimal sigma on the log scale.

    Precomputes all validation neighbors once; each of the ~n_iter evaluations
    is a pure tensor op with no encoder forward pass or faiss query.
    """
    print("  Precomputing validation encodings and neighbors...")
    val_enc = np.zeros((len(X_val), wrapper.encoded_size), dtype=np.float32)
    bs = 256
    wrapper.encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X_val), bs):
            enc = wrapper.encoder(torch.as_tensor(X_val[i : i + bs])).cpu().numpy()
            val_enc[i : i + enc.shape[0]] = enc

    _, I = wrapper.index.search(val_enc, wrapper.n_comp)
    x_neigh = torch.as_tensor(np.take(wrapper.samples_x_enc, I, axis=0))
    y_neigh = torch.as_tensor(np.take(wrapper.samples_y, I, axis=0))
    x_enc_t = torch.as_tensor(val_enc)
    y_true_t = torch.as_tensor(y_val)
    print(f"  Cached {len(val_enc)} val samples × {wrapper.n_comp} neighbors each.")

    sigma0 = float(wrapper.model.kernel.sigma.detach())
    lo = math.log(sigma0 / 8.0)
    hi = math.log(sigma0 * 8.0)
    phi = (math.sqrt(5) - 1) / 2
    print(f"  sigma0={sigma0:.4f}  search range=[{math.exp(lo):.4f}, {math.exp(hi):.4f}]")
    print(f"  {'iter':>4}  {'sigma':>10}  {'val_nll':>10}")

    def f(log_s: float) -> float:
        return _nll_for_sigma(
            math.exp(log_s), x_enc_t, x_neigh, y_neigh, y_true_t, wrapper.dim_y
        )

    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    print(f"  {0:>4}  {math.exp(c):>10.4f}  {fc:>10.4f}  (c)")
    print(f"  {0:>4}  {math.exp(d):>10.4f}  {fd:>10.4f}  (d)")

    for step in range(1, n_iter + 1):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
            print(f"  {step:>4}  {math.exp(c):>10.4f}  {fc:>10.4f}  interval=[{math.exp(a):.4f}, {math.exp(b):.4f}]")
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
            print(f"  {step:>4}  {math.exp(d):>10.4f}  {fd:>10.4f}  interval=[{math.exp(a):.4f}, {math.exp(b):.4f}]")

    best_sigma = math.exp((a + b) / 2.0)
    best_nll = f(math.log(best_sigma))
    print(f"\n  ✓ sigma* = {best_sigma:.6f}  val_nll={best_nll:.4f}  ({n_iter + 2} evaluations)")
    return best_sigma


def fit_sigma_adam(
    wrapper: MemKDMClassModelWrapper,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
) -> float:
    """Gradient-based sigma optimisation (notebook-style, for comparison)."""
    opt = torch.optim.Adam(wrapper.model.parameters(), lr=lr)
    best_val_nll = float("inf")
    best_sigma = float(wrapper.model.kernel.sigma.detach())
    bad = 0
    print(f"  {'epoch':>5}  {'train_nll':>10}  {'val_nll':>10}  {'sigma':>10}  {'note':>6}")
    for epoch in range(1, epochs + 1):
        wrapper.model.train()
        ep_loss, n = 0.0, 0
        for inputs, y_true in wrapper.iter_batches(X_val, y_val, batch_size=batch_size):
            probs = wrapper.model(inputs)
            loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), y_true.long())
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * y_true.size(0)
            n += y_true.size(0)
        wrapper.model.eval()
        val_loss, nv = 0.0, 0
        with torch.no_grad():
            for inputs, y_true in wrapper.iter_batches(X_val, y_val, batch_size=batch_size, shuffle=False):
                probs = wrapper.model(inputs)
                val_loss += F.nll_loss(torch.log(probs.clamp_min(1e-7)), y_true.long()).item() * y_true.size(0)
                nv += y_true.size(0)
        val_nll = val_loss / nv
        sigma = float(wrapper.model.kernel.sigma.detach())
        note = ""
        if val_nll < best_val_nll - 1e-4:
            best_val_nll, best_sigma, bad, note = val_nll, sigma, 0, "best"
        else:
            bad += 1
            note = f"({bad}/{patience})"
        print(f"  {epoch:>5}  {ep_loss / n:>10.4f}  {val_nll:>10.4f}  {sigma:>10.4f}  {note}")
        if bad >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break
    print(f"\n  ✓ sigma* = {best_sigma:.6f}  best_val_nll={best_val_nll:.4f}")
    return best_sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="Fashion-MNIST memory-KDM classifier")
    parser.add_argument("--encoded-size", type=int, default=20,
                        help="Encoder output dimensionality (default: 20)")
    parser.add_argument("--n-comp", type=int, default=200,
                        help="Number of kNN neighbors used per query (default: 200)")
    parser.add_argument("--epochs", type=int, default=12,
                        help="SupCon training epochs (default: 12)")
    parser.add_argument("--sigma-method", choices=["golden", "adam", "both"], default="golden",
                        help="Sigma optimisation method (default: golden)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for contrastive training (default: 256)")
    parser.add_argument("--encoder-path", type=str, default=None,
                        help="Path to a saved encoder (.pt). If given, skip training and load this file. "
                             "The file is also saved here after training when this flag is absent.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("Fashion-MNIST Memory-KDM Classifier")
    print("=" * 60)
    print(f"  device        : {device}")
    print(f"  encoded_size  : {args.encoded_size}")
    print(f"  n_comp        : {args.n_comp}")
    print(f"  supcon_epochs : {args.epochs}")
    print(f"  sigma_method  : {args.sigma_method}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  encoder_path  : {args.encoder_path or '(train and save to encoder.pt)'}")
    print(f"  seed          : {args.seed}")

    # ---- Data ----------------------------------------------------------------
    print("\n[1/5] Loading Fashion-MNIST...")
    base_tf  = transforms.ToTensor()
    train_ds = datasets.FashionMNIST("~/.cache/fmnist", train=True,  download=True, transform=base_tf)
    test_ds  = datasets.FashionMNIST("~/.cache/fmnist", train=False, download=True, transform=base_tf)

    n_train = len(train_ds)
    rng     = np.random.RandomState(args.seed)
    perm    = rng.permutation(n_train)
    n_val   = n_train // 5
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    def stack(ds, idx=None):
        items = range(len(ds)) if idx is None else idx
        xs = torch.stack([ds[int(i)][0] for i in items])
        ys = torch.tensor([ds[int(i)][1] for i in items], dtype=torch.long)
        return xs, ys

    X_train_t, y_train_t = stack(train_ds, train_idx)
    X_val_t,   y_val_t   = stack(train_ds, val_idx)
    X_test_t,  y_test_t  = stack(test_ds)
    y_train_np = y_train_t.numpy()
    y_val_np   = y_val_t.numpy()
    y_test_np  = y_test_t.numpy()
    print(f"  train : {X_train_t.shape}  ({len(train_idx)} samples)")
    print(f"  val   : {X_val_t.shape}  ({len(val_idx)} samples)")
    print(f"  test  : {X_test_t.shape}  ({len(test_ds)} samples)")
    print(f"  classes: {int(y_train_t.max()) + 1}")

    # ---- Encoder training / loading ----------------------------------------
    print("\n[2/5] Encoder architecture")
    encoder  = make_encoder(encoded_size=args.encoded_size)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(encoder)
    print(f"  Total parameters: {n_params:,}")

    if args.encoder_path is not None:
        print(f"\n[3/5] Loading encoder weights from {args.encoder_path} ...")
        checkpoint = torch.load(args.encoder_path, map_location="cpu")
        encoder.load_state_dict(checkpoint["state_dict"])
        loaded_size = checkpoint.get("encoded_size")
        if loaded_size is not None and loaded_size != args.encoded_size:
            raise ValueError(
                f"Checkpoint encoded_size={loaded_size} does not match "
                f"--encoded-size {args.encoded_size}"
            )
        print(f"  Loaded (encoded_size={args.encoded_size}, "
              f"trained for {checkpoint.get('epochs', '?')} epochs).")
    else:
        save_path = "encoder.pt"
        print(f"\n[3/5] Training encoder with SupCon loss ({args.epochs} epochs, "
              f"batch={args.batch_size}, temp=0.1)...")
        print("  Large batches matter for contrastive learning: more negative pairs per step.")
        t0 = time.time()
        encoder = train_encoder(
            encoder, Subset(train_ds, train_idx.tolist()),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        t_enc = time.time() - t0
        print(f"\n  Encoder training done in {t_enc:.1f}s")
        torch.save({"state_dict": encoder.state_dict(),
                    "encoded_size": args.encoded_size,
                    "epochs": args.epochs}, save_path)
        print(f"  Encoder saved to {save_path}")

    print("\n  Computing embeddings for all splits...")
    t1 = time.time()
    enc_train = encode_all(encoder, X_train_t)
    enc_val   = encode_all(encoder, X_val_t)
    enc_test  = encode_all(encoder, X_test_t)
    print(f"  Encoded {len(enc_train)+len(enc_val)+len(enc_test)} samples in {time.time()-t1:.1f}s")
    print(f"  Embedding stats — mean norm: {np.linalg.norm(enc_train, axis=1).mean():.4f}  "
          f"(should be ~1.0 for L2-normalized)")

    print("\n  kNN sanity check (k=10, exact faiss):")
    knn_val  = eval_knn(enc_train, y_train_np, enc_val,  y_val_np,  k=10)
    knn_test = eval_knn(enc_train, y_train_np, enc_test, y_test_np, k=10)
    print(f"    val  kNN accuracy : {knn_val:.4f}")
    print(f"    test kNN accuracy : {knn_test:.4f}")

    # ---- Build MemKDM wrapper -----------------------------------------------
    print(f"\n[4/5] Building MemKDMClassModelWrapper")
    print(f"  Encoding {len(X_train_t)} training images and building faiss index "
          f"({args.encoded_size}-d flat L2)...")
    t0 = time.time()
    dim_y   = int(y_train_t.max().item()) + 1
    wrapper = MemKDMClassModelWrapper(
        encoded_size=args.encoded_size,
        dim_y=dim_y,
        samples_x=X_train_t.numpy(),
        samples_y=y_train_np,
        encoder=encoder,
        n_comp=args.n_comp,
        sigma=0.1,
    )
    wrapper.init_sigma(mult=1.0)
    sigma_init = float(wrapper.model.kernel.sigma.detach())
    t_wrap = time.time() - t0
    print(f"  faiss index: {wrapper.index.ntotal} vectors  built in {t_wrap:.1f}s")
    print(f"  sigma_init (median 1-NN distance × 1.0): {sigma_init:.6f}")
    print(f"  Learnable parameters: {sum(p.numel() for p in wrapper.model.parameters())} "
          f"(raw_sigma only)")

    def accuracy(wrap, X_np, y_np):
        probs = wrap.predict(X_np)
        return float((np.argmax(probs, axis=1) == y_np).mean())

    print("\n  Accuracy with initial sigma:")
    val_acc_init  = accuracy(wrapper, X_val_t.numpy(),  y_val_np)
    test_acc_init = accuracy(wrapper, X_test_t.numpy(), y_test_np)
    print(f"    val  : {val_acc_init:.4f}")
    print(f"    test : {test_acc_init:.4f}")

    # ---- Sigma optimisation -------------------------------------------------
    print(f"\n[5/5] Sigma optimisation (method={args.sigma_method})")

    t_golden, t_adam = None, None
    test_acc_g, test_acc_a = None, None

    if args.sigma_method in ("golden", "both"):
        print("\n--- Golden-section search on validation NLL ---")
        print("  Each evaluation is a cheap cached-neighbor tensor op; no encoder or faiss calls.")
        t0 = time.time()
        sigma_g = fit_sigma_golden(wrapper, X_val_t.numpy(), y_val_np)
        t_golden = time.time() - t0
        wrapper.model.kernel.sigma = sigma_g
        val_acc_g  = accuracy(wrapper, X_val_t.numpy(),  y_val_np)
        test_acc_g = accuracy(wrapper, X_test_t.numpy(), y_test_np)
        print(f"\n  Time: {t_golden:.2f}s")
        print(f"  val  accuracy : {val_acc_g:.4f}")
        print(f"  test accuracy : {test_acc_g:.4f}")

    if args.sigma_method in ("adam", "both"):
        print("\n--- Adam gradient optimisation (baseline comparison) ---")
        wrapper.model.kernel.sigma = sigma_init  # reset to initial
        t0 = time.time()
        sigma_a = fit_sigma_adam(wrapper, X_val_t.numpy(), y_val_np)
        t_adam = time.time() - t0
        wrapper.model.kernel.sigma = sigma_a
        val_acc_a  = accuracy(wrapper, X_val_t.numpy(),  y_val_np)
        test_acc_a = accuracy(wrapper, X_test_t.numpy(), y_test_np)
        print(f"\n  Time: {t_adam:.2f}s")
        print(f"  val  accuracy : {val_acc_a:.4f}")
        print(f"  test accuracy : {test_acc_a:.4f}")

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)
    print(f"  {'Method':<26}  {'test acc':>9}  {'time':>8}")
    print(f"  {'-'*26}  {'-'*9}  {'-'*8}")
    print(f"  {'kNN (k=10)':<26}  {knn_test:>9.4f}  {'—':>8}")
    print(f"  {'MemKDM (sigma_init)':<26}  {test_acc_init:>9.4f}  {'—':>8}")
    if test_acc_g is not None:
        print(f"  {'MemKDM + golden search':<26}  {test_acc_g:>9.4f}  {t_golden:>7.2f}s")
    if test_acc_a is not None:
        print(f"  {'MemKDM + Adam':<26}  {test_acc_a:>9.4f}  {t_adam:>7.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
