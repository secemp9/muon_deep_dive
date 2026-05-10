#!/usr/bin/env python3
"""
H3b: Partial Equalization -- Top-k SVs Capture Muon Advantage?
================================================================

MOTIVATION (from H3 surprise):
  Muon's polar factor equalizes ALL singular values to 1 (UV^T).
  This is qualitatively different from normalized SGD which preserves
  SV ratios. But does Muon NEED full equalization? Or does equalizing
  only the top-k singular values capture most of the advantage?

  This tests whether Muon's benefit comes from:
  (a) Suppressing the dominant SV (top-1 equalization suffices), or
  (b) Lifting ALL small SVs (full equalization needed).

PROTOCOL:
  Construct a "partial polar factor" optimizer:
    G = U diag(sigma) V^T
    For top-k SVs: set sigma_i = 1
    For remaining SVs: keep sigma_i / ||sigma|| (normalized but not equalized)
    Update = U diag(sigma_partial) V^T
  Sweep k in {1, 2, 4, 8, 16, 32 (=full Muon)}.
  Compare final loss for each k to full Muon and normalized SGD.

KEY TESTS:
  T1: Does k=1 (suppress only top SV) capture >50% of Muon's advantage?
  T2: Is there a sharp knee where most advantage is captured (k << dim)?
  T3: Does k=dim exactly recover Muon's performance (sanity check)?

Setup: 4-layer, 32x32, 500 steps, 10 seeds, LR swept per method.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NUM_SEEDS = 10
BATCH_SIZE = 64

K_VALUES = [1, 2, 4, 8, 16, 32]  # 32 = full Muon (all SVs equalized)
LR_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]


def partial_polar(G, k):
    """
    Partial polar factor: equalize top-k SVs, normalize the rest.
    k=dim gives full polar factor (all SVs=1).
    k=0 gives Frobenius-normalized gradient.
    """
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    d = len(sigma)
    sigma_new = sigma.copy()
    norm = np.linalg.norm(sigma)
    if norm < 1e-15:
        return G

    # Top-k: set to 1
    sigma_new[:min(k, d)] = 1.0
    # Remaining: normalize to have consistent scale
    if k < d:
        remaining = sigma[k:]
        r_norm = np.linalg.norm(remaining)
        if r_norm > 1e-15:
            # Scale remaining so their total energy matches what equalized would give
            sigma_new[k:] = remaining / r_norm * np.sqrt(d - k)

    return U @ np.diag(sigma_new) @ Vt


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def compute_gradients(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


def train(weights_init, X, Y, lr, k):
    """Train with partial polar factor (k=dim is full Muon, k=0 is normalized SGD)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(NUM_LAYERS):
            pp = partial_polar(grads[i], k)
            mom[i] = MOMENTUM * mom[i] + pp
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y)


def train_norm_sgd(weights_init, X, Y, lr):
    """Normalized SGD (Frobenius normalization of momentum)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(NUM_LAYERS):
            mom[i] = MOMENTUM * mom[i] + grads[i]
            v_norm = np.linalg.norm(mom[i], 'fro')
            step_dir = mom[i] / max(v_norm, 1e-12)
            weights[i] = weights[i] - lr * step_dir
    return compute_loss(weights, X, Y)


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3
    return X, Y


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H3b: PARTIAL EQUALIZATION -- Top-k SVs Capture Muon Advantage?")
    print("=" * 100)
    print(f"k values: {K_VALUES} (k={DIM} = full Muon)")
    print(f"Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    # LR sweep for each k
    print("Phase 1: LR sweep per k...")
    best_lrs = {}
    for k in K_VALUES:
        best_lr, best_loss = LR_CANDIDATES[-1], float('inf')
        for lr in LR_CANDIDATES:
            losses = []
            for s in seeds[:3]:
                X, Y = make_data(s)
                w = init_weights(s + 5000)
                fl = train(w, X, Y, lr, k)
                losses.append(fl)
            ml = np.mean([l for l in losses if np.isfinite(l)]) if any(np.isfinite(l) for l in losses) else float('inf')
            if ml < best_loss:
                best_loss = ml
                best_lr = lr
        best_lrs[k] = best_lr
        print(f"  k={k:>3}: best_lr={best_lr:.4f}")

    # Also sweep for normalized SGD
    best_norm_lr, best_norm_loss = LR_CANDIDATES[-1], float('inf')
    for lr in LR_CANDIDATES:
        losses = []
        for s in seeds[:3]:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            fl = train_norm_sgd(w, X, Y, lr)
            losses.append(fl)
        ml = np.mean([l for l in losses if np.isfinite(l)]) if any(np.isfinite(l) for l in losses) else float('inf')
        if ml < best_norm_loss:
            best_norm_loss = ml
            best_norm_lr = lr
    print(f"  NormSGD: best_lr={best_norm_lr:.4f}")

    # Full training
    print("\nPhase 2: Full training...")
    results = {}
    for k in K_VALUES:
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            fl = train(w, X, Y, best_lrs[k], k)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        results[k] = np.mean(finite) if finite else float('inf')

    norm_losses = []
    for s in seeds:
        X, Y = make_data(s)
        w = init_weights(s + 5000)
        fl = train_norm_sgd(w, X, Y, best_norm_lr)
        norm_losses.append(fl)
    norm_result = np.mean([l for l in norm_losses if np.isfinite(l)])

    # Results
    print(f"\n{'=' * 100}")
    print("RESULTS: Final Loss vs Number of Equalized SVs")
    print(f"{'=' * 100}")

    muon_loss = results[DIM]
    norm_loss = norm_result

    print(f"\n  {'k':>5}  {'Loss':>14}  {'vs NormSGD':>12}  {'vs Muon':>10}  {'% of gap':>10}")
    print("  " + "-" * 55)

    total_gap = norm_loss - muon_loss if np.isfinite(norm_loss) and np.isfinite(muon_loss) else 1.0

    print(f"  {'NormSGD':>5}  {norm_loss:>14.6e}  {'ref':>12}  {norm_loss/max(muon_loss,1e-30):>10.1f}x  {'0%':>10}")
    for k in K_VALUES:
        r = results[k]
        vs_norm = norm_loss / max(r, 1e-30)
        vs_muon = r / max(muon_loss, 1e-30)
        gap_captured = (norm_loss - r) / max(total_gap, 1e-30) * 100 if total_gap > 1e-30 else 0
        marker = " <-- full Muon" if k == DIM else ""
        print(f"  {k:>5}  {r:>14.6e}  {vs_norm:>12.1f}x  {vs_muon:>10.2f}x  {gap_captured:>9.1f}%{marker}")

    # Hypothesis tests
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    gap_k1 = (norm_loss - results[1]) / max(total_gap, 1e-30) * 100
    t1 = gap_k1 > 50
    print(f"\n  T1: k=1 captures >50% of Muon advantage?")
    print(f"      Gap captured: {gap_k1:.1f}%")
    print(f"      --> {'PASS' if t1 else 'FAIL'}")

    # Find knee
    gaps = [(k, (norm_loss - results[k]) / max(total_gap, 1e-30) * 100) for k in K_VALUES]
    knee_k = None
    for k, pct in gaps:
        if pct > 80:
            knee_k = k
            break
    t2 = knee_k is not None and knee_k < DIM // 2
    print(f"\n  T2: Sharp knee at k << dim (>80% gap captured at k < {DIM//2})?")
    print(f"      Knee at k={knee_k} ({gaps[K_VALUES.index(knee_k)][1]:.1f}% gap)" if knee_k else "      No knee found")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    t3 = abs(results[DIM] - muon_loss) / max(muon_loss, 1e-30) < 0.05
    print(f"\n  T3: k=dim recovers full Muon within 5%?")
    print(f"      k={DIM} loss={results[DIM]:.6e}, Muon loss={muon_loss:.6e}")
    print(f"      --> {'PASS (sanity check)' if t3 else 'FAIL (implementation bug?)'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
