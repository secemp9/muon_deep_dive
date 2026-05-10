#!/usr/bin/env python3
"""
H22b: MUON-CLIP -- Fix Muon at Extreme Anisotropy by Clipping Noise SVs
=========================================================================

MOTIVATION:
  H17: Muon LOSES at extreme anisotropy. When the gradient is effectively
  rank-k (k << dim), the polar factor sets ALL SVs to 1, amplifying noise
  SVs by sigma_1/sigma_noise >> 1. This wastes update budget on noise.

FIX: Muon-clip:
  Before orthogonalization, zero out SVs below a threshold.
  Keep only top-k SVs where k = effective rank of the momentum matrix.
  This prevents noise SVs from consuming step budget.

PROTOCOL:
  4-layer deep linear 32x32, ill-conditioned target:
    kappa_target in {1, 10, 100, 1000, 10000}.
  Compare: (a) SGD, (b) Muon, (c) Muon-clip (keep top-k SVs, k=erank of mom).
  LR sweep for all. 300 steps, 5 seeds.

KEY TEST: does Muon-clip restore advantage at high kappa where plain Muon struggles?

Setup: 4-layer, 32x32, 300 steps, 5 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

KAPPA_VALUES = [1, 10, 100, 1000, 10000]

LR_SGD = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
LR_MUON = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
LR_CLIP = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]


# =============================================================================
# NETWORK
# =============================================================================

def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def effective_rank(M):
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv ** 2
    total = np.sum(sv2)
    if total < 1e-30:
        return 1.0
    sv2 = sv2 / total
    sv2 = sv2[sv2 > 1e-30]
    entropy = -np.sum(sv2 * np.log(sv2))
    return np.exp(entropy)


def muon_clip_step(G, k):
    """Muon-clip: zero out bottom SVs of G, keep top-k, then orthogonalize."""
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    k = max(1, min(k, len(sigma)))
    sigma_clip = sigma.copy()
    sigma_clip[k:] = 0
    G_clip = U @ np.diag(sigma_clip) @ Vt
    return newton_schulz(G_clip)


def make_ill_conditioned_data(kappa, seed):
    """
    Create data such that the target W_star has condition number = kappa.
    This makes the optimization landscape ill-conditioned.
    """
    rng = np.random.RandomState(seed)
    # Target with specified condition number
    U, _ = np.linalg.qr(rng.randn(DIM, DIM))
    V, _ = np.linalg.qr(rng.randn(DIM, DIM))
    sigmas = np.logspace(0, -np.log10(max(kappa, 1)), DIM)
    W_target = U @ np.diag(sigmas) @ V.T

    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


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


# =============================================================================
# TRAINING
# =============================================================================

def train(weights_init, X, Y, lr, opt, clip_k=None):
    """
    Train with specified optimizer.
    opt: 'sgd', 'muon', 'muon_clip'
    clip_k: for muon_clip, number of SVs to keep per layer (can be a list or single int)
    """
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]

    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)

        for l in range(N_LAYERS):
            if opt == 'sgd':
                mom[l] = MOMENTUM * mom[l] + grads[l]
            elif opt == 'muon':
                mom[l] = MOMENTUM * mom[l] + newton_schulz(grads[l])
            elif opt == 'muon_clip':
                # Determine k from effective rank of momentum (or gradient if first step)
                if clip_k is not None:
                    k = clip_k
                else:
                    # Use erank of current momentum (or gradient for first step)
                    ref = mom[l] if np.linalg.norm(mom[l]) > 1e-15 else grads[l]
                    k = max(1, int(np.round(effective_rank(ref))))
                mom[l] = MOMENTUM * mom[l] + muon_clip_step(grads[l], k)

            weights[l] = weights[l] - lr * mom[l]

    return compute_loss(weights, X, Y)


# =============================================================================
# LR SWEEP
# =============================================================================

def sweep_lr(opt, lr_candidates, kappa, eval_seeds, clip_k=None):
    """Sweep LR using first 3 seeds, return best LR and its mean loss."""
    best_lr, best_loss = lr_candidates[-1], float('inf')
    for lr in lr_candidates:
        losses = []
        for s in eval_seeds:
            X, Y = make_ill_conditioned_data(kappa, s)
            w = init_weights(s + 5000)
            fl = train(w, X, Y, lr, opt, clip_k)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        ml = np.mean(finite) if finite else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_lr = lr
    return best_lr, best_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]
    sweep_seeds = seeds[:3]

    print("=" * 100)
    print("H22b: MUON-CLIP -- Fix Muon at Extreme Anisotropy")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}")
    print(f"Steps: {NUM_STEPS}, Seeds: {NUM_SEEDS}")
    print(f"kappa_target: {KAPPA_VALUES}")
    print()

    results = {}

    for kappa in KAPPA_VALUES:
        print(f"\n  kappa={kappa}")

        # Measure gradient anisotropy at init
        eranks = []
        anisotropies = []
        for s in sweep_seeds:
            X, Y = make_ill_conditioned_data(kappa, s)
            w = init_weights(s + 5000)
            grads = compute_gradients(w, X, Y)
            for G in grads:
                sv = np.linalg.svd(G, compute_uv=False)
                eranks.append(effective_rank(G))
                anisotropies.append(sv[0] / max(sv[-1], 1e-30))
        mean_erank = np.mean(eranks)
        mean_aniso = np.mean(anisotropies)
        k_clip = max(1, int(np.round(mean_erank)))
        print(f"    Gradient erank: {mean_erank:.1f}/{DIM}, anisotropy: {mean_aniso:.1f}, clip k={k_clip}")

        # LR sweep for SGD
        best_sgd_lr, _ = sweep_lr('sgd', LR_SGD, kappa, sweep_seeds)
        print(f"    SGD best LR: {best_sgd_lr}")

        # LR sweep for Muon
        best_muon_lr, _ = sweep_lr('muon', LR_MUON, kappa, sweep_seeds)
        print(f"    Muon best LR: {best_muon_lr}")

        # LR sweep for Muon-clip
        best_clip_lr, _ = sweep_lr('muon_clip', LR_CLIP, kappa, sweep_seeds, clip_k=k_clip)
        print(f"    Muon-clip best LR: {best_clip_lr}")

        # Full evaluation on all seeds
        sgd_losses = []
        muon_losses = []
        clip_losses = []
        for s in seeds:
            X, Y = make_ill_conditioned_data(kappa, s)
            w = init_weights(s + 5000)
            sgd_losses.append(train(w, X, Y, best_sgd_lr, 'sgd'))
            muon_losses.append(train(w, X, Y, best_muon_lr, 'muon'))
            clip_losses.append(train(w, X, Y, best_clip_lr, 'muon_clip', clip_k=k_clip))

        sgd_mean = np.mean([l for l in sgd_losses if np.isfinite(l)] or [float('inf')])
        muon_mean = np.mean([l for l in muon_losses if np.isfinite(l)] or [float('inf')])
        clip_mean = np.mean([l for l in clip_losses if np.isfinite(l)] or [float('inf')])

        adv_muon = sgd_mean / max(muon_mean, 1e-30)
        adv_clip = sgd_mean / max(clip_mean, 1e-30)
        clip_vs_muon = muon_mean / max(clip_mean, 1e-30)

        results[kappa] = {
            'sgd': sgd_mean, 'muon': muon_mean, 'clip': clip_mean,
            'adv_muon': adv_muon, 'adv_clip': adv_clip,
            'clip_vs_muon': clip_vs_muon,
            'erank': mean_erank, 'aniso': mean_aniso, 'k_clip': k_clip,
        }
        print(f"    SGD={sgd_mean:.4e}  Muon={muon_mean:.4e}  Clip={clip_mean:.4e}")
        print(f"    Muon adv: {adv_muon:.2f}x   Clip adv: {adv_clip:.2f}x   Clip/Muon: {clip_vs_muon:.2f}x")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("RESULTS: MUON-CLIP vs MUON vs SGD ACROSS ANISOTROPY")
    print(f"{'=' * 100}")

    print(f"\n  {'kappa':>8}  {'erank':>6}  {'aniso':>10}  {'SGD':>12}  {'Muon':>12}  "
          f"{'Clip':>12}  {'Muon adv':>10}  {'Clip adv':>10}  {'Clip>Muon?':>11}")
    print("  " + "-" * 100)
    for kappa in KAPPA_VALUES:
        r = results[kappa]
        clip_better = "YES" if r['clip_vs_muon'] > 1.1 else "NO"
        print(f"  {kappa:>8}  {r['erank']:>6.1f}  {r['aniso']:>10.0f}  {r['sgd']:>12.4e}  "
              f"{r['muon']:>12.4e}  {r['clip']:>12.4e}  {r['adv_muon']:>10.1f}x  "
              f"{r['adv_clip']:>10.1f}x  {clip_better:>11}")

    # KEY TEST: Does Muon-clip restore advantage at high kappa?
    print(f"\n  === KEY TEST: Does Muon-clip restore advantage at high kappa? ===")
    high_kappas = [k for k in KAPPA_VALUES if k >= 1000]
    for kappa in high_kappas:
        r = results[kappa]
        restored = r['adv_clip'] > r['adv_muon'] * 1.2
        print(f"    kappa={kappa}: Muon adv={r['adv_muon']:.2f}x, Clip adv={r['adv_clip']:.2f}x  "
              f"--> {'RESTORED' if restored else 'NOT RESTORED'}")

    # Does Muon struggle at high kappa?
    print(f"\n  === Muon struggle detection ===")
    if len(KAPPA_VALUES) >= 2:
        low_k = KAPPA_VALUES[0]
        for kappa in KAPPA_VALUES:
            r = results[kappa]
            r_low = results[low_k]
            lost = r['adv_muon'] < 1.0
            declined = r['adv_muon'] < r_low['adv_muon'] * 0.5
            status = "LOST" if lost else ("DECLINED" if declined else "OK")
            print(f"    kappa={kappa}: Muon adv={r['adv_muon']:.2f}x  [{status}]")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
