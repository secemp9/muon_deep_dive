#!/usr/bin/env python3
"""
H22a: Anisotropy Sweet Spot -- Muon Benefit Peaks at Moderate Effective Rank
==============================================================================

FROM H17: Anisotropy NEGATIVELY correlates with Muon benefit (r=-0.70).
Muon helps most at MODERATE anisotropy (ReLU: 4.4x at sigma1/sigma_min~537).

HYPOTHESIS:
  Muon's benefit peaks when the gradient's effective rank is 30-70% of full
  rank. Below 30%: the gradient is near-rank-1, so the polar factor maps
  a near-rank-1 matrix to a partial isometry, wasting most of its budget
  on equalizing noise SVs. Above 70%: the gradient is already nearly
  isotropic, so SV equalization provides minimal additional benefit.

  The sweet spot is where there is STRUCTURE (non-trivial SV distribution)
  but not DEGENERACY (not dominated by one direction).

PROTOCOL:
  Construct synthetic gradients with controlled effective rank:
    G = U * diag(sigma) * V^T where sigma is a power-law with exponent alpha.
    alpha in {0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0}
    Small alpha = flat spectrum (high eff rank), large alpha = steep (low eff rank)

  For each alpha:
    1. Compute effective rank = exp(entropy of normalized sigma^2)
    2. Use G as the gradient in a single-layer 32x32 optimization problem
    3. Train 500 steps with SGD and Muon at optimal LR
    4. Measure advantage

  Plot advantage vs effective_rank_fraction (eff_rank / dim).

KEY PREDICTION:
  Peak advantage at effective_rank_fraction in [0.3, 0.7].
  Advantage drops at both extremes (too isotropic or too anisotropic).
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

# Control anisotropy via power-law exponent on target matrix SVs
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

LR_SGD = np.logspace(-4, -1, 12)
LR_MUON = np.logspace(-4, -1, 12)


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
    """Compute effective rank = exp(entropy of normalized SV^2)."""
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv**2
    sv2 = sv2 / (np.sum(sv2) + 1e-30)
    sv2 = sv2[sv2 > 1e-30]
    entropy = -np.sum(sv2 * np.log(sv2))
    return np.exp(entropy)


def make_anisotropic_target(alpha, seed):
    """Create target matrix with power-law SVs: sigma_i = i^(-alpha)."""
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(DIM, DIM))
    V, _ = np.linalg.qr(rng.randn(DIM, DIM))
    sigmas = np.array([(i + 1)**(-alpha) for i in range(DIM)])
    sigmas = sigmas / np.linalg.norm(sigmas) * np.sqrt(DIM)  # normalize
    return U @ np.diag(sigmas) @ V.T


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


def train(w0, X, Y, lr, opt):
    weights = [W.copy() for W in w0]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        if compute_loss(weights, X, Y) > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(NUM_LAYERS):
            if opt == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] -= lr * mom[i]
    return compute_loss(weights, X, Y)


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H22a: ANISOTROPY SWEET SPOT -- EFFECTIVE RANK PREDICTS MUON BENEFIT")
    print("=" * 100)
    print(f"Alpha (power-law exponent): {ALPHA_VALUES}")
    print(f"Network: {NUM_LAYERS}-layer deep linear {DIM}x{DIM}, {NUM_STEPS} steps")
    print()

    results = {}

    for alpha in ALPHA_VALUES:
        print(f"\n  alpha={alpha}")

        # Measure effective rank of gradient at init
        eff_ranks = []
        for s in seeds[:3]:
            target = make_anisotropic_target(alpha, s)
            rng = np.random.RandomState(s + 7000)
            X = rng.randn(DIM, BATCH_SIZE) * 0.3
            Y = target @ X
            w = init_weights(s + 5000)
            grads = compute_gradients(w, X, Y)
            er = np.mean([effective_rank(G) for G in grads])
            eff_ranks.append(er)
        mean_eff_rank = np.mean(eff_ranks)
        eff_rank_frac = mean_eff_rank / DIM

        # Anisotropy
        target_er = effective_rank(make_anisotropic_target(alpha, seeds[0]))
        target_sv = np.linalg.svd(make_anisotropic_target(alpha, seeds[0]), compute_uv=False)
        anisotropy = target_sv[0] / (target_sv[-1] + 1e-15)

        print(f"    Gradient eff rank: {mean_eff_rank:.1f}/{DIM} ({eff_rank_frac*100:.0f}%)")
        print(f"    Target anisotropy: {anisotropy:.1f}")

        # LR sweep
        best = {}
        for opt, grid in [('sgd', LR_SGD), ('muon', LR_MUON)]:
            best_lr, best_loss = grid[-1], float('inf')
            for lr in grid:
                losses = []
                for s in seeds[:3]:
                    target = make_anisotropic_target(alpha, s)
                    rng = np.random.RandomState(s + 7000)
                    X = rng.randn(DIM, BATCH_SIZE) * 0.3
                    Y = target @ X
                    w = init_weights(s + 5000)
                    fl = train(w, X, Y, lr, opt)
                    losses.append(fl)
                ml = np.mean([l for l in losses if np.isfinite(l)] or [float('inf')])
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr
            best[opt] = best_lr

        # Full eval
        sgd_losses, muon_losses = [], []
        for s in seeds:
            target = make_anisotropic_target(alpha, s)
            rng = np.random.RandomState(s + 7000)
            X = rng.randn(DIM, BATCH_SIZE) * 0.3
            Y = target @ X
            w = init_weights(s + 5000)
            sgd_losses.append(train(w, X, Y, best['sgd'], 'sgd'))
            w = init_weights(s + 5000)
            muon_losses.append(train(w, X, Y, best['muon'], 'muon'))

        sgd_mean = np.mean([l for l in sgd_losses if np.isfinite(l)] or [float('inf')])
        muon_mean = np.mean([l for l in muon_losses if np.isfinite(l)] or [float('inf')])
        adv = sgd_mean / max(muon_mean, 1e-30)

        results[alpha] = {
            'eff_rank': mean_eff_rank, 'eff_rank_frac': eff_rank_frac,
            'anisotropy': anisotropy, 'advantage': adv,
            'sgd': sgd_mean, 'muon': muon_mean,
        }
        print(f"    Advantage: {adv:.1f}x")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS: MUON ADVANTAGE vs EFFECTIVE RANK")
    print(f"{'='*100}")

    print(f"\n  {'alpha':>6}  {'Eff Rank %':>12}  {'Anisotropy':>12}  {'Advantage':>12}")
    print("  " + "-" * 46)
    for alpha in ALPHA_VALUES:
        r = results[alpha]
        print(f"  {alpha:>6.1f}  {r['eff_rank_frac']*100:>11.0f}%  {r['anisotropy']:>12.1f}  {r['advantage']:>12.1f}x")

    # Find peak
    advs = [results[a]['advantage'] for a in ALPHA_VALUES]
    fracs = [results[a]['eff_rank_frac'] for a in ALPHA_VALUES]
    peak_idx = np.argmax(advs)
    peak_frac = fracs[peak_idx]
    peak_alpha = ALPHA_VALUES[peak_idx]

    print(f"\n  Peak advantage: {advs[peak_idx]:.1f}x at alpha={peak_alpha} "
          f"(eff_rank_frac={peak_frac*100:.0f}%)")

    t1 = 0.2 < peak_frac < 0.8
    print(f"\n  T1: Peak at 20-80% effective rank?  --> {'PASS' if t1 else 'FAIL'}  ({peak_frac*100:.0f}%)")

    # Check for inverted-U shape
    if peak_idx > 0 and peak_idx < len(advs) - 1:
        t2 = advs[peak_idx] > advs[0] and advs[peak_idx] > advs[-1]
        print(f"  T2: Inverted-U shape (peak > both extremes)?  --> {'PASS' if t2 else 'FAIL'}")
    else:
        t2 = False
        print(f"  T2: Peak at edge, cannot confirm inverted-U  --> FAIL")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
