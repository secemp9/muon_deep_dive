#!/usr/bin/env python3
"""
H22b: Extreme Anisotropy -- Polar Factor Wastes Budget on Noise SVs
=====================================================================

FROM H17: Negative correlation r=-0.70 between anisotropy and Muon benefit.
At extreme anisotropy (sigma1/sigma_min > 1000), Muon advantage DROPS.

HYPOTHESIS:
  When gradient anisotropy is extreme, the gradient is effectively rank-k
  (k << dim) with the bottom (dim-k) SVs being noise. The polar factor
  sets ALL SVs to 1, which amplifies the noise SVs by a factor of
  sigma_1/sigma_noise >> 1. This wastes update budget on noise directions
  and can even HURT by injecting noise into the optimization.

  Specifically: if G has effective rank k, then ortho(G) allocates
  (dim-k)/dim fraction of its Frobenius norm to noise directions.
  For k=1 (near rank-1 gradient), this is (dim-1)/dim ~ 97% NOISE.

PROTOCOL:
  Single-layer 32x32, controlled gradient anisotropy via data construction.
  Construct data X with condition number kappa in {1, 10, 100, 1000, 10000}.
  Higher kappa -> more anisotropic gradients.

  For each kappa:
    1. Measure gradient effective rank and sigma1/sigma_min
    2. Decompose Muon's update: what fraction of ||ortho(G)||_F goes to
       signal SVs (top-k by gradient magnitude) vs noise SVs (bottom dim-k)?
    3. Train 500 steps, compare Muon vs SGD

  Also: construct "Muon-clip" = ortho(G) but zero out bottom (dim-k) SVs
  of the GRADIENT before orthogonalization. Does this fix the problem?

KEY MEASUREMENTS:
  - Noise fraction: ||ortho(G)_noise||_F^2 / ||ortho(G)||_F^2
  - Signal fraction: ||ortho(G)_signal||_F^2 / ||ortho(G)||_F^2
  - Whether noise fraction correlates with Muon losing its advantage
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

KAPPA_VALUES = [1, 10, 100, 1000, 10000]

LR_SGD = np.logspace(-4, -1, 12)
LR_MUON = np.logspace(-4, -1, 12)
LR_CLIP = np.logspace(-4, -1, 12)


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
    sv2 = sv**2
    sv2 = sv2 / (np.sum(sv2) + 1e-30)
    sv2 = sv2[sv2 > 1e-30]
    entropy = -np.sum(sv2 * np.log(sv2))
    return np.exp(entropy)


def make_anisotropic_data(kappa, seed):
    """Create data with controlled condition number."""
    rng = np.random.RandomState(seed)
    # Input data with condition number kappa
    U, _ = np.linalg.qr(rng.randn(DIM, DIM))
    sigmas = np.logspace(0, -np.log10(kappa), DIM)
    X = U @ np.diag(sigmas) @ rng.randn(DIM, BATCH_SIZE)

    W_target = rng.randn(DIM, DIM) * 0.5
    Y = W_target @ X
    return X, Y


def noise_fraction_analysis(G):
    """
    Analyze how much of ortho(G)'s energy goes to noise SVs.
    Define "signal" as top-k SVs of G where k = effective_rank(G).
    """
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    er = int(np.round(effective_rank(G)))
    er = max(1, min(er, len(sigma)))

    # Polar factor
    Q = U @ Vt  # ortho(G) = UV^T, all SVs = 1

    # Signal: contribution of top-er right singular vectors
    # In polar factor, sigma_i = 1 for all i
    # Signal fraction = er / dim (since all SVs equal in polar factor)
    signal_frac = er / len(sigma)
    noise_frac = 1.0 - signal_frac

    # Gradient signal concentration
    grad_signal = np.sum(sigma[:er]**2) / (np.sum(sigma**2) + 1e-30)

    return signal_frac, noise_frac, er, grad_signal


def muon_clip_step(G, k):
    """Muon but zero out bottom (dim-k) SVs of G before orthogonalization."""
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    sigma_clip = sigma.copy()
    sigma_clip[k:] = 0
    G_clip = U @ np.diag(sigma_clip) @ Vt
    return newton_schulz(G_clip)


def train(X, Y, lr, opt, seed):
    rng = np.random.RandomState(seed)
    W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
    mom = np.zeros_like(W)

    for step in range(NUM_STEPS):
        pred = W @ X
        loss = 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        N = X.shape[1]
        G = (pred - Y) @ X.T / N

        if opt == 'muon':
            mom = MOMENTUM * mom + newton_schulz(G)
        elif opt == 'sgd':
            mom = MOMENTUM * mom + G
        elif opt.startswith('muon_clip_'):
            k = int(opt.split('_')[-1])
            mom = MOMENTUM * mom + muon_clip_step(G, k)

        W = W - lr * mom

    pred = W @ X
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H22b: EXTREME ANISOTROPY -- POLAR FACTOR WASTES BUDGET ON NOISE SVs")
    print("=" * 100)
    print(f"Data kappa: {KAPPA_VALUES}")
    print(f"Single layer {DIM}x{DIM}, {NUM_STEPS} steps")
    print()

    results = {}

    for kappa in KAPPA_VALUES:
        print(f"\n  kappa={kappa}")

        # Measure gradient properties
        noise_fracs = []
        eff_ranks = []
        grad_signals = []
        for s in seeds[:3]:
            X, Y = make_anisotropic_data(kappa, s)
            rng = np.random.RandomState(s + 5000)
            W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
            N = X.shape[1]
            G = ((W @ X - Y) @ X.T) / N

            sf, nf, er, gs = noise_fraction_analysis(G)
            noise_fracs.append(nf)
            eff_ranks.append(er)
            grad_signals.append(gs)

        mean_nf = np.mean(noise_fracs)
        mean_er = np.mean(eff_ranks)
        mean_gs = np.mean(grad_signals)
        print(f"    Eff rank: {mean_er:.1f}/{DIM}, Noise fraction in ortho(G): {mean_nf:.3f}, "
              f"Gradient signal concentration: {mean_gs:.3f}")

        # LR sweep for SGD, Muon, and Muon-clip
        k_clip = max(1, int(np.round(mean_er)))
        opts = [('sgd', LR_SGD), ('muon', LR_MUON), (f'muon_clip_{k_clip}', LR_CLIP)]

        best = {}
        for opt_name, grid in opts:
            best_lr, best_loss = grid[-1], float('inf')
            for lr in grid:
                losses = [train(*make_anisotropic_data(kappa, s), lr, opt_name, s+5000)
                          for s in seeds[:3]]
                ml = np.mean([l for l in losses if np.isfinite(l)] or [float('inf')])
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr
            best[opt_name] = best_lr

        # Full eval
        final = {}
        for opt_name, _ in opts:
            losses = [train(*make_anisotropic_data(kappa, s), best[opt_name], opt_name, s+5000)
                      for s in seeds]
            final[opt_name] = np.mean([l for l in losses if np.isfinite(l)] or [float('inf')])

        adv_muon = final['sgd'] / max(final['muon'], 1e-30)
        adv_clip = final['sgd'] / max(final[f'muon_clip_{k_clip}'], 1e-30)

        results[kappa] = {
            'noise_frac': mean_nf, 'eff_rank': mean_er,
            'sgd': final['sgd'], 'muon': final['muon'],
            'muon_clip': final[f'muon_clip_{k_clip}'],
            'adv_muon': adv_muon, 'adv_clip': adv_clip, 'k_clip': k_clip,
        }
        print(f"    Muon adv: {adv_muon:.1f}x, Muon-clip(k={k_clip}) adv: {adv_clip:.1f}x")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS: NOISE FRACTION vs MUON ADVANTAGE")
    print(f"{'='*100}")

    print(f"\n  {'kappa':>8}  {'Noise%':>8}  {'EffRank':>8}  {'Muon adv':>10}  {'Clip adv':>10}  {'Clip fixes?':>12}")
    print("  " + "-" * 60)
    for kappa in KAPPA_VALUES:
        r = results[kappa]
        fixes = "YES" if r['adv_clip'] > r['adv_muon'] * 1.2 else "NO"
        print(f"  {kappa:>8}  {r['noise_frac']*100:>7.1f}%  {r['eff_rank']:>8.1f}  "
              f"{r['adv_muon']:>10.1f}x  {r['adv_clip']:>10.1f}x  {fixes:>12}")

    # Correlation: noise fraction vs advantage
    nfs = [results[k]['noise_frac'] for k in KAPPA_VALUES]
    advs = [results[k]['adv_muon'] for k in KAPPA_VALUES]
    r_corr = np.corrcoef(nfs, np.log(np.clip(advs, 1e-10, None)))[0, 1]

    print(f"\n  Correlation(noise_fraction, log(advantage)): r = {r_corr:.3f}")

    t1 = r_corr < -0.3
    print(f"\n  T1: Higher noise fraction -> lower Muon advantage?  --> {'PASS' if t1 else 'FAIL'}  (r={r_corr:.3f})")

    # Does clipping help at high kappa?
    high_kappa = KAPPA_VALUES[-1]
    clip_helps = results[high_kappa]['adv_clip'] > results[high_kappa]['adv_muon'] * 1.2
    print(f"  T2: Muon-clip helps at kappa={high_kappa}?  --> {'PASS' if clip_helps else 'FAIL'}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
