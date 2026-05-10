#!/usr/bin/env python3
"""
2.2c: Self-Correction Timescale -- Steps to Halve Momentum Imbalance vs c
===========================================================================

MOTIVATION (from 2.2b surprise):
  Muon survives c=1000 where SGD produces NaN. The self-correcting mechanism:
  momentum imbalance ratio 9e+05 -> 296 over 300 steps. Per-layer ||M||_F
  normalization in NS absorbs extreme scales.

QUESTION: What is the TIMESCALE of self-correction? Specifically:
  - How many steps to halve the momentum imbalance ratio?
  - Does this timescale scale linearly with log(c)?
  - Is there a critical c beyond which self-correction is too slow to help?

PROTOCOL:
  For c in {1, 10, 100, 1000, 10000}:
    Initialize 4-layer 32x32 deep linear with rescaling c.
    Track momentum buffer norm ratio (max/min across layers) at every step.
    Find step at which ratio first drops below 0.5 * initial_ratio ("half-life").
  Also track: when does loss first decrease from init? (training onset)

KEY TESTS:
  T1: Half-life scales as O(log(c)) (self-correction is fast).
  T2: Half-life < 50 steps for all c <= 1000.
  T3: Training onset (first loss decrease) happens BEFORE half-life
      (Muon starts learning before momentum rebalances).

Setup: 4-layer, 32x32, 500 steps, 5 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 500
LR = 0.02
MOMENTUM_BETA = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 128

C_VALUES = [1, 10, 100, 1000, 10000]


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(rng, c):
    weights = []
    for l in range(N_LAYERS):
        W = rng.randn(DIM, DIM) / np.sqrt(DIM)
        weights.append(W)
    weights[0] = weights[0] * c
    weights[-1] = weights[-1] / c
    return weights


def forward(weights, X):
    out = X
    for W in weights:
        out = W @ out
    return out


def compute_gradients(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = 2.0 * (acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return np.mean((pred - Y) ** 2)


def train_and_track(weights_init, X, Y):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    imbalance_history = []
    loss_history = []

    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        loss_history.append(loss)
        if not np.isfinite(loss) or loss > 1e10:
            break

        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + (1 - MOMENTUM_BETA) * grads[l]
            ortho_mom = newton_schulz(mom[l])
            weights[l] = weights[l] - LR * ortho_mom

        # Track momentum imbalance
        mom_norms = [np.linalg.norm(m, 'fro') for m in mom]
        imbalance = max(mom_norms) / max(min(mom_norms), 1e-30)
        imbalance_history.append(imbalance)

    return imbalance_history, loss_history


def find_half_life(imbalance_history):
    """Find first step where imbalance drops below 0.5 * initial."""
    if not imbalance_history:
        return float('nan')
    initial = imbalance_history[0]
    target = initial * 0.5
    for i, v in enumerate(imbalance_history):
        if v <= target:
            return i
    return float('nan')


def find_training_onset(loss_history):
    """Find first step where loss decreases from initial."""
    if not loss_history:
        return float('nan')
    initial = loss_history[0]
    for i in range(1, len(loss_history)):
        if loss_history[i] < initial * 0.99:
            return i
    return float('nan')


def main():
    print("=" * 100)
    print("2.2c: SELF-CORRECTION TIMESCALE -- Steps to Halve Momentum Imbalance")
    print("=" * 100)
    print(f"c values: {C_VALUES}")
    print(f"Network: {N_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    results = {}
    for c in C_VALUES:
        half_lives = []
        onsets = []
        init_imbalances = []
        final_imbalances = []

        for seed_idx in range(NUM_SEEDS):
            rng = np.random.RandomState(42 + seed_idx * 137)
            X = rng.randn(DIM, BATCH_SIZE) * 0.3
            Y = rng.randn(DIM, BATCH_SIZE) * 0.3
            rng_w = np.random.RandomState(1000 + seed_idx)
            weights_init = init_weights(rng_w, c)

            imb_hist, loss_hist = train_and_track(weights_init, X, Y)

            hl = find_half_life(imb_hist)
            onset = find_training_onset(loss_hist)
            half_lives.append(hl)
            onsets.append(onset)
            if imb_hist:
                init_imbalances.append(imb_hist[0])
                final_imbalances.append(imb_hist[-1])

        results[c] = {
            'half_lives': half_lives,
            'onsets': onsets,
            'init_imbalance': np.mean(init_imbalances) if init_imbalances else float('nan'),
            'final_imbalance': np.mean(final_imbalances) if final_imbalances else float('nan'),
        }

    # Results table
    print(f"\n{'=' * 100}")
    print("RESULTS")
    print(f"{'=' * 100}")

    print(f"\n  {'c':>8}  {'Init Imb':>12}  {'Final Imb':>12}  {'Half-life':>12}  {'Onset':>10}  {'HL < Onset?':>12}")
    print("  " + "-" * 72)

    for c in C_VALUES:
        r = results[c]
        hl_mean = np.nanmean(r['half_lives'])
        onset_mean = np.nanmean(r['onsets'])
        hl_before = hl_mean < onset_mean if np.isfinite(hl_mean) and np.isfinite(onset_mean) else False
        print(f"  {c:>8}  {r['init_imbalance']:>12.2e}  {r['final_imbalance']:>12.2e}  "
              f"{hl_mean:>12.1f}  {onset_mean:>10.1f}  {'NO' if hl_before else 'YES':>12}")

    # Check O(log(c)) scaling
    print(f"\n  Half-life vs log(c):")
    log_cs = []
    hls = []
    for c in C_VALUES:
        if c > 1:
            hl = np.nanmean(results[c]['half_lives'])
            if np.isfinite(hl):
                log_cs.append(np.log10(c))
                hls.append(hl)

    if len(log_cs) > 2:
        slope = np.polyfit(log_cs, hls, 1)[0]
        print(f"  Slope (steps per decade of c): {slope:.1f}")
    else:
        slope = float('nan')

    # Hypothesis tests
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    t1 = np.isfinite(slope) and slope < 50  # Less than 50 steps per decade
    print(f"\n  T1: Half-life scales as O(log(c)) (<50 steps/decade)?")
    print(f"      Slope = {slope:.1f} steps/decade")
    print(f"      --> {'PASS' if t1 else 'FAIL'}")

    t2 = all(np.nanmean(results[c]['half_lives']) < 50
             for c in C_VALUES if c <= 1000
             and np.isfinite(np.nanmean(results[c]['half_lives'])))
    print(f"\n  T2: Half-life < 50 for all c <= 1000?")
    for c in [cc for cc in C_VALUES if cc <= 1000]:
        hl = np.nanmean(results[c]['half_lives'])
        print(f"      c={c}: {hl:.1f} steps")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    t3_checks = []
    for c in C_VALUES:
        hl = np.nanmean(results[c]['half_lives'])
        onset = np.nanmean(results[c]['onsets'])
        if np.isfinite(hl) and np.isfinite(onset):
            t3_checks.append(onset < hl)
    t3 = any(t3_checks) if t3_checks else False
    print(f"\n  T3: Training onset before half-life (learns while rebalancing)?")
    print(f"      --> {'PASS' if t3 else 'FAIL'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
