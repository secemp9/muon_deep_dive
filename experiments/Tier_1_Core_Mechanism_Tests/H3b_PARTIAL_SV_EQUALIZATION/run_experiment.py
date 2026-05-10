#!/usr/bin/env python3
"""
H3b: PARTIAL SV EQUALIZATION — Does Equalizing Only Top-k SVs Suffice?
========================================================================

MOTIVATION:
  Full polar factor sets ALL singular values to 1 (UV^T). But does Muon
  NEED full equalization? What if you equalize only the top-k SVs (set
  them to their mean) while keeping the rest proportional?

PROTOCOL:
  For each k in {1, 2, 4, 8, 16, 32(=full polar)}:
    1. Compute SVD of momentum: M = U Sigma V^T
    2. Set top-k sigma values to their mean: sigma_1=...=sigma_k = mean(sigma_{1:k})
    3. Keep sigma_{k+1}...sigma_n proportional (multiply by constant so
       ||step||_F = ||ortho(M)||_F)
    4. Use this as the step direction

  LR sweep for each k. Measure final loss. Plot loss vs k. Find the knee.

KEY TEST:
  If k=1 (just suppress the dominant SV) recovers >50% of full polar's
  advantage, the mechanism is primarily about preventing top-SV domination.

Setup: 4-layer deep linear 32x32, 500 steps, 5 seeds. Sweep k in {1,2,4,8,16,32}.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NUM_SEEDS = 5
BATCH_SIZE = 64

K_VALUES = [1, 2, 4, 8, 16, 32]  # 32 = full polar (all SVs equalized)
LR_CANDIDATES = [0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]


# =============================================================================
# NETWORK
# =============================================================================

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


# =============================================================================
# PARTIAL SV EQUALIZATION STEP
# =============================================================================

def partial_sv_equalize(M, k):
    """
    Partial SV equalization of momentum matrix M.

    Given M = U diag(sigma) V^T:
      - Top-k sigmas: set to their mean (equalize them)
      - Remaining sigmas: keep proportional, scale so ||result||_F = ||ortho(M)||_F

    k=DIM gives full polar factor (all SVs = same value => effectively UV^T scaled).
    k=0 gives original M rescaled to match norm.
    """
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    d = len(sigma)
    if np.linalg.norm(sigma) < 1e-15:
        return M

    sigma_new = sigma.copy()

    # Target Frobenius norm: ||ortho(M)||_F = sqrt(d) (polar factor has all SVs=1)
    target_norm = np.sqrt(d)

    # Equalize top-k: set them to their mean
    kk = min(k, d)
    if kk > 0:
        top_mean = np.mean(sigma[:kk])
        sigma_new[:kk] = top_mean

    if kk >= d:
        # Full equalization: all SVs = same value, scale to target norm
        # sigma_new[:] = top_mean => ||sigma_new|| = top_mean * sqrt(d)
        # We want ||result||_F = target_norm = sqrt(d)
        # So scale: sigma_new *= sqrt(d) / (top_mean * sqrt(d)) = 1/top_mean
        if top_mean > 1e-15:
            sigma_new *= target_norm / np.linalg.norm(sigma_new)
        return U @ np.diag(sigma_new) @ Vt

    # Partial equalization: top-k equalized, rest proportional
    # Scale the rest so total ||sigma_new|| = target_norm
    top_energy = np.sum(sigma_new[:kk]**2)
    remaining_energy_budget = target_norm**2 - top_energy

    if remaining_energy_budget > 1e-15:
        rest_current_energy = np.sum(sigma[kk:]**2)
        if rest_current_energy > 1e-15:
            scale = np.sqrt(remaining_energy_budget / rest_current_energy)
            sigma_new[kk:] = sigma[kk:] * scale
        else:
            # Rest is zero, distribute evenly
            sigma_new[kk:] = np.sqrt(remaining_energy_budget / (d - kk))
    else:
        # Top-k already exceeds budget: just normalize to target
        sigma_new[kk:] = 0
        sigma_new *= target_norm / np.linalg.norm(sigma_new)

    return U @ np.diag(sigma_new) @ Vt


# =============================================================================
# TRAINING
# =============================================================================

def train_partial_sv(weights_init, X, Y, lr, k):
    """Train with partial SV equalization applied to momentum."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(NUM_LAYERS):
            mom[i] = MOMENTUM * mom[i] + grads[i]
            step_dir = partial_sv_equalize(mom[i], k)
            weights[i] = weights[i] - lr * step_dir
    return compute_loss(weights, X, Y)


def train_sgd(weights_init, X, Y, lr):
    """Plain SGD with momentum (baseline)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(NUM_LAYERS):
            mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y)


def make_data(seed):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


# =============================================================================
# LR SWEEP
# =============================================================================

def sweep_lr(train_fn, seeds, candidates):
    """
    Sweep over LR candidates, use first 3 seeds for selection.
    train_fn(weights_init, X, Y, lr) -> final_loss.
    """
    best_lr, best_loss = candidates[-1], float('inf')
    for lr in candidates:
        losses = []
        for s in seeds[:3]:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            fl = train_fn(w, X, Y, lr)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        ml = np.mean(finite) if finite else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_lr = lr
    return best_lr


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H3b: PARTIAL SV EQUALIZATION -- Does Equalizing Only Top-k SVs Suffice?")
    print("=" * 100)
    print(f"k values: {K_VALUES} (k={DIM} = full polar)")
    print(f"Network: {NUM_LAYERS}-layer deep linear, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    # Phase 1: LR sweep for each k
    print("Phase 1: LR sweep per k...")
    best_lrs = {}
    for k in K_VALUES:
        fn = lambda w, X, Y, lr, _k=k: train_partial_sv(w, X, Y, lr, _k)
        best_lr = sweep_lr(fn, seeds, LR_CANDIDATES)
        best_lrs[k] = best_lr
        print(f"  k={k:>3}: best_lr={best_lr:.4f}")

    # Also sweep SGD baseline
    sgd_lr = sweep_lr(train_sgd, seeds, LR_CANDIDATES)
    print(f"  SGD:   best_lr={sgd_lr:.4f}")

    # Phase 2: Full training with all seeds
    print("\nPhase 2: Full training (all seeds)...")
    results_k = {}
    for k in K_VALUES:
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000)
            fl = train_partial_sv(w, X, Y, best_lrs[k], k)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        mean_loss = np.mean(finite) if finite else float('inf')
        std_loss = np.std(finite) if len(finite) > 1 else 0.0
        results_k[k] = {'mean': mean_loss, 'std': std_loss, 'lr': best_lrs[k]}
        print(f"  k={k:>3}: loss={mean_loss:.6e} +/- {std_loss:.6e}  (lr={best_lrs[k]:.4f})")

    # SGD baseline
    sgd_losses = []
    for s in seeds:
        X, Y = make_data(s)
        w = init_weights(s + 5000)
        fl = train_sgd(w, X, Y, sgd_lr)
        sgd_losses.append(fl)
    sgd_finite = [l for l in sgd_losses if np.isfinite(l)]
    sgd_mean = np.mean(sgd_finite) if sgd_finite else float('inf')
    sgd_std = np.std(sgd_finite) if len(sgd_finite) > 1 else 0.0
    print(f"  SGD:   loss={sgd_mean:.6e} +/- {sgd_std:.6e}  (lr={sgd_lr:.4f})")

    # ==========================================================================
    # RESULTS TABLE
    # ==========================================================================

    muon_loss = results_k[DIM]['mean']  # k=32 = full polar = Muon
    total_gap = sgd_mean - muon_loss

    print(f"\n\n{'=' * 100}")
    print("RESULTS: Final Loss vs Number of Equalized SVs (k)")
    print(f"{'=' * 100}")
    print(f"\n  SGD baseline loss:  {sgd_mean:.6e}")
    print(f"  Full Muon loss:     {muon_loss:.6e}")
    print(f"  Total gap:          {total_gap:.6e}")

    print(f"\n  {'k':>5}  {'Loss':>14}  {'Std':>14}  {'LR':>8}  {'vs SGD':>10}  {'vs Muon':>10}  {'% Gap Closed':>14}")
    print("  " + "-" * 82)
    print(f"  {'SGD':>5}  {sgd_mean:>14.6e}  {sgd_std:>14.6e}  {sgd_lr:>8.4f}  {'ref':>10}  {sgd_mean/max(muon_loss,1e-30):>10.2f}x  {'0.0%':>14}")
    for k in K_VALUES:
        r = results_k[k]
        vs_sgd = sgd_mean / max(r['mean'], 1e-30)
        vs_muon = r['mean'] / max(muon_loss, 1e-30)
        gap_closed = (sgd_mean - r['mean']) / max(total_gap, 1e-30) * 100 if total_gap > 1e-30 else 0
        marker = "  <-- full polar" if k == DIM else ""
        print(f"  {k:>5}  {r['mean']:>14.6e}  {r['std']:>14.6e}  {r['lr']:>8.4f}  "
              f"{vs_sgd:>10.2f}x  {vs_muon:>10.2f}x  {gap_closed:>13.1f}%{marker}")

    # ==========================================================================
    # HYPOTHESIS TESTS
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    # T1: k=1 recovers >50% of full polar advantage
    gap_k1 = (sgd_mean - results_k[1]['mean']) / max(total_gap, 1e-30) * 100 if total_gap > 1e-30 else 0
    t1 = gap_k1 > 50
    print(f"\n  T1: k=1 (suppress only top SV) captures >50% of Muon advantage?")
    print(f"      SGD loss:  {sgd_mean:.6e}")
    print(f"      k=1 loss:  {results_k[1]['mean']:.6e}")
    print(f"      Muon loss: {muon_loss:.6e}")
    print(f"      Gap closed by k=1: {gap_k1:.1f}%")
    if t1:
        print(f"      --> PASS: Mechanism is primarily about top-SV domination suppression")
    else:
        print(f"      --> FAIL: Need more than top-1 equalization; bottom SVs matter too")

    # T2: Find the knee (smallest k capturing >80% of gap)
    knee_k = None
    for k in K_VALUES:
        gap_k = (sgd_mean - results_k[k]['mean']) / max(total_gap, 1e-30) * 100 if total_gap > 1e-30 else 0
        if gap_k > 80:
            knee_k = k
            break
    t2 = knee_k is not None and knee_k < DIM // 2
    print(f"\n  T2: Sharp knee at k << dim (>80% gap captured at k < {DIM//2})?")
    if knee_k is not None:
        gap_at_knee = (sgd_mean - results_k[knee_k]['mean']) / max(total_gap, 1e-30) * 100
        print(f"      Knee at k={knee_k} ({gap_at_knee:.1f}% gap closed)")
    else:
        print(f"      No knee found (no k achieves >80% gap closure)")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    # T3: k=DIM exactly recovers Muon
    # Since k=DIM IS our Muon proxy, this is a sanity check (should be ~100%)
    gap_full = (sgd_mean - results_k[DIM]['mean']) / max(total_gap, 1e-30) * 100 if total_gap > 1e-30 else 0
    t3 = gap_full > 95
    print(f"\n  T3: k={DIM} (full equalization) recovers full polar performance?")
    print(f"      k={DIM} gap closed: {gap_full:.1f}%")
    print(f"      --> {'PASS (sanity check)' if t3 else 'FAIL (implementation issue?)'}")

    # ==========================================================================
    # DIMINISHING RETURNS ANALYSIS
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("DIMINISHING RETURNS ANALYSIS")
    print(f"{'=' * 100}")
    print(f"\n  Marginal gain from increasing k:")
    prev_gap = 0
    for k in K_VALUES:
        gap_k = (sgd_mean - results_k[k]['mean']) / max(total_gap, 1e-30) * 100 if total_gap > 1e-30 else 0
        marginal = gap_k - prev_gap
        bar = "#" * int(max(marginal, 0) / 2)
        print(f"    k={k:>3}: {gap_k:>7.1f}% total  (marginal +{marginal:>5.1f}%)  {bar}")
        prev_gap = gap_k

    # ==========================================================================
    # CONCLUSION
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("CONCLUSION")
    print(f"{'=' * 100}")

    tests_passed = sum([t1, t2, t3])
    print(f"\n  Tests passed: {tests_passed}/3")

    if t1:
        print(f"\n  PRIMARY FINDING: Equalizing just the top SV captures >{gap_k1:.0f}% of Muon's")
        print(f"  advantage. The mechanism is primarily about preventing top-SV domination.")
        print(f"  This supports the 'spectral democracy' interpretation: the main benefit")
        print(f"  comes from preventing any single direction from dominating the update.")
    else:
        print(f"\n  PRIMARY FINDING: Top-1 equalization captures only {gap_k1:.0f}% of the gap.")
        print(f"  Muon's benefit requires equalizing MANY singular values, not just")
        print(f"  suppressing the dominant one. This supports the 'full democracy'")
        print(f"  interpretation: ALL directions need equal voice.")

    if knee_k is not None:
        print(f"\n  KNEE: k={knee_k} captures >80% of the advantage.")
        if knee_k <= DIM // 4:
            print(f"  Only {knee_k}/{DIM} SVs need equalization -- the top few dominate.")
        else:
            print(f"  Need {knee_k}/{DIM} SVs equalized -- benefit is distributed.")
    else:
        print(f"\n  NO KNEE: Need full equalization for >80% benefit.")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")

    # ==========================================================================
    # PLOT
    # ==========================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'H3b: Partial SV Equalization\n'
                     f'{NUM_LAYERS}-layer deep linear {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds',
                     fontsize=13, fontweight='bold')

        # (a) Loss vs k
        ax = axes[0]
        k_vals = K_VALUES
        loss_vals = [results_k[k]['mean'] for k in k_vals]
        std_vals = [results_k[k]['std'] for k in k_vals]
        ax.errorbar(k_vals, loss_vals, yerr=std_vals, marker='o', linewidth=2,
                     capsize=4, color='#CC3311', label='Partial SV Eq.')
        ax.axhline(y=sgd_mean, color='#4477AA', linestyle='--', linewidth=1.5, label=f'SGD ({sgd_mean:.2e})')
        ax.axhline(y=muon_loss, color='#228B22', linestyle='--', linewidth=1.5, label=f'Full Polar ({muon_loss:.2e})')
        ax.set_xlabel('k (number of SVs equalized)')
        ax.set_ylabel('Final Loss')
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(k) for k in k_vals])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('Loss vs k')

        # (b) % Gap Closed vs k
        ax = axes[1]
        gaps = [(sgd_mean - results_k[k]['mean']) / max(total_gap, 1e-30) * 100 for k in k_vals]
        ax.plot(k_vals, gaps, marker='s', linewidth=2, color='#9933CC')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
        ax.axhline(y=100, color='#228B22', linestyle='-', alpha=0.3, label='100% (full polar)')
        ax.set_xlabel('k (number of SVs equalized)')
        ax.set_ylabel('% of SGD-Muon Gap Closed')
        ax.set_xscale('log', base=2)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(k) for k in k_vals])
        ax.set_ylim(-5, 110)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('Gap Closure vs k')

        # (c) Marginal gain
        ax = axes[2]
        marginals = [gaps[0]] + [gaps[i] - gaps[i-1] for i in range(1, len(gaps))]
        ax.bar(range(len(k_vals)), marginals, color='#FF8800', edgecolor='black')
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels([f'k={k}' for k in k_vals], rotation=30)
        ax.set_ylabel('Marginal % Gap Closed')
        ax.set_title('Marginal Gain per k Step')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(marginals):
            ax.text(i, max(v, 0) + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(SCRIPT_DIR, 'h3b_partial_sv_equalization.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {plot_path}")
    except ImportError:
        print("\nWARNING: matplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
