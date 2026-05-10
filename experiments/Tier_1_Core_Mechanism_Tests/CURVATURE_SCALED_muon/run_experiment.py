#!/usr/bin/env python3
"""
Experiment 3.4: Curvature-Scaled Muon
======================================

CONTEXT:
  2.20 showed more NS steps = better directional alignment toward Newton, but
  k=20 HURTS final loss because the step size is locked at unit spectral norm.
  The Newton direction has norm ||H^{-1}g|| which varies per step.

  3.1 showed dividing by sqrt(curvature) diverges -- unbounded rescaling.

FIX:
  After ortho_k(G), rescale by a CLAMPED curvature-aware factor:
    scale = clip( ||G||_F / ||ortho_k(G)||_F * gamma,  min=0.1,  max=10.0 )
  where gamma is a tunable damping factor.

  Also test rescaling by ||momentum||_F (the gradient magnitude info that
  orthogonalization strips away).

SETUP:
  - 2-layer 4x4 deep linear network (32 params), 500 training steps
  - Variants:
    (a) Muon k=5                       (baseline)
    (b) Muon k=20                      (worse per 1.4c-ii)
    (c) Muon k=20 + clamped curvature rescale (gamma=1.0)
    (d) Muon k=20 + rescale by ||momentum||_F
    (e) Muon k=5 + clamped curvature rescale (gamma=1.0)

KEY TEST:
  Does (c) or (d) fix k=20's degradation? Does rescaling at k=5 help?
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4
NUM_LAYERS = 2
NUM_STEPS = 500
LR = 0.02
MOMENTUM = 0.9
GAMMA = 1.0          # damping factor for curvature rescaling
SCALE_MIN = 0.1
SCALE_MAX = 10.0
NUM_SEEDS = 10
DATA_POINTS = 32     # number of data vectors


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(dim, num_layers, seed):
    """Initialize layers near identity."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(dim) + rng.randn(dim, dim) * 0.1
        weights.append(W.copy())
    return weights


def forward_linear(weights, X):
    """Forward pass through deep linear net."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y_target):
    """MSE loss."""
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)


def compute_gradients(weights, X, Y_target):
    """Backprop through deep linear net."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    delta = (activations[-1] - Y_target) / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


# =============================================================================
# NEWTON-SCHULZ ITERATION
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=5):
    """Newton-Schulz iteration with Muon's quintic coefficients.
    a=3.4445, b=-4.7750, c=2.0315
    X_{k+1} = a*X + b*X@(X^T@X) + c*X@(X^T@X)^2
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        XtX = X.T @ X
        X_XtX = X @ XtX
        XtX2 = XtX @ XtX
        X_XtX2 = X @ XtX2
        X = a * X + b * X_XtX + c * X_XtX2

    return X


# =============================================================================
# OPTIMIZERS
# =============================================================================

def train_muon(weights, X, Y_target, lr, num_steps, ns_iters=5,
               rescale_mode='none', gamma=1.0, scale_min=0.1, scale_max=10.0,
               momentum=0.9):
    """
    Train with Muon optimizer.

    rescale_mode:
      'none'      -- standard Muon (no rescaling after ortho)
      'curvature' -- scale = clip(||G||_F / ||ortho(G)||_F * gamma, min, max)
      'momentum'  -- scale = ||velocity||_F  (gradient magnitude info)
    """
    num_layers = len(weights)
    velocities = [np.zeros_like(W) for W in weights]
    losses = []
    scales_used = []  # track per-step average scale across layers

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        losses.append(loss)

        grads = compute_gradients(weights, X, Y_target)

        step_scales = []
        for i in range(num_layers):
            G = grads[i]
            G_norm = np.linalg.norm(G, 'fro')

            # Newton-Schulz orthogonalization
            G_orth = newton_schulz_orthogonalize(G, num_iters=ns_iters)
            G_orth_norm = np.linalg.norm(G_orth, 'fro')

            # Compute rescaling factor
            if rescale_mode == 'curvature':
                if G_orth_norm > 1e-12:
                    scale = np.clip(G_norm / G_orth_norm * gamma, scale_min, scale_max)
                else:
                    scale = 1.0
                G_orth = G_orth * scale
            elif rescale_mode == 'momentum':
                # Use momentum norm as scale (after update)
                vel_norm = np.linalg.norm(velocities[i], 'fro')
                if vel_norm > 1e-12 and step > 0:
                    scale = np.clip(vel_norm, scale_min, scale_max)
                else:
                    scale = 1.0
                G_orth = G_orth * scale
            else:
                scale = 1.0

            step_scales.append(scale)

            # Momentum update
            velocities[i] = momentum * velocities[i] + G_orth
            weights[i] = weights[i] - lr * velocities[i]

        scales_used.append(np.mean(step_scales))

    final_loss = compute_loss(weights, X, Y_target)
    losses.append(final_loss)

    return losses, scales_used, weights


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_single_seed(seed):
    """Run all 5 variants for a single seed and return results."""
    rng = np.random.RandomState(seed)

    # Generate random target
    W_target = [rng.randn(DIM, DIM) * 0.3 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, DATA_POINTS) * 0.5
    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target

    variants = {
        '(a) Muon k=5': dict(ns_iters=5, rescale_mode='none'),
        '(b) Muon k=20': dict(ns_iters=20, rescale_mode='none'),
        '(c) k=20 + curv rescale': dict(ns_iters=20, rescale_mode='curvature', gamma=GAMMA),
        '(d) k=20 + mom rescale': dict(ns_iters=20, rescale_mode='momentum'),
        '(e) k=5 + curv rescale': dict(ns_iters=5, rescale_mode='curvature', gamma=GAMMA),
    }

    results = {}
    for name, kwargs in variants.items():
        w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
        losses, scales, _ = train_muon(
            w_init, X, Y_target, lr=LR, num_steps=NUM_STEPS,
            momentum=MOMENTUM, scale_min=SCALE_MIN, scale_max=SCALE_MAX,
            **kwargs
        )
        results[name] = {
            'losses': losses,
            'scales': scales,
            'final_loss': losses[-1],
        }

    return results


def main():
    print()
    print("=" * 100)
    print("  Experiment 3.4: Curvature-Scaled Muon")
    print("=" * 100)
    print()
    print("  CONTEXT: k=20 NS gives better Newton alignment but HURTS loss because step")
    print("  size is locked at unit spectral norm. Fix: rescale by clamped curvature factor.")
    print()
    print(f"  Config: {NUM_LAYERS}-layer {DIM}x{DIM} deep linear, {NUM_STEPS} steps, lr={LR}")
    print(f"  Curvature rescale: scale = clip(||G||/||ortho(G)|| * gamma, {SCALE_MIN}, {SCALE_MAX})")
    print(f"  gamma = {GAMMA}, momentum = {MOMENTUM}")
    print(f"  Averaging over {NUM_SEEDS} seeds")
    print()

    # =========================================================================
    # Run all seeds
    # =========================================================================
    variant_names = [
        '(a) Muon k=5',
        '(b) Muon k=20',
        '(c) k=20 + curv rescale',
        '(d) k=20 + mom rescale',
        '(e) k=5 + curv rescale',
    ]

    all_final_losses = {name: [] for name in variant_names}
    all_loss_curves = {name: [] for name in variant_names}
    all_scale_curves = {name: [] for name in variant_names}

    for i in range(NUM_SEEDS):
        seed = 42 + i * 137
        results = run_single_seed(seed)
        for name in variant_names:
            all_final_losses[name].append(results[name]['final_loss'])
            all_loss_curves[name].append(results[name]['losses'])
            all_scale_curves[name].append(results[name]['scales'])
        print(f"  Seed {i+1:2d}/{NUM_SEEDS}: "
              + "  ".join(f"{name.split(')')[0]})={results[name]['final_loss']:.2e}"
                         for name in variant_names))

    # =========================================================================
    # Summary Table
    # =========================================================================
    print()
    print("=" * 100)
    print("  SUMMARY TABLE (final loss, mean +/- std over seeds)")
    print("=" * 100)
    print()

    ref_losses = np.array(all_final_losses['(a) Muon k=5'])
    ref_mean = np.mean(ref_losses)

    print(f"  {'Variant':<30s} {'Final loss (mean)':>18s} {'Std':>12s} "
          f"{'vs (a) k=5':>14s} {'Median':>14s}")
    print(f"  {'-'*30} {'-'*18} {'-'*12} {'-'*14} {'-'*14}")

    for name in variant_names:
        fl = np.array(all_final_losses[name])
        fl_mean = np.mean(fl)
        fl_std = np.std(fl)
        fl_median = np.median(fl)
        if ref_mean > 1e-15:
            vs_ref = (fl_mean - ref_mean) / ref_mean * 100
            vs_str = f"{vs_ref:+.1f}%"
        else:
            vs_str = "N/A"
        print(f"  {name:<30s} {fl_mean:18.6e} {fl_std:12.2e} {vs_str:>14s} {fl_median:14.6e}")

    # =========================================================================
    # Rescaling factor analysis
    # =========================================================================
    print()
    print("=" * 100)
    print("  RESCALING FACTOR ANALYSIS")
    print("=" * 100)
    print()

    for name in variant_names:
        if 'rescale' in name:
            all_scales = np.array(all_scale_curves[name])  # (num_seeds, num_steps)
            scale_mean = np.mean(all_scales, axis=0)
            print(f"  {name}:")
            print(f"    Scale at step 0:   mean={scale_mean[0]:.4f}")
            print(f"    Scale at step 100: mean={scale_mean[min(100, len(scale_mean)-1)]:.4f}")
            print(f"    Scale at step 250: mean={scale_mean[min(250, len(scale_mean)-1)]:.4f}")
            print(f"    Scale at step 499: mean={scale_mean[-1]:.4f}")
            print(f"    Overall range: [{np.min(all_scales):.4f}, {np.max(all_scales):.4f}]")
            # How often does it hit the clamp?
            hit_min = np.mean(all_scales <= SCALE_MIN + 1e-8) * 100
            hit_max = np.mean(all_scales >= SCALE_MAX - 1e-8) * 100
            print(f"    Fraction hitting min clamp ({SCALE_MIN}): {hit_min:.1f}%")
            print(f"    Fraction hitting max clamp ({SCALE_MAX}): {hit_max:.1f}%")
            print()

    # =========================================================================
    # Loss curve comparison at key steps
    # =========================================================================
    print()
    print("=" * 100)
    print("  LOSS CURVES AT KEY STEPS (averaged over seeds)")
    print("=" * 100)
    print()

    check_steps = [0, 50, 100, 200, 300, 400, 500]
    header = f"  {'Step':>6}"
    for name in variant_names:
        short = name.split(')')[0] + ')'
        header += f"  {short:>16}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*16}" * len(variant_names))

    for step in check_steps:
        row = f"  {step:>6}"
        for name in variant_names:
            curves = np.array(all_loss_curves[name])
            if step < curves.shape[1]:
                mean_loss = np.mean(curves[:, step])
                row += f"  {mean_loss:16.6e}"
            else:
                row += f"  {'---':>16}"
        print(row)

    # =========================================================================
    # HYPOTHESIS TESTS
    # =========================================================================
    print()
    print("=" * 100)
    print("  HYPOTHESIS TESTS")
    print("=" * 100)
    print()

    # Gather arrays
    losses_a = np.array(all_final_losses['(a) Muon k=5'])
    losses_b = np.array(all_final_losses['(b) Muon k=20'])
    losses_c = np.array(all_final_losses['(c) k=20 + curv rescale'])
    losses_d = np.array(all_final_losses['(d) k=20 + mom rescale'])
    losses_e = np.array(all_final_losses['(e) k=5 + curv rescale'])

    # Test 1: k=20 is worse than k=5 (confirming the problem)
    mean_a = np.mean(losses_a)
    mean_b = np.mean(losses_b)
    t1_pass = mean_b > mean_a
    pct_1 = (mean_b - mean_a) / mean_a * 100 if mean_a > 1e-15 else float('nan')
    print(f"  T1: k=20 is worse than k=5 (confirms the problem)")
    print(f"      mean(b)={mean_b:.6e} vs mean(a)={mean_a:.6e} ({pct_1:+.1f}%)")
    print(f"      Per-seed wins for k=5: {np.sum(losses_a < losses_b)}/{NUM_SEEDS}")
    print(f"      {'PASS' if t1_pass else 'FAIL'}: k=20 {'IS' if t1_pass else 'is NOT'} worse")
    print()

    # Test 2: Curvature rescaling fixes k=20 (c better than b)
    mean_c = np.mean(losses_c)
    t2_pass = mean_c < mean_b
    pct_2 = (mean_c - mean_b) / mean_b * 100 if mean_b > 1e-15 else float('nan')
    print(f"  T2: Curvature rescaling fixes k=20 degradation? (c < b)")
    print(f"      mean(c)={mean_c:.6e} vs mean(b)={mean_b:.6e} ({pct_2:+.1f}%)")
    print(f"      Per-seed wins for (c): {np.sum(losses_c < losses_b)}/{NUM_SEEDS}")
    print(f"      {'PASS' if t2_pass else 'FAIL'}: rescaled k=20 {'IS' if t2_pass else 'is NOT'} better than plain k=20")
    print()

    # Test 3: Momentum rescaling fixes k=20 (d better than b)
    mean_d = np.mean(losses_d)
    t3_pass = mean_d < mean_b
    pct_3 = (mean_d - mean_b) / mean_b * 100 if mean_b > 1e-15 else float('nan')
    print(f"  T3: Momentum rescaling fixes k=20 degradation? (d < b)")
    print(f"      mean(d)={mean_d:.6e} vs mean(b)={mean_b:.6e} ({pct_3:+.1f}%)")
    print(f"      Per-seed wins for (d): {np.sum(losses_d < losses_b)}/{NUM_SEEDS}")
    print(f"      {'PASS' if t3_pass else 'FAIL'}: mom-rescaled k=20 {'IS' if t3_pass else 'is NOT'} better than plain k=20")
    print()

    # Test 4: Does curvature rescaling make k=20 match or beat k=5? (c <= a)
    t4_pass = mean_c <= mean_a * 1.05  # within 5%
    pct_4 = (mean_c - mean_a) / mean_a * 100 if mean_a > 1e-15 else float('nan')
    print(f"  T4: Does curvature rescaling make k=20 match k=5? (c <= a within 5%)")
    print(f"      mean(c)={mean_c:.6e} vs mean(a)={mean_a:.6e} ({pct_4:+.1f}%)")
    print(f"      Per-seed wins for (c) over (a): {np.sum(losses_c < losses_a)}/{NUM_SEEDS}")
    print(f"      {'PASS' if t4_pass else 'FAIL'}: rescaled k=20 {'MATCHES' if t4_pass else 'does NOT match'} k=5")
    print()

    # Test 5: Does rescaling at k=5 help? (e < a)
    mean_e = np.mean(losses_e)
    t5_pass = mean_e < mean_a
    pct_5 = (mean_e - mean_a) / mean_a * 100 if mean_a > 1e-15 else float('nan')
    print(f"  T5: Does rescaling at k=5 improve further? (e < a)")
    print(f"      mean(e)={mean_e:.6e} vs mean(a)={mean_a:.6e} ({pct_5:+.1f}%)")
    print(f"      Per-seed wins for (e) over (a): {np.sum(losses_e < losses_a)}/{NUM_SEEDS}")
    print(f"      {'PASS' if t5_pass else 'FAIL'}: k=5+rescale {'IS' if t5_pass else 'is NOT'} better than plain k=5")
    print()

    # Test 6: Best variant overall
    variant_means = {
        '(a)': mean_a, '(b)': mean_b, '(c)': mean_c,
        '(d)': mean_d, '(e)': mean_e,
    }
    best_name = min(variant_means, key=variant_means.get)
    best_val = variant_means[best_name]
    print(f"  T6: Best variant overall: {best_name} with mean final loss = {best_val:.6e}")
    print()

    # =========================================================================
    # Per-seed detail table
    # =========================================================================
    print("=" * 100)
    print("  PER-SEED FINAL LOSSES")
    print("=" * 100)
    print()
    header = f"  {'Seed':>6}"
    for name in variant_names:
        short = name.split(')')[0] + ')'
        header += f"  {short:>16}"
    header += f"  {'Best':>8}"
    print(header)
    print(f"  {'-'*6}" + f"  {'-'*16}" * len(variant_names) + f"  {'-'*8}")

    for i in range(NUM_SEEDS):
        row = f"  {i+1:>6}"
        seed_losses = {}
        for j, name in enumerate(variant_names):
            fl = all_final_losses[name][i]
            row += f"  {fl:16.6e}"
            seed_losses[name.split(')')[0] + ')'] = fl
        best = min(seed_losses, key=seed_losses.get)
        row += f"  {best:>8}"
        print(row)

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    print()
    print("=" * 100)
    print("  OVERALL VERDICT")
    print("=" * 100)
    print()

    if t1_pass:
        print("  [CONFIRMED] k=20 is worse than k=5 (the known problem).")
    else:
        print("  [UNEXPECTED] k=20 is NOT worse than k=5 in this run.")

    if t2_pass and t4_pass:
        print("  [SUCCESS] Curvature rescaling FIXES k=20 degradation and matches/beats k=5.")
    elif t2_pass:
        print("  [PARTIAL] Curvature rescaling improves k=20 but does not fully match k=5.")
    else:
        print("  [FAIL] Curvature rescaling does NOT fix k=20.")

    if t3_pass:
        print("  [SUCCESS] Momentum rescaling also fixes k=20 degradation.")
    else:
        print("  [FAIL] Momentum rescaling does NOT fix k=20.")

    if t5_pass:
        print("  [BONUS] Rescaling at k=5 provides further improvement.")
    else:
        print("  [NO BONUS] Rescaling at k=5 does not help (k=5 already good enough).")

    print()
    print("=" * 100)
    print("  EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
