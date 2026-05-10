#!/usr/bin/env python3
"""
Experiment H6: LR Artifact Check
=================================

CRITICAL QUESTION:
  Experiment 3.4 showed curvature rescaling gives ~130x improvement over vanilla
  Muon k=5. But the rescale factor hit the min-clamp (0.1) 96.5% of the time.
  This means the rescaler is effectively just multiplying by 0.1 = reducing the
  effective LR by 10x.

  Is the 130x improvement entirely explained by Muon's default LR (0.02) being
  10x too high? If lr=0.002 vanilla Muon matches rescaled Muon at lr=0.02, the
  answer is YES -- the "curvature rescaling" is just LR reduction.

SETUP:
  Same as 3.4: 2-layer 4x4 deep linear net, 500 steps, 10 seeds.

SWEEP:
  - Vanilla Muon at LR = {0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1}
  - Curvature-rescaled Muon k=5 at lr=0.02 (the 3.4 reference)
  - Curvature-rescaled Muon k=5 at the best vanilla LR
  - SGD at several LRs for context

KEY TESTS:
  T1: Is the best vanilla Muon LR near 0.002 (= 0.02 * 0.1)?
      If yes => rescaler is just LR reduction.
  T2: Does best-LR vanilla Muon match rescaled Muon within 5%?
      If yes => the 130x is an artifact of bad default LR.
  T3: Does rescaled Muon at the best vanilla LR FURTHER improve?
      If yes => rescaling has genuine value beyond LR tuning.
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
MOMENTUM = 0.9
GAMMA = 1.0
SCALE_MIN = 0.1
SCALE_MAX = 10.0
NUM_SEEDS = 10
DATA_POINTS = 32

# LR sweep values
VANILLA_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SGD_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ORIGINAL_LR = 0.02  # the 3.4 default


# =============================================================================
# NETWORK UTILITIES (identical to 3.4)
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
# NEWTON-SCHULZ ITERATION (identical to 3.4)
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=5):
    """Newton-Schulz iteration: Muon's quintic polynomial."""
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        XtX = X.T @ X
        X = a * X + b * (X @ XtX) + c * (X @ (XtX @ XtX))

    return X


# =============================================================================
# OPTIMIZERS
# =============================================================================

def train_muon(weights, X, Y_target, lr, num_steps, ns_iters=5,
               rescale_mode='none', gamma=1.0, scale_min=0.1, scale_max=10.0,
               momentum=0.9):
    """
    Muon optimizer with optional curvature rescaling.
    Returns (loss_history, scale_history, final_weights).
    """
    num_layers = len(weights)
    velocities = [np.zeros_like(W) for W in weights]
    losses = []
    scales_used = []

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        losses.append(loss)

        # Divergence guard
        if not np.isfinite(loss) or loss > 1e10:
            for remaining in range(num_steps - step):
                losses.append(float('inf'))
                scales_used.append(1.0)
            break

        grads = compute_gradients(weights, X, Y_target)

        step_scales = []
        for i in range(num_layers):
            G = grads[i]
            G_norm = np.linalg.norm(G, 'fro')

            G_orth = newton_schulz_orthogonalize(G, num_iters=ns_iters)
            G_orth_norm = np.linalg.norm(G_orth, 'fro')

            if rescale_mode == 'curvature':
                if G_orth_norm > 1e-12:
                    scale = np.clip(G_norm / G_orth_norm * gamma, scale_min, scale_max)
                else:
                    scale = 1.0
                G_orth = G_orth * scale
            else:
                scale = 1.0

            step_scales.append(scale)

            velocities[i] = momentum * velocities[i] + G_orth
            weights[i] = weights[i] - lr * velocities[i]

        scales_used.append(np.mean(step_scales))

    final_loss = compute_loss(weights, X, Y_target)
    losses.append(final_loss)

    return losses, scales_used, weights


def train_sgd(weights, X, Y_target, lr, num_steps, momentum=0.9):
    """SGD with momentum."""
    num_layers = len(weights)
    velocities = [np.zeros_like(W) for W in weights]
    losses = []

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y_target)
        losses.append(loss)

        if not np.isfinite(loss) or loss > 1e10:
            for remaining in range(num_steps - step):
                losses.append(float('inf'))
            break

        grads = compute_gradients(weights, X, Y_target)

        for i in range(num_layers):
            velocities[i] = momentum * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    losses.append(final_loss)

    return losses, weights


# =============================================================================
# DATA GENERATION (identical seed scheme to 3.4)
# =============================================================================

def make_problem(seed):
    """Generate target and data for a single seed."""
    rng = np.random.RandomState(seed)
    W_target = [rng.randn(DIM, DIM) * 0.3 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, DATA_POINTS) * 0.5
    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target
    return X, Y_target


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print()
    print("=" * 110)
    print("  Experiment H6: LR Artifact Check -- Is the 130x from curvature rescaling just LR reduction?")
    print("=" * 110)
    print()
    print(f"  Setup: {NUM_LAYERS}-layer {DIM}x{DIM} deep linear, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print(f"  Original 3.4 used lr={ORIGINAL_LR} with curvature rescale hitting min-clamp (0.1) 96.5% of time")
    print(f"  Hypothesis: effective LR was ~{ORIGINAL_LR * 0.1} => sweeping vanilla Muon LR to check")
    print()

    # =========================================================================
    # PHASE 1: Vanilla Muon LR sweep
    # =========================================================================
    print("-" * 110)
    print("  PHASE 1: Vanilla Muon LR sweep")
    print("-" * 110)

    vanilla_results = {}  # lr -> list of final losses across seeds
    for lr in VANILLA_LRS:
        final_losses = []
        for seed in seeds:
            X, Y_target = make_problem(seed)
            w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
            losses, _, _ = train_muon(w_init, X, Y_target, lr=lr, num_steps=NUM_STEPS,
                                       ns_iters=5, rescale_mode='none', momentum=MOMENTUM)
            final_losses.append(losses[-1])
        vanilla_results[lr] = final_losses
        mean_l = np.mean(final_losses)
        med_l = np.median(final_losses)
        finite_frac = np.mean(np.isfinite(final_losses)) * 100
        print(f"    lr={lr:<8.4f}  mean={mean_l:12.6e}  median={med_l:12.6e}  "
              f"finite={finite_frac:.0f}%")

    # Find best vanilla LR (by mean of finite losses)
    best_vanilla_lr = None
    best_vanilla_mean = float('inf')
    for lr in VANILLA_LRS:
        fl = np.array(vanilla_results[lr])
        finite_fl = fl[np.isfinite(fl)]
        if len(finite_fl) > 0:
            m = np.mean(finite_fl)
            if m < best_vanilla_mean:
                best_vanilla_mean = m
                best_vanilla_lr = lr

    print(f"\n    >>> Best vanilla Muon LR: {best_vanilla_lr} (mean loss = {best_vanilla_mean:.6e})")
    print(f"    >>> Expected if rescaler is pure LR reduction: ~{ORIGINAL_LR * 0.1}")
    print()

    # =========================================================================
    # PHASE 2: SGD LR sweep
    # =========================================================================
    print("-" * 110)
    print("  PHASE 2: SGD LR sweep")
    print("-" * 110)

    sgd_results = {}
    for lr in SGD_LRS:
        final_losses = []
        for seed in seeds:
            X, Y_target = make_problem(seed)
            w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
            losses, _ = train_sgd(w_init, X, Y_target, lr=lr, num_steps=NUM_STEPS,
                                   momentum=MOMENTUM)
            final_losses.append(losses[-1])
        sgd_results[lr] = final_losses
        mean_l = np.mean(final_losses)
        finite_frac = np.mean(np.isfinite(final_losses)) * 100
        print(f"    lr={lr:<8.4f}  mean={mean_l:12.6e}  finite={finite_frac:.0f}%")

    best_sgd_lr = None
    best_sgd_mean = float('inf')
    for lr in SGD_LRS:
        fl = np.array(sgd_results[lr])
        finite_fl = fl[np.isfinite(fl)]
        if len(finite_fl) > 0:
            m = np.mean(finite_fl)
            if m < best_sgd_mean:
                best_sgd_mean = m
                best_sgd_lr = lr

    print(f"\n    >>> Best SGD LR: {best_sgd_lr} (mean loss = {best_sgd_mean:.6e})")
    print()

    # =========================================================================
    # PHASE 3: Curvature-rescaled Muon at lr=0.02 (3.4 reference)
    # =========================================================================
    print("-" * 110)
    print("  PHASE 3: Curvature-rescaled Muon")
    print("-" * 110)

    # 3a: Rescaled at original LR (the 3.4 result)
    rescaled_orig_losses = []
    rescaled_orig_scales = []
    for seed in seeds:
        X, Y_target = make_problem(seed)
        w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
        losses, scales, _ = train_muon(w_init, X, Y_target, lr=ORIGINAL_LR,
                                        num_steps=NUM_STEPS, ns_iters=5,
                                        rescale_mode='curvature', gamma=GAMMA,
                                        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
                                        momentum=MOMENTUM)
        rescaled_orig_losses.append(losses[-1])
        rescaled_orig_scales.extend(scales)

    rescaled_orig_mean = np.mean(rescaled_orig_losses)
    scales_arr = np.array(rescaled_orig_scales)
    hit_min_pct = np.mean(scales_arr <= SCALE_MIN + 1e-8) * 100

    print(f"    Rescaled Muon at lr={ORIGINAL_LR}:  mean loss = {rescaled_orig_mean:.6e}")
    print(f"    Scale factor stats: mean={np.mean(scales_arr):.4f}, "
          f"median={np.median(scales_arr):.4f}, "
          f"min={np.min(scales_arr):.4f}, max={np.max(scales_arr):.4f}")
    print(f"    Fraction hitting min clamp ({SCALE_MIN}): {hit_min_pct:.1f}%")
    print()

    # 3b: Rescaled at best vanilla LR
    rescaled_best_losses = []
    rescaled_best_scales = []
    for seed in seeds:
        X, Y_target = make_problem(seed)
        w_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
        losses, scales, _ = train_muon(w_init, X, Y_target, lr=best_vanilla_lr,
                                        num_steps=NUM_STEPS, ns_iters=5,
                                        rescale_mode='curvature', gamma=GAMMA,
                                        scale_min=SCALE_MIN, scale_max=SCALE_MAX,
                                        momentum=MOMENTUM)
        rescaled_best_losses.append(losses[-1])
        rescaled_best_scales.extend(scales)

    rescaled_best_mean = np.mean(rescaled_best_losses)
    scales_best_arr = np.array(rescaled_best_scales)
    hit_min_best_pct = np.mean(scales_best_arr <= SCALE_MIN + 1e-8) * 100

    print(f"    Rescaled Muon at lr={best_vanilla_lr} (best vanilla LR):  mean loss = {rescaled_best_mean:.6e}")
    print(f"    Scale factor stats: mean={np.mean(scales_best_arr):.4f}, "
          f"median={np.median(scales_best_arr):.4f}, "
          f"min={np.min(scales_best_arr):.4f}, max={np.max(scales_best_arr):.4f}")
    print(f"    Fraction hitting min clamp ({SCALE_MIN}): {hit_min_best_pct:.1f}%")
    print()

    # =========================================================================
    # PHASE 4: COMPREHENSIVE TABLE
    # =========================================================================
    print("=" * 110)
    print("  COMPREHENSIVE RESULTS TABLE")
    print("=" * 110)
    print()

    # Collect all Muon results into table rows
    print(f"  {'LR':>8}  |  {'Vanilla Muon':>16}  {'(std)':>12}  |  "
          f"{'Rescaled Muon':>16}  {'(std)':>12}  |  {'SGD':>16}  {'(std)':>12}")
    print(f"  {'':->8}--+--{'':->16}--{'':->12}--+--"
          f"{'':->16}--{'':->12}--+--{'':->16}--{'':->12}")

    for lr in VANILLA_LRS:
        # Vanilla Muon
        vm = np.array(vanilla_results[lr])
        vm_finite = vm[np.isfinite(vm)]
        vm_str = f"{np.mean(vm_finite):16.6e}" if len(vm_finite) > 0 else f"{'DIVERGED':>16}"
        vm_std = f"{np.std(vm_finite):12.2e}" if len(vm_finite) > 0 else f"{'---':>12}"

        # Rescaled Muon (only computed for original and best)
        if abs(lr - ORIGINAL_LR) < 1e-10:
            rm = np.array(rescaled_orig_losses)
            rm_str = f"{np.mean(rm):16.6e}"
            rm_std = f"{np.std(rm):12.2e}"
        elif abs(lr - best_vanilla_lr) < 1e-10 and abs(best_vanilla_lr - ORIGINAL_LR) > 1e-10:
            rm = np.array(rescaled_best_losses)
            rm_str = f"{np.mean(rm):16.6e}"
            rm_std = f"{np.std(rm):12.2e}"
        else:
            rm_str = f"{'---':>16}"
            rm_std = f"{'---':>12}"

        # SGD
        if lr in sgd_results:
            sg = np.array(sgd_results[lr])
            sg_finite = sg[np.isfinite(sg)]
            sg_str = f"{np.mean(sg_finite):16.6e}" if len(sg_finite) > 0 else f"{'DIVERGED':>16}"
            sg_std = f"{np.std(sg_finite):12.2e}" if len(sg_finite) > 0 else f"{'---':>12}"
        else:
            sg_str = f"{'---':>16}"
            sg_std = f"{'---':>12}"

        marker = ""
        if abs(lr - ORIGINAL_LR) < 1e-10:
            marker = "  <-- 3.4 default"
        if abs(lr - best_vanilla_lr) < 1e-10:
            marker += "  <-- BEST vanilla"

        print(f"  {lr:>8.4f}  |  {vm_str}  {vm_std}  |  "
              f"{rm_str}  {rm_std}  |  {sg_str}  {sg_std}{marker}")

    print()

    # Also print the rescaled result at best vanilla LR if not already in table
    if best_vanilla_lr not in VANILLA_LRS:
        print(f"  NOTE: best_vanilla_lr={best_vanilla_lr} not in sweep -- this should not happen.")

    # =========================================================================
    # PHASE 5: KEY HYPOTHESIS TESTS
    # =========================================================================
    print("=" * 110)
    print("  KEY HYPOTHESIS TESTS")
    print("=" * 110)
    print()

    # Reference values
    vanilla_orig = np.array(vanilla_results[ORIGINAL_LR])
    vanilla_orig_mean = np.mean(vanilla_orig[np.isfinite(vanilla_orig)]) if np.any(np.isfinite(vanilla_orig)) else float('inf')
    vanilla_best = np.array(vanilla_results[best_vanilla_lr])
    vanilla_best_mean = np.mean(vanilla_best[np.isfinite(vanilla_best)])

    print(f"  Reference values:")
    print(f"    Vanilla Muon at lr={ORIGINAL_LR} (3.4 default):    mean = {vanilla_orig_mean:.6e}")
    print(f"    Vanilla Muon at lr={best_vanilla_lr} (best):         mean = {vanilla_best_mean:.6e}")
    print(f"    Rescaled Muon at lr={ORIGINAL_LR} (3.4 result):    mean = {rescaled_orig_mean:.6e}")
    print(f"    Rescaled Muon at lr={best_vanilla_lr} (best+resc):   mean = {rescaled_best_mean:.6e}")
    print(f"    Best SGD at lr={best_sgd_lr}:                       mean = {best_sgd_mean:.6e}")
    print()

    # Improvement ratios
    if rescaled_orig_mean > 1e-15:
        ratio_orig_vs_rescaled = vanilla_orig_mean / rescaled_orig_mean
        print(f"  Ratio: vanilla(0.02) / rescaled(0.02) = {ratio_orig_vs_rescaled:.1f}x")
        print(f"    (This is the '130x' improvement from 3.4)")
    print()

    if vanilla_best_mean > 1e-15:
        ratio_best_vs_orig = vanilla_orig_mean / vanilla_best_mean
        print(f"  Ratio: vanilla(0.02) / vanilla(best={best_vanilla_lr}) = {ratio_best_vs_orig:.1f}x")
        print(f"    (How much of the improvement is from LR alone)")
    print()

    # --- T1: Is the best vanilla LR near 0.002? ---
    print("  " + "-" * 106)
    expected_lr = ORIGINAL_LR * SCALE_MIN  # 0.02 * 0.1 = 0.002
    t1_ratio = best_vanilla_lr / expected_lr
    t1_pass = 0.5 <= t1_ratio <= 2.0  # within factor of 2
    t1_exact = abs(best_vanilla_lr - expected_lr) < 1e-10

    print(f"  T1: Is the best vanilla Muon LR near {expected_lr} (= {ORIGINAL_LR} x {SCALE_MIN})?")
    print(f"      Best vanilla LR = {best_vanilla_lr}")
    print(f"      Expected LR     = {expected_lr}")
    print(f"      Ratio best/expected = {t1_ratio:.2f}")
    if t1_exact:
        print(f"      RESULT: EXACT MATCH -- best vanilla LR IS exactly the clamped LR")
    elif t1_pass:
        print(f"      RESULT: APPROXIMATE MATCH -- within factor of 2")
    else:
        print(f"      RESULT: NO MATCH -- best vanilla LR is far from expected")

    if t1_pass:
        print(f"      INTERPRETATION: The rescaler IS primarily acting as LR reduction.")
    else:
        print(f"      INTERPRETATION: The rescaler does something beyond simple LR reduction.")
    print()

    # --- T2: Does best-LR vanilla Muon match rescaled Muon within 5%? ---
    print("  " + "-" * 106)
    if rescaled_orig_mean > 1e-15:
        t2_ratio = vanilla_best_mean / rescaled_orig_mean
        t2_pct_diff = abs(t2_ratio - 1.0) * 100
        t2_pass = t2_pct_diff < 5.0
    else:
        t2_ratio = float('inf')
        t2_pct_diff = float('inf')
        t2_pass = False

    print(f"  T2: Does best-LR vanilla Muon match rescaled Muon at lr=0.02 within 5%?")
    print(f"      Vanilla best (lr={best_vanilla_lr}):  {vanilla_best_mean:.6e}")
    print(f"      Rescaled (lr={ORIGINAL_LR}):          {rescaled_orig_mean:.6e}")
    print(f"      Ratio = {t2_ratio:.4f}  (diff = {t2_pct_diff:.1f}%)")
    if t2_pass:
        print(f"      RESULT: MATCH within 5% -- the 130x IS an artifact of bad default LR")
    else:
        if vanilla_best_mean < rescaled_orig_mean:
            print(f"      RESULT: NO MATCH -- vanilla at optimal LR is BETTER than rescaled.")
            print(f"      The rescaling provides NO value; proper LR tuning is sufficient.")
        else:
            print(f"      RESULT: NO MATCH -- rescaled Muon is still better than best vanilla.")
            print(f"      The rescaling provides genuine value beyond LR reduction.")
    print()

    # --- T3: Does rescaled Muon at best vanilla LR further improve? ---
    print("  " + "-" * 106)
    if vanilla_best_mean > 1e-15:
        t3_ratio = vanilla_best_mean / rescaled_best_mean
        t3_pct_improvement = (1.0 - rescaled_best_mean / vanilla_best_mean) * 100
    else:
        t3_ratio = float('nan')
        t3_pct_improvement = float('nan')
    t3_pass = rescaled_best_mean < vanilla_best_mean * 0.95  # >5% improvement

    print(f"  T3: Does rescaled Muon at the best vanilla LR further improve?")
    print(f"      Vanilla at lr={best_vanilla_lr}:            {vanilla_best_mean:.6e}")
    print(f"      Rescaled at lr={best_vanilla_lr}:           {rescaled_best_mean:.6e}")
    print(f"      Improvement from rescaling: {t3_pct_improvement:.1f}%  (ratio = {t3_ratio:.2f}x)")
    if t3_pass:
        print(f"      RESULT: YES -- rescaling provides >{5}% further improvement")
        print(f"      Curvature rescaling has genuine value beyond LR tuning.")
    else:
        if rescaled_best_mean > vanilla_best_mean:
            print(f"      RESULT: NO -- rescaling actually HURTS at optimal LR")
            print(f"      Curvature rescaling is purely an LR artifact.")
        else:
            print(f"      RESULT: MARGINAL -- rescaling helps <5%, essentially just LR effect")
    print()

    # =========================================================================
    # PER-SEED COMPARISON: best vanilla vs rescaled
    # =========================================================================
    print("=" * 110)
    print("  PER-SEED COMPARISON: Vanilla(best LR) vs Rescaled(original LR) vs Rescaled(best LR)")
    print("=" * 110)
    print()

    vanilla_best_arr = np.array(vanilla_results[best_vanilla_lr])
    rescaled_orig_arr = np.array(rescaled_orig_losses)
    rescaled_best_arr = np.array(rescaled_best_losses)

    print(f"  {'Seed':>6}  {'Vanilla(best)':>16}  {'Rescaled(0.02)':>16}  "
          f"{'Rescaled(best)':>16}  {'Winner':>20}")
    print(f"  {'':->6}  {'':->16}  {'':->16}  {'':->16}  {'':->20}")

    wins_vanilla = 0
    wins_resc_orig = 0
    wins_resc_best = 0
    for i in range(NUM_SEEDS):
        vb = vanilla_best_arr[i]
        ro = rescaled_orig_arr[i]
        rb = rescaled_best_arr[i]

        candidates = {'Vanilla(best)': vb, 'Resc(0.02)': ro, 'Resc(best)': rb}
        winner = min(candidates, key=lambda k: candidates[k] if np.isfinite(candidates[k]) else float('inf'))
        if winner == 'Vanilla(best)':
            wins_vanilla += 1
        elif winner == 'Resc(0.02)':
            wins_resc_orig += 1
        else:
            wins_resc_best += 1

        print(f"  {i+1:>6}  {vb:16.6e}  {ro:16.6e}  {rb:16.6e}  {winner:>20}")

    print()
    print(f"  Win counts: Vanilla(best)={wins_vanilla}, "
          f"Resc(0.02)={wins_resc_orig}, Resc(best)={wins_resc_best}")
    print()

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    print("=" * 110)
    print("  OVERALL VERDICT")
    print("=" * 110)
    print()

    # Determine the story
    if t1_pass and t2_pass and not t3_pass:
        print("  VERDICT: THE 130x IS AN LR ARTIFACT.")
        print()
        print("  Evidence:")
        print(f"    1. Best vanilla LR ({best_vanilla_lr}) matches predicted LR "
              f"({expected_lr}) from clamp analysis.")
        print(f"    2. Vanilla at optimal LR matches rescaled Muon within {t2_pct_diff:.1f}%.")
        print(f"    3. Adding rescaling at optimal LR provides only {t3_pct_improvement:.1f}% change.")
        print()
        print("  The curvature rescaler in 3.4 accidentally found a better LR by")
        print("  multiplying by 0.1 (its min clamp) 96.5% of the time. The 'curvature")
        print("  adaptation' is a 10x LR reduction wearing a lab coat.")
    elif t3_pass:
        print("  VERDICT: CURVATURE RESCALING HAS GENUINE VALUE BEYOND LR TUNING.")
        print()
        print("  Evidence:")
        print(f"    1. Best vanilla LR: {best_vanilla_lr} (predicted: {expected_lr})")
        print(f"    2. Vanilla at optimal LR: {vanilla_best_mean:.6e}")
        print(f"    3. Rescaled at optimal LR: {rescaled_best_mean:.6e} ({t3_pct_improvement:.1f}% better)")
        print()
        print("  Even after correcting the LR, rescaling provides further improvement.")
        partially = ""
        if t1_pass:
            partially = "PARTIALLY an LR artifact (most of the 130x is LR), but "
        print(f"  The 130x is {partially}rescaling adds genuine per-step adaptation.")
    else:
        print("  VERDICT: MIXED / COMPLEX RESULT.")
        print()
        print(f"    T1 (best LR near 0.002): {'PASS' if t1_pass else 'FAIL'} -- best LR = {best_vanilla_lr}")
        print(f"    T2 (vanilla matches rescaled): {'PASS' if t2_pass else 'FAIL'} -- diff = {t2_pct_diff:.1f}%")
        print(f"    T3 (rescaling adds value at best LR): {'PASS' if t3_pass else 'FAIL'} -- {t3_pct_improvement:.1f}%")
        print()
        if not t1_pass and not t2_pass:
            print("  The rescaler is NOT primarily acting as LR reduction. The mechanism")
            print("  appears to be something more subtle than a simple constant multiplier.")
        elif t1_pass and not t2_pass:
            print("  The LR is in the right ballpark but the match is not within 5%.")
            print("  The rescaler is mostly LR reduction but with some additional effect.")

    print()
    print("=" * 110)
    print("  EXPERIMENT COMPLETE")
    print("=" * 110)
    print()


if __name__ == '__main__':
    main()
