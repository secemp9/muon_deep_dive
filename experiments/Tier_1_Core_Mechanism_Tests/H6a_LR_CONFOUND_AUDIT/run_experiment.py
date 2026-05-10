#!/usr/bin/env python3
"""
H6a: LR CONFOUND AUDIT OF D-TEST — THE PAPER'S CREDIBILITY DEPENDS ON THIS
=============================================================================

CONTEXT:
  D-TEST claimed O(T * kappa^L) vs O(T) complexity separation with R^2=0.91.
  It used:
    - SGD LR tuned per depth as lr = 2/(lambda_max * L)
    - Muon LR fixed at 0.005

  H6 showed the 130x curvature rescaling was entirely an LR artifact.
  This experiment applies the same audit to D-TEST.

CONCERN:
  - SGD's optimal LR may change MORE with depth than the formula gives
  - Muon's optimal LR may also change with depth but was fixed
  - The depth exponent (1.10x per layer) could be a LR mismatch growing with depth

PROTOCOL:
  For each depth L in {2, 4, 8, 16}:
    For each optimizer (SGD, Muon):
      Sweep 7 LR candidates, 3 seeds each.
      Find best LR by median final loss.
    Compute advantage = best_SGD_loss / best_Muon_loss

  Fit log(advantage) vs L.
  Compare R^2 with D-TEST's R^2=0.91.

VERDICT:
  If R^2 drops below 0.5 or exponent flattens => D-TEST was an LR confound
  If R^2 stays >0.8 and exponent is similar => D-TEST survives the audit
"""

import numpy as np
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION — matches D-TEST setup
# =============================================================================

DIM = 32
DEPTHS = [2, 4, 8, 16]
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
BATCH_SIZE = 64
NUM_SEEDS = 3

# LR sweep ranges — extended to avoid hitting sweep boundaries
SGD_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
MUON_LRS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


# =============================================================================
# NETWORK AND TRAINING UTILITIES
# =============================================================================

def newton_schulz(M, n_iters=NS_ITERS):
    """Newton-Schulz iteration for orthogonal polar factor."""
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(dim, depth, seed):
    """Initialize near identity for stability (same as D-TEST)."""
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(depth)]


def make_data(dim, seed):
    """Generate target matrix and data (same as D-TEST: single random target)."""
    rng = np.random.RandomState(seed)
    W_target = rng.randn(dim, dim) * 0.5
    X = rng.randn(dim, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


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


def train(weights_init, X, Y, lr, optimizer, n_steps=NUM_STEPS):
    """Train and return (final_loss, loss_history)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    losses = []

    for step in range(n_steps):
        loss = compute_loss(weights, X, Y)
        losses.append(loss)
        if not np.isfinite(loss) or loss > 1e10:
            # Fill rest with inf
            losses.extend([float('inf')] * (n_steps - step))
            return float('inf'), losses

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if optimizer == 'muon':
                ortho_g = newton_schulz(grads[i])
                mom[i] = MOMENTUM * mom[i] + ortho_g
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]

    final_loss = compute_loss(weights, X, Y)
    losses.append(final_loss)
    return final_loss, losses


# =============================================================================
# D-TEST's original LR selection for SGD (for comparison)
# =============================================================================

def dtest_sgd_lr(depth, X, Y):
    """Replicate D-TEST's lr = 2/(lambda_max * L) formula."""
    rng_state = np.random.get_state()
    np.random.seed(42)
    test_weights = init_weights(DIM, depth, 42)
    np.random.set_state(rng_state)

    W_prod = np.eye(DIM)
    for W in test_weights:
        W_prod = W @ W_prod
    sv_prod = np.linalg.svd(W_prod, compute_uv=False)
    sv_X = np.linalg.svd(X, compute_uv=False)
    N = X.shape[1]
    lambda_max = (sv_prod[0] ** 2) * (sv_X[0] ** 2) / N
    lr = min(2.0 / (lambda_max * depth), 0.1)
    return lr


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    t_start = time.time()
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]
    total_runs = len(DEPTHS) * (len(SGD_LRS) + len(MUON_LRS)) * NUM_SEEDS
    run_count = 0

    print()
    print("=" * 110)
    print("  H6a: LR CONFOUND AUDIT OF D-TEST DEPTH SCALING")
    print("  THE PAPER'S CREDIBILITY DEPENDS ON THIS RESULT")
    print("=" * 110)
    print()
    print(f"  Setup: {DIM}x{DIM} deep linear, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print(f"  Depths: {DEPTHS}")
    print(f"  SGD LR sweep:  {SGD_LRS}")
    print(f"  Muon LR sweep: {MUON_LRS}")
    print(f"  Total training runs: {total_runs}")
    print()

    # =========================================================================
    # PHASE 1: Full LR sweep for both optimizers at every depth
    # =========================================================================

    # results[depth][optimizer][lr] = list of final losses across seeds
    results = {}

    for depth in DEPTHS:
        print(f"  --- Depth L={depth} ---")
        results[depth] = {'sgd': {}, 'muon': {}}

        # Generate data (fixed per seed, independent of depth for fair comparison)
        # But depth affects the problem structure, so we use a common data seed
        for opt_name, lr_list in [('sgd', SGD_LRS), ('muon', MUON_LRS)]:
            for lr in lr_list:
                seed_losses = []
                for s in seeds:
                    X, Y = make_data(DIM, s)
                    w_init = init_weights(DIM, depth, s + 5000)
                    final_loss, _ = train(w_init, X, Y, lr, opt_name)
                    seed_losses.append(final_loss)
                    run_count += 1

                results[depth][opt_name][lr] = seed_losses
                finite = [l for l in seed_losses if np.isfinite(l)]
                median_l = np.median(finite) if finite else float('inf')

            # Print progress
            best_lr_for_opt = None
            best_median = float('inf')
            for lr in lr_list:
                finite = [l for l in results[depth][opt_name][lr] if np.isfinite(l)]
                med = np.median(finite) if finite else float('inf')
                if med < best_median:
                    best_median = med
                    best_lr_for_opt = lr
            print(f"    {opt_name.upper():>5}: best_lr={best_lr_for_opt:.4f}  "
                  f"median_loss={best_median:.6e}  "
                  f"({run_count}/{total_runs} runs done)")

    elapsed = time.time() - t_start
    print(f"\n  All sweeps complete in {elapsed:.1f}s ({total_runs} runs)")

    # =========================================================================
    # PHASE 2: Extract best LR and loss for each (depth, optimizer)
    # =========================================================================

    print()
    print("=" * 110)
    print("  PHASE 2: BEST LR PER DEPTH AND OPTIMIZER")
    print("=" * 110)
    print()

    best = {}  # best[depth][optimizer] = {'lr': ..., 'median_loss': ..., 'mean_loss': ..., 'losses': [...]}

    for depth in DEPTHS:
        best[depth] = {}
        for opt_name, lr_list in [('sgd', SGD_LRS), ('muon', MUON_LRS)]:
            best_lr = None
            best_median = float('inf')
            for lr in lr_list:
                finite = [l for l in results[depth][opt_name][lr] if np.isfinite(l)]
                if finite:
                    med = np.median(finite)
                    if med < best_median:
                        best_median = med
                        best_lr = lr
            finite_best = [l for l in results[depth][opt_name][best_lr] if np.isfinite(l)]
            best[depth][opt_name] = {
                'lr': best_lr,
                'median_loss': best_median,
                'mean_loss': np.mean(finite_best) if finite_best else float('inf'),
                'losses': results[depth][opt_name][best_lr],
            }

    # Print the table
    print(f"  {'Depth':>5} | {'Best SGD LR':>12} {'SGD loss':>14} | "
          f"{'Best Muon LR':>12} {'Muon loss':>14} | "
          f"{'Advantage':>12} {'log(adv)':>10}")
    print(f"  {'':->5}-+-{'':->12}-{'':->14}-+-{'':->12}-{'':->14}-+-{'':->12}-{'':->10}")

    depth_arr = []
    log_advantage_arr = []
    advantage_arr = []

    for depth in DEPTHS:
        sgd_l = best[depth]['sgd']['median_loss']
        muon_l = best[depth]['muon']['median_loss']
        sgd_lr = best[depth]['sgd']['lr']
        muon_lr = best[depth]['muon']['lr']

        if muon_l > 1e-30 and np.isfinite(sgd_l) and np.isfinite(muon_l):
            advantage = sgd_l / muon_l
            log_adv = np.log(advantage)
            depth_arr.append(depth)
            log_advantage_arr.append(log_adv)
            advantage_arr.append(advantage)
        else:
            advantage = float('inf')
            log_adv = float('inf')

        print(f"  {depth:>5} | {sgd_lr:>12.4f} {sgd_l:>14.6e} | "
              f"{muon_lr:>12.4f} {muon_l:>14.6e} | "
              f"{advantage:>12.2f}x {log_adv:>10.4f}")

    # =========================================================================
    # PHASE 3: D-TEST COMPARISON — original fixed-LR results
    # =========================================================================

    print()
    print("=" * 110)
    print("  PHASE 3: D-TEST COMPARISON (ORIGINAL FIXED-LR PROTOCOL)")
    print("=" * 110)
    print()

    # Replicate D-TEST: fixed Muon LR=0.005, SGD LR from formula
    dtest_depth_arr = []
    dtest_log_adv_arr = []
    dtest_advantage_arr = []

    print(f"  {'Depth':>5} | {'D-TEST SGD LR':>14} {'SGD loss':>14} | "
          f"{'D-TEST Muon LR':>14} {'Muon loss':>14} | "
          f"{'Advantage':>12} {'log(adv)':>10}")
    print(f"  {'':->5}-+-{'':->14}-{'':->14}-+-{'':->14}-{'':->14}-+-{'':->12}-{'':->10}")

    DTEST_MUON_LR = 0.005

    for depth in DEPTHS:
        sgd_losses_dtest = []
        muon_losses_dtest = []
        for s in seeds:
            X, Y = make_data(DIM, s)
            w_init = init_weights(DIM, depth, s + 5000)
            sgd_lr_dtest = dtest_sgd_lr(depth, X, Y)

            # SGD run
            w_sgd = [W.copy() for W in w_init]
            fl_sgd, _ = train(w_sgd, X, Y, sgd_lr_dtest, 'sgd')
            sgd_losses_dtest.append(fl_sgd)

            # Muon run
            w_muon = [W.copy() for W in w_init]
            fl_muon, _ = train(w_muon, X, Y, DTEST_MUON_LR, 'muon')
            muon_losses_dtest.append(fl_muon)

        sgd_med = np.median([l for l in sgd_losses_dtest if np.isfinite(l)]) if any(np.isfinite(l) for l in sgd_losses_dtest) else float('inf')
        muon_med = np.median([l for l in muon_losses_dtest if np.isfinite(l)]) if any(np.isfinite(l) for l in muon_losses_dtest) else float('inf')

        if muon_med > 1e-30 and np.isfinite(sgd_med) and np.isfinite(muon_med):
            adv = sgd_med / muon_med
            log_adv = np.log(adv)
            dtest_depth_arr.append(depth)
            dtest_log_adv_arr.append(log_adv)
            dtest_advantage_arr.append(adv)
        else:
            adv = float('inf')
            log_adv = float('inf')

        sgd_lr_show = dtest_sgd_lr(depth, *make_data(DIM, seeds[0]))
        print(f"  {depth:>5} | {sgd_lr_show:>14.6f} {sgd_med:>14.6e} | "
              f"{DTEST_MUON_LR:>14.4f} {muon_med:>14.6e} | "
              f"{adv:>12.2f}x {log_adv:>10.4f}")

    # =========================================================================
    # PHASE 4: LINEAR FIT AND R^2
    # =========================================================================

    print()
    print("=" * 110)
    print("  PHASE 4: LINEAR FIT — log(advantage) vs L")
    print("=" * 110)
    print()

    def linear_fit(depths, log_advs, label):
        """Fit log(advantage) = a*L + b, return (slope, intercept, R^2)."""
        if len(depths) < 2:
            print(f"  {label}: INSUFFICIENT DATA (only {len(depths)} points)")
            return 0.0, 0.0, 0.0

        d = np.array(depths, dtype=float)
        y = np.array(log_advs, dtype=float)
        A = np.vstack([d, np.ones(len(d))]).T
        result = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = result[0]

        ss_res = np.sum((y - (slope * d + intercept)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0.0

        print(f"  {label}:")
        print(f"    Fit: log(advantage) = {slope:.4f} * L + ({intercept:.4f})")
        print(f"    Per-layer factor: e^slope = {np.exp(slope):.4f}x")
        print(f"    R^2 = {r2:.4f}")
        print()
        return slope, intercept, r2

    # Swept-LR fit (THIS IS THE AUDIT)
    slope_swept, intercept_swept, r2_swept = linear_fit(
        depth_arr, log_advantage_arr, "SWEPT-LR (per-depth best for BOTH optimizers)")

    # D-TEST-replica fit (for comparison)
    slope_dtest, intercept_dtest, r2_dtest = linear_fit(
        dtest_depth_arr, dtest_log_adv_arr, "D-TEST REPLICA (fixed Muon LR, formula SGD LR)")

    # =========================================================================
    # PHASE 5: LR SCALING WITH DEPTH
    # =========================================================================

    print()
    print("=" * 110)
    print("  PHASE 5: HOW DOES OPTIMAL LR SCALE WITH DEPTH?")
    print("=" * 110)
    print()

    print(f"  {'Depth':>5} | {'Best SGD LR':>12} | {'Best Muon LR':>12} | "
          f"{'D-TEST SGD LR':>14} | {'SGD swept/dtest':>16}")
    print(f"  {'':->5}-+-{'':->12}-+-{'':->12}-+-{'':->14}-+-{'':->16}")

    sgd_swept_lrs = []
    muon_swept_lrs = []
    dtest_sgd_lrs = []

    for depth in DEPTHS:
        sgd_lr_swept = best[depth]['sgd']['lr']
        muon_lr_swept = best[depth]['muon']['lr']
        sgd_lr_dt = dtest_sgd_lr(depth, *make_data(DIM, seeds[0]))

        sgd_swept_lrs.append(sgd_lr_swept)
        muon_swept_lrs.append(muon_lr_swept)
        dtest_sgd_lrs.append(sgd_lr_dt)

        ratio = sgd_lr_swept / sgd_lr_dt if sgd_lr_dt > 0 else float('nan')
        print(f"  {depth:>5} | {sgd_lr_swept:>12.4f} | {muon_lr_swept:>12.4f} | "
              f"{sgd_lr_dt:>14.6f} | {ratio:>16.2f}x")

    # Check if SGD LR decreases faster with depth than Muon LR
    print()
    if len(DEPTHS) >= 2:
        sgd_lr_ratio = sgd_swept_lrs[-1] / sgd_swept_lrs[0] if sgd_swept_lrs[0] > 0 else float('nan')
        muon_lr_ratio = muon_swept_lrs[-1] / muon_swept_lrs[0] if muon_swept_lrs[0] > 0 else float('nan')
        print(f"  SGD LR ratio (depth {DEPTHS[-1]} / depth {DEPTHS[0]}):  {sgd_lr_ratio:.4f}")
        print(f"  Muon LR ratio (depth {DEPTHS[-1]} / depth {DEPTHS[0]}): {muon_lr_ratio:.4f}")
        print()
        if np.isfinite(sgd_lr_ratio) and np.isfinite(muon_lr_ratio):
            if sgd_lr_ratio < muon_lr_ratio * 0.5:
                print("  WARNING: SGD's optimal LR decreases MUCH faster with depth than Muon's.")
                print("  This asymmetric scaling could EXPLAIN the depth-exponent as an LR confound.")
            elif sgd_lr_ratio < muon_lr_ratio:
                print("  NOTE: SGD's optimal LR decreases somewhat faster with depth than Muon's.")
                print("  This may partially confound the depth-exponent measurement.")
            else:
                print("  GOOD: SGD and Muon optimal LRs scale similarly with depth.")
                print("  The depth-exponent is NOT primarily an LR scaling artifact.")

    # =========================================================================
    # PHASE 6: DETAILED LR LANDSCAPE PER DEPTH
    # =========================================================================

    print()
    print("=" * 110)
    print("  PHASE 6: FULL LR LANDSCAPE (median final loss)")
    print("=" * 110)

    for depth in DEPTHS:
        print(f"\n  Depth L={depth}:")
        print(f"    SGD LRs:")
        for lr in SGD_LRS:
            finite = [l for l in results[depth]['sgd'][lr] if np.isfinite(l)]
            med = np.median(finite) if finite else float('inf')
            mean = np.mean(finite) if finite else float('inf')
            marker = " <-- BEST" if lr == best[depth]['sgd']['lr'] else ""
            frac = len(finite) / NUM_SEEDS * 100
            print(f"      lr={lr:.4f}  median={med:12.6e}  mean={mean:12.6e}  "
                  f"converged={frac:.0f}%{marker}")

        print(f"    Muon LRs:")
        for lr in MUON_LRS:
            finite = [l for l in results[depth]['muon'][lr] if np.isfinite(l)]
            med = np.median(finite) if finite else float('inf')
            mean = np.mean(finite) if finite else float('inf')
            marker = " <-- BEST" if lr == best[depth]['muon']['lr'] else ""
            frac = len(finite) / NUM_SEEDS * 100
            print(f"      lr={lr:.4f}  median={med:12.6e}  mean={mean:12.6e}  "
                  f"converged={frac:.0f}%{marker}")

    # =========================================================================
    # PHASE 7: RESIDUAL TABLE — advantage at EACH measurement point
    # =========================================================================

    print()
    print("=" * 110)
    print("  PHASE 7: ADVANTAGE AT MULTIPLE TRAINING STEPS (using best LRs)")
    print("=" * 110)
    print()

    MEASUREMENT_STEPS = [50, 100, 150, 200, 250, 300]

    print(f"  {'Depth':>5} |", end="")
    for ms in MEASUREMENT_STEPS:
        print(f"  Step {ms:>3}", end="")
    print()
    print(f"  {'':->5}-+", end="")
    for _ in MEASUREMENT_STEPS:
        print(f"{'':->10}", end="")
    print()

    for depth in DEPTHS:
        # Run SGD and Muon at best LRs, store full loss curves
        sgd_curves = []
        muon_curves = []
        for s in seeds:
            X, Y = make_data(DIM, s)
            w_init = init_weights(DIM, depth, s + 5000)

            w_sgd = [W.copy() for W in w_init]
            _, losses_sgd = train(w_sgd, X, Y, best[depth]['sgd']['lr'], 'sgd')
            sgd_curves.append(losses_sgd)

            w_muon = [W.copy() for W in w_init]
            _, losses_muon = train(w_muon, X, Y, best[depth]['muon']['lr'], 'muon')
            muon_curves.append(losses_muon)

        print(f"  {depth:>5} |", end="")
        for ms in MEASUREMENT_STEPS:
            sgd_at_step = [c[ms] if ms < len(c) else c[-1] for c in sgd_curves]
            muon_at_step = [c[ms] if ms < len(c) else c[-1] for c in muon_curves]
            sgd_med = np.median([l for l in sgd_at_step if np.isfinite(l)]) if any(np.isfinite(l) for l in sgd_at_step) else float('inf')
            muon_med = np.median([l for l in muon_at_step if np.isfinite(l)]) if any(np.isfinite(l) for l in muon_at_step) else float('inf')
            if muon_med > 1e-30 and np.isfinite(sgd_med):
                adv = sgd_med / muon_med
                print(f"  {adv:>7.2f}x", end="")
            else:
                print(f"  {'INF':>8}", end="")
        print()

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================

    print()
    print("=" * 110)
    print("  FINAL VERDICT: DOES D-TEST SURVIVE THE LR CONFOUND AUDIT?")
    print("=" * 110)
    print()

    print(f"  D-TEST ORIGINAL CLAIM:")
    print(f"    log(advantage) ~ {0.0953:.4f} * L, R^2 = 0.91")
    print(f"    Per-layer factor: 1.10x")
    print()

    print(f"  D-TEST REPLICA (this experiment, same protocol, 3 seeds):")
    print(f"    Slope = {slope_dtest:.4f}, R^2 = {r2_dtest:.4f}")
    print(f"    Per-layer factor: {np.exp(slope_dtest):.4f}x")
    print()

    print(f"  SWEPT-LR AUDIT (per-depth best LR for BOTH optimizers):")
    print(f"    Slope = {slope_swept:.4f}, R^2 = {r2_swept:.4f}")
    print(f"    Per-layer factor: {np.exp(slope_swept):.4f}x")
    print()

    # Key comparisons
    slope_change = abs(slope_swept - slope_dtest) / (abs(slope_dtest) + 1e-15)
    r2_change = r2_dtest - r2_swept

    print(f"  COMPARISONS:")
    print(f"    Slope change: {slope_dtest:.4f} -> {slope_swept:.4f} "
          f"({slope_change*100:.1f}% change)")
    print(f"    R^2 change:   {r2_dtest:.4f} -> {r2_swept:.4f} "
          f"(dropped by {r2_change:.4f})")
    print()

    # --- VERDICT ---
    print(f"  {'='*80}")

    if r2_swept > 0.8 and slope_swept > 0.03:
        print(f"  VERDICT: D-TEST SURVIVES THE AUDIT")
        print(f"  {'='*80}")
        print()
        print(f"  The exponential depth scaling PERSISTS even when both optimizers")
        print(f"  get their per-depth optimal learning rates.")
        print(f"    - R^2 = {r2_swept:.4f} (threshold: 0.8) -- PASS")
        print(f"    - Slope = {slope_swept:.4f} (per-layer factor {np.exp(slope_swept):.4f}x)")
        if slope_change < 0.5:
            print(f"    - Slope changed by only {slope_change*100:.1f}% -- robust")
        else:
            print(f"    - Slope changed by {slope_change*100:.1f}% -- the MAGNITUDE is reduced")
            print(f"      but the QUALITATIVE finding (exponential scaling) holds.")
        print()
        print(f"  The depth-exponent is a REAL algorithmic advantage, not an LR confound.")

    elif r2_swept > 0.5 and slope_swept > 0.01:
        print(f"  VERDICT: D-TEST PARTIALLY SURVIVES — WEAKENED BUT NOT DEAD")
        print(f"  {'='*80}")
        print()
        print(f"  The exponential trend exists but is weaker after LR correction.")
        print(f"    - R^2 = {r2_swept:.4f} (below 0.8 threshold but above 0.5)")
        print(f"    - Slope = {slope_swept:.4f} (per-layer factor {np.exp(slope_swept):.4f}x)")
        print()
        if slope_change > 0.5:
            print(f"  WARNING: The original D-TEST OVERSTATED the effect by {slope_change*100:.0f}%")
            print(f"  due to LR confounding. The paper should report the corrected values.")
        else:
            print(f"  The effect size is similar but noisier with proper LR tuning.")

    elif r2_swept < 0.5 or slope_swept < 0.01:
        print(f"  VERDICT: D-TEST DOES NOT SURVIVE — RETRACT THE CLAIM")
        print(f"  {'='*80}")
        print()
        print(f"  The exponential depth scaling DISAPPEARS when both optimizers")
        print(f"  get proper per-depth LR tuning.")
        print(f"    - R^2 dropped from {r2_dtest:.4f} to {r2_swept:.4f}")
        if slope_swept < 0.01:
            print(f"    - Slope dropped from {slope_dtest:.4f} to {slope_swept:.4f}")
            print(f"      The per-layer advantage factor is essentially 1.0x (no scaling)")
        print()
        print(f"  The D-TEST result was an LR CONFOUND. The apparent exponential scaling")
        print(f"  came from SGD being given increasingly suboptimal LRs at greater depths")
        print(f"  while Muon's fixed LR happened to be near-optimal across depths.")

    else:
        print(f"  VERDICT: INCONCLUSIVE")
        print(f"  {'='*80}")
        print()
        print(f"  Cannot determine if the depth scaling is real or artifactual.")
        print(f"  R^2 = {r2_swept:.4f}, slope = {slope_swept:.4f}")

    print()
    elapsed_total = time.time() - t_start
    print(f"  Total experiment time: {elapsed_total:.1f}s")
    print()
    print("=" * 110)
    print("  EXPERIMENT COMPLETE")
    print("=" * 110)
    print()


if __name__ == '__main__':
    main()
