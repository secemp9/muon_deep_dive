#!/usr/bin/env python3
"""
1.3c-i: Isolate lambda_max from alpha -- Track Both Independently
=================================================================

WeightWatcher alpha is the power-law exponent of the W^TW eigenvalue
distribution.  This experiment tracks lambda_max/lambda_median AND alpha
separately to determine whether Muon controls the bulk spectrum (slow
alpha drift) or just suppresses lambda_max.

HYPOTHESIS:
  - Muon controls the bulk spectrum: alpha stays large (flatter / more
    uniform eigenvalue distribution) throughout training.
  - SGD lets alpha -> 2 fast (heavy tail / concentration).
  - lambda_max grows faster under SGD (consistent with sigma_1 growth
    from 1.3b-i-A).
  - Clipping lambda_max changes the story for SGD but not Muon,
    because Muon already controls the outlier.

Power-law fit method (crude but sufficient for relative comparison):
  Sort eigenvalues descending.  Log-log linear regression of eigenvalue
  vs rank.  The negative slope is alpha.

Setup: 4-layer deep linear net, 32x32, quadratic loss, 500 steps.
       SGD (with momentum) vs Muon.
"""

import numpy as np
import os

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
BATCH_SIZE = 64
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
MEASURE_EVERY = 25

# Random target matrix (fixed)
W_target = np.random.randn(DIM, DIM) * 0.5

# Random input data (fixed batch)
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Steps to print in tables
TABLE_STEPS = [0, 100, 200, 300, 500]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def init_weights(num_layers, seed=42):
    """Initialize layers near identity for stability."""
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


def forward(weights, X):
    """Forward pass: W_L @ ... @ W_1 @ X."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, target):
    """Loss = 0.5 * ||W_product @ X - T @ X||^2 / N."""
    pred = forward(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients(weights, X, target):
    """Backprop through deep linear net."""
    num_layers = len(weights)
    N = X.shape[1]

    # Forward pass storing activations
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    # Backward pass
    target_out = target @ X
    delta = (activations[-1] - target_out) / N

    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta

    return grads


def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    """
    Newton-Schulz iteration to approximate the orthogonal polar factor.
    Returns closest orthogonal matrix to G (i.e., U @ V^T from SVD).
    """
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A

    return X


# =============================================================================
# POWER-LAW ALPHA FITTING
# =============================================================================

def fit_power_law_alpha(eigenvalues):
    """
    Fit power-law exponent alpha from sorted eigenvalues of W^TW.

    Method: Sort eigenvalues descending. Compute log-log linear regression
    of eigenvalue vs rank.  alpha = -slope (positive for decaying spectra).

    Returns alpha (float).
    """
    eigs = np.sort(eigenvalues)[::-1]  # descending
    eigs = eigs[eigs > 1e-30]  # remove near-zero
    n = len(eigs)
    if n < 3:
        return np.nan

    ranks = np.arange(1, n + 1).astype(float)
    log_rank = np.log(ranks)
    log_eig = np.log(eigs)

    # Linear regression: log_eig = slope * log_rank + intercept
    # alpha = -slope
    A = np.vstack([log_rank, np.ones(n)]).T
    result = np.linalg.lstsq(A, log_eig, rcond=None)
    slope = result[0][0]

    return -slope


def fit_power_law_alpha_clipped(eigenvalues):
    """
    Same as fit_power_law_alpha but EXCLUDING the top eigenvalue.
    This simulates WeightWatcher's clip_xmax behavior.
    """
    eigs = np.sort(eigenvalues)[::-1]  # descending
    eigs = eigs[1:]  # exclude top eigenvalue
    eigs = eigs[eigs > 1e-30]
    n = len(eigs)
    if n < 3:
        return np.nan

    ranks = np.arange(1, n + 1).astype(float)
    log_rank = np.log(ranks)
    log_eig = np.log(eigs)

    A = np.vstack([log_rank, np.ones(n)]).T
    result = np.linalg.lstsq(A, log_eig, rcond=None)
    slope = result[0][0]

    return -slope


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================

def find_stable_lr_sgd():
    """Find maximum stable SGD learning rate."""
    candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
    for lr in candidates:
        np.random.seed(42)
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss(weights, X_data, W_target)
        stable = True
        for step in range(200):
            grads = compute_gradients(weights, X_data, W_target)
            for i in range(NUM_LAYERS):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]
            loss = compute_loss(weights, X_data, W_target)
            if np.isnan(loss) or loss > initial_loss * 50:
                stable = False
                break
        if stable:
            return lr
    return 0.001


def sgd_step(weights, velocities, lr):
    """One step of SGD with momentum."""
    grads = compute_gradients(weights, X_data, W_target)
    for i in range(len(weights)):
        velocities[i] = MOMENTUM * velocities[i] + grads[i]
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


def muon_step(weights, velocities, lr):
    """One step of Muon with momentum."""
    grads = compute_gradients(weights, X_data, W_target)
    for i in range(len(weights)):
        ortho_grad = newton_schulz_orthogonalize(grads[i])
        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
        weights[i] = weights[i] - lr * velocities[i]
    return weights, velocities


# =============================================================================
# MEASUREMENT ENGINE
# =============================================================================

def compute_layer_spectrum_stats(W):
    """
    Given weight matrix W, compute eigenvalues of W^TW and return stats.

    Returns dict with:
      - eigenvalues: full sorted (descending) eigenvalues of W^TW
      - lambda_max, lambda_median, lambda_min
      - alpha: power-law exponent (full)
      - alpha_clipped: power-law exponent (excluding top eigenvalue)
      - outlier_ratio: lambda_max / lambda_median
    """
    WtW = W.T @ W
    eigs = np.linalg.eigvalsh(WtW)  # returns sorted ascending
    eigs = eigs[::-1]  # descending
    eigs = np.maximum(eigs, 0.0)  # ensure non-negative (numerical)

    lmax = eigs[0]
    lmin = eigs[-1]
    lmedian = np.median(eigs)
    alpha = fit_power_law_alpha(eigs)
    alpha_clipped = fit_power_law_alpha_clipped(eigs)
    outlier_ratio = lmax / lmedian if lmedian > 1e-30 else np.inf

    return {
        'eigenvalues': eigs,
        'lambda_max': lmax,
        'lambda_median': lmedian,
        'lambda_min': lmin,
        'alpha': alpha,
        'alpha_clipped': alpha_clipped,
        'outlier_ratio': outlier_ratio,
    }


def run_and_measure(optimizer_name, optimizer_fn, lr, num_steps):
    """
    Run optimizer for num_steps.  At every MEASURE_EVERY steps, record:
      - alpha (per layer and mean)
      - alpha_clipped (per layer and mean)
      - lambda_max, lambda_median, lambda_min (per layer)
      - outlier_ratio = lambda_max / lambda_median (per layer and mean)
      - loss
    """
    np.random.seed(42)
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    # Determine measurement steps
    measure_steps = list(range(0, num_steps + 1, MEASURE_EVERY))
    if num_steps not in measure_steps:
        measure_steps.append(num_steps)
    measure_steps = sorted(set(measure_steps))

    n_measures = len(measure_steps)

    # Storage
    alpha_all = np.zeros((n_measures, NUM_LAYERS))
    alpha_clipped_all = np.zeros((n_measures, NUM_LAYERS))
    lmax_all = np.zeros((n_measures, NUM_LAYERS))
    lmedian_all = np.zeros((n_measures, NUM_LAYERS))
    lmin_all = np.zeros((n_measures, NUM_LAYERS))
    outlier_ratio_all = np.zeros((n_measures, NUM_LAYERS))
    losses = np.zeros(n_measures)

    measure_idx = 0
    diverged = False

    # Measure at step 0
    if measure_steps[0] == 0:
        for i in range(NUM_LAYERS):
            stats = compute_layer_spectrum_stats(weights[i])
            alpha_all[0, i] = stats['alpha']
            alpha_clipped_all[0, i] = stats['alpha_clipped']
            lmax_all[0, i] = stats['lambda_max']
            lmedian_all[0, i] = stats['lambda_median']
            lmin_all[0, i] = stats['lambda_min']
            outlier_ratio_all[0, i] = stats['outlier_ratio']
        losses[0] = compute_loss(weights, X_data, W_target)
        measure_idx = 1

    for step in range(1, num_steps + 1):
        weights, velocities = optimizer_fn(weights, velocities, lr)

        loss = compute_loss(weights, X_data, W_target)
        if np.isnan(loss) or loss > 1e10:
            print(f"    WARNING: {optimizer_name} diverged at step {step}!")
            # Fill remaining with NaN
            for mi in range(measure_idx, n_measures):
                alpha_all[mi] = np.nan
                alpha_clipped_all[mi] = np.nan
                lmax_all[mi] = np.nan
                lmedian_all[mi] = np.nan
                lmin_all[mi] = np.nan
                outlier_ratio_all[mi] = np.nan
                losses[mi] = np.nan
            diverged = True
            break

        if measure_idx < n_measures and step == measure_steps[measure_idx]:
            for i in range(NUM_LAYERS):
                stats = compute_layer_spectrum_stats(weights[i])
                alpha_all[measure_idx, i] = stats['alpha']
                alpha_clipped_all[measure_idx, i] = stats['alpha_clipped']
                lmax_all[measure_idx, i] = stats['lambda_max']
                lmedian_all[measure_idx, i] = stats['lambda_median']
                lmin_all[measure_idx, i] = stats['lambda_min']
                outlier_ratio_all[measure_idx, i] = stats['outlier_ratio']
            losses[measure_idx] = loss
            measure_idx += 1

    return {
        'measure_steps': np.array(measure_steps),
        'alpha': alpha_all,              # (n_measures, NUM_LAYERS)
        'alpha_clipped': alpha_clipped_all,
        'lambda_max': lmax_all,
        'lambda_median': lmedian_all,
        'lambda_min': lmin_all,
        'outlier_ratio': outlier_ratio_all,
        'losses': losses,
        'diverged': diverged,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 100)
print("1.3c-i: ISOLATE lambda_max FROM alpha -- TRACK BOTH INDEPENDENTLY")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer deep linear net (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"Track: alpha (power-law exponent), alpha_clipped (excluding top eigenvalue),")
print(f"       lambda_max/lambda_median (outlier ratio)")
print(f"Measure every {MEASURE_EVERY} steps")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}")
print("=" * 100)

# Find stable SGD learning rate
lr_sgd = find_stable_lr_sgd()
print(f"\nSGD learning rate (max stable): {lr_sgd}")
print(f"Muon learning rate (fixed):     {LR_MUON}")

# Initial loss
np.random.seed(42)
w_test = init_weights(NUM_LAYERS)
loss_init = compute_loss(w_test, X_data, W_target)
print(f"Initial loss: {loss_init:.6e}")

# Run both optimizers
print(f"\n{'=' * 100}")
print("RUNNING OPTIMIZERS")
print("=" * 100)

print("\n  Running SGD...", flush=True)
results_sgd = run_and_measure('SGD', sgd_step, lr_sgd, NUM_STEPS)
print(f"    SGD final loss: {results_sgd['losses'][-1]:.6e}")

print("\n  Running Muon...", flush=True)
results_muon = run_and_measure('Muon', muon_step, LR_MUON, NUM_STEPS)
print(f"    Muon final loss: {results_muon['losses'][-1]:.6e}")

# Get step arrays
steps_sgd = results_sgd['measure_steps']
steps_muon = results_muon['measure_steps']


# =============================================================================
# HELPER: Find index of a step in the measure_steps array
# =============================================================================

def step_idx(measure_steps, step_val):
    """Return the index of step_val in measure_steps, or None."""
    idx = np.where(measure_steps == step_val)[0]
    return idx[0] if len(idx) > 0 else None


# =============================================================================
# TABLE 1: alpha(t) -- power-law exponent evolution
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 1: POWER-LAW EXPONENT alpha(t) -- MEAN ACROSS LAYERS")
print("  alpha = negative slope of log(eigenvalue) vs log(rank)")
print("  Higher alpha => steeper decay => MORE heavy-tailed / concentrated")
print("  Lower alpha => flatter spectrum => more uniform eigenvalue distribution")
print("=" * 100)

print(f"\n  {'Step':>6} | {'SGD alpha':>10} | {'Muon alpha':>11} | {'SGD-Muon':>10} | {'SGD alpha_c':>12} | {'Muon alpha_c':>13} | {'SGD-Muon clip':>14}")
print("  " + "-" * 95)

for ts in TABLE_STEPS:
    idx_s = step_idx(steps_sgd, ts)
    idx_m = step_idx(steps_muon, ts)
    if idx_s is not None and idx_m is not None:
        a_sgd = np.nanmean(results_sgd['alpha'][idx_s])
        a_muon = np.nanmean(results_muon['alpha'][idx_m])
        ac_sgd = np.nanmean(results_sgd['alpha_clipped'][idx_s])
        ac_muon = np.nanmean(results_muon['alpha_clipped'][idx_m])
        print(f"  {ts:6d} | {a_sgd:10.4f} | {a_muon:11.4f} | {a_sgd - a_muon:+10.4f} | "
              f"{ac_sgd:12.4f} | {ac_muon:13.4f} | {ac_sgd - ac_muon:+14.4f}")


# =============================================================================
# TABLE 2: lambda_max / lambda_median (outlier ratio)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 2: OUTLIER RATIO lambda_max / lambda_median -- MEAN ACROSS LAYERS")
print("  Higher ratio => top eigenvalue is more of an outlier vs bulk")
print("=" * 100)

print(f"\n  {'Step':>6} | {'SGD ratio':>10} | {'Muon ratio':>11} | {'SGD/Muon':>10}")
print("  " + "-" * 52)

for ts in TABLE_STEPS:
    idx_s = step_idx(steps_sgd, ts)
    idx_m = step_idx(steps_muon, ts)
    if idx_s is not None and idx_m is not None:
        r_sgd = np.nanmean(results_sgd['outlier_ratio'][idx_s])
        r_muon = np.nanmean(results_muon['outlier_ratio'][idx_m])
        ratio_str = f"{r_sgd / r_muon:.4f}" if r_muon > 1e-10 else "N/A"
        print(f"  {ts:6d} | {r_sgd:10.4f} | {r_muon:11.4f} | {ratio_str:>10}")


# =============================================================================
# TABLE 3: Per-layer alpha at key steps
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 3: PER-LAYER alpha AT KEY STEPS")
print("=" * 100)

print(f"\n  {'Step':>6} | ", end="")
for layer in range(NUM_LAYERS):
    print(f"{'SGD L' + str(layer):>8} {'Muon L' + str(layer):>8} | ", end="")
print()
print("  " + "-" * (8 + (18 + 3) * NUM_LAYERS))

for ts in TABLE_STEPS:
    idx_s = step_idx(steps_sgd, ts)
    idx_m = step_idx(steps_muon, ts)
    if idx_s is not None and idx_m is not None:
        print(f"  {ts:6d} | ", end="")
        for layer in range(NUM_LAYERS):
            a_sgd = results_sgd['alpha'][idx_s, layer]
            a_muon = results_muon['alpha'][idx_m, layer]
            print(f"{a_sgd:8.4f} {a_muon:8.4f} | ", end="")
        print()


# =============================================================================
# TABLE 4: Per-layer lambda_max, lambda_median, lambda_min at key steps
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 4: EIGENVALUE SUMMARY (LAYER MEANS) AT KEY STEPS")
print("=" * 100)

print(f"\n  {'Step':>6} | {'SGD lmax':>10} {'SGD lmed':>10} {'SGD lmin':>10} | "
      f"{'Muon lmax':>10} {'Muon lmed':>10} {'Muon lmin':>10}")
print("  " + "-" * 85)

for ts in TABLE_STEPS:
    idx_s = step_idx(steps_sgd, ts)
    idx_m = step_idx(steps_muon, ts)
    if idx_s is not None and idx_m is not None:
        lmax_s = np.nanmean(results_sgd['lambda_max'][idx_s])
        lmed_s = np.nanmean(results_sgd['lambda_median'][idx_s])
        lmin_s = np.nanmean(results_sgd['lambda_min'][idx_s])
        lmax_m = np.nanmean(results_muon['lambda_max'][idx_m])
        lmed_m = np.nanmean(results_muon['lambda_median'][idx_m])
        lmin_m = np.nanmean(results_muon['lambda_min'][idx_m])
        print(f"  {ts:6d} | {lmax_s:10.4f} {lmed_s:10.4f} {lmin_s:10.4f} | "
              f"{lmax_m:10.4f} {lmed_m:10.4f} {lmin_m:10.4f}")


# =============================================================================
# TABLE 5: Effect of clipping -- does removing lambda_max change alpha?
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 5: CLIPPING EFFECT -- |alpha - alpha_clipped| (MEAN ACROSS LAYERS)")
print("  Large difference => lambda_max is an outlier distorting the fit")
print("  Small difference => lambda_max is consistent with bulk spectrum")
print("=" * 100)

print(f"\n  {'Step':>6} | {'SGD |da|':>10} | {'Muon |da|':>11} | {'SGD-Muon':>10}")
print("  " + "-" * 52)

for ts in TABLE_STEPS:
    idx_s = step_idx(steps_sgd, ts)
    idx_m = step_idx(steps_muon, ts)
    if idx_s is not None and idx_m is not None:
        da_sgd = np.nanmean(np.abs(results_sgd['alpha'][idx_s] - results_sgd['alpha_clipped'][idx_s]))
        da_muon = np.nanmean(np.abs(results_muon['alpha'][idx_m] - results_muon['alpha_clipped'][idx_m]))
        print(f"  {ts:6d} | {da_sgd:10.4f} | {da_muon:11.4f} | {da_sgd - da_muon:+10.4f}")


# =============================================================================
# ANALYSIS: Key Tests
# =============================================================================

print(f"\n\n{'=' * 100}")
print("KEY TESTS")
print("=" * 100)

# Compute mean alpha trajectories
alpha_mean_sgd = np.nanmean(results_sgd['alpha'], axis=1)    # (n_measures,)
alpha_mean_muon = np.nanmean(results_muon['alpha'], axis=1)

alpha_c_mean_sgd = np.nanmean(results_sgd['alpha_clipped'], axis=1)
alpha_c_mean_muon = np.nanmean(results_muon['alpha_clipped'], axis=1)

outlier_mean_sgd = np.nanmean(results_sgd['outlier_ratio'], axis=1)
outlier_mean_muon = np.nanmean(results_muon['outlier_ratio'], axis=1)

# Use final step for tests
final_idx_s = -1
final_idx_m = -1

# T1: Does alpha evolve differently for Muon vs SGD?
# Compute alpha drift: |alpha(final) - alpha(0)|
alpha_drift_sgd = abs(alpha_mean_sgd[final_idx_s] - alpha_mean_sgd[0])
alpha_drift_muon = abs(alpha_mean_muon[final_idx_m] - alpha_mean_muon[0])

# Also: compute total alpha change (signed)
alpha_change_sgd = alpha_mean_sgd[final_idx_s] - alpha_mean_sgd[0]
alpha_change_muon = alpha_mean_muon[final_idx_m] - alpha_mean_muon[0]

# T2: Does Muon keep a flatter spectrum (lower alpha = less heavy-tailed)?
# NOTE: higher alpha = steeper log-log slope = MORE concentrated
# Flatter spectrum = smaller alpha (eigenvalues more uniform)
alpha_final_sgd = alpha_mean_sgd[final_idx_s]
alpha_final_muon = alpha_mean_muon[final_idx_m]

# T3: Does lambda_max grow faster for SGD?
lmax_growth_sgd = np.nanmean(results_sgd['lambda_max'][final_idx_s]) / np.nanmean(results_sgd['lambda_max'][0])
lmax_growth_muon = np.nanmean(results_muon['lambda_max'][final_idx_m]) / np.nanmean(results_muon['lambda_max'][0])

# T4: Does clipping lambda_max change the story for SGD but not Muon?
clip_effect_sgd = np.nanmean(np.abs(results_sgd['alpha'][final_idx_s] - results_sgd['alpha_clipped'][final_idx_s]))
clip_effect_muon = np.nanmean(np.abs(results_muon['alpha'][final_idx_m] - results_muon['alpha_clipped'][final_idx_m]))

# Print test results
test1_sgd_evolves_more = alpha_drift_sgd > alpha_drift_muon
test2_muon_flatter = alpha_final_muon < alpha_final_sgd
test3_sgd_lmax_faster = lmax_growth_sgd > lmax_growth_muon
test4_clip_sgd_more = clip_effect_sgd > clip_effect_muon

print(f"""
  T1: ALPHA EVOLUTION DIFFERS BETWEEN OPTIMIZERS
      SGD  alpha drift: |alpha(500) - alpha(0)| = |{alpha_mean_sgd[final_idx_s]:.4f} - {alpha_mean_sgd[0]:.4f}| = {alpha_drift_sgd:.4f}
      Muon alpha drift: |alpha(500) - alpha(0)| = |{alpha_mean_muon[final_idx_m]:.4f} - {alpha_mean_muon[0]:.4f}| = {alpha_drift_muon:.4f}
      SGD  alpha change (signed): {alpha_change_sgd:+.4f}
      Muon alpha change (signed): {alpha_change_muon:+.4f}
      SGD drifts more: {alpha_drift_sgd:.4f} vs {alpha_drift_muon:.4f}
      -> {"CONFIRMED" if test1_sgd_evolves_more else "REJECTED"}: SGD alpha drifts {'more' if test1_sgd_evolves_more else 'less'} than Muon

  T2: MUON KEEPS FLATTER SPECTRUM (LOWER alpha = LESS HEAVY-TAILED)
      SGD  final alpha: {alpha_final_sgd:.4f}
      Muon final alpha: {alpha_final_muon:.4f}
      -> {"CONFIRMED" if test2_muon_flatter else "REJECTED"}: Muon spectrum is {'flatter' if test2_muon_flatter else 'steeper'} than SGD

  T3: lambda_max GROWS FASTER FOR SGD
      SGD  lambda_max growth factor (final/init): {lmax_growth_sgd:.4f}
      Muon lambda_max growth factor (final/init): {lmax_growth_muon:.4f}
      -> {"CONFIRMED" if test3_sgd_lmax_faster else "REJECTED"}: SGD lambda_max grows {'faster' if test3_sgd_lmax_faster else 'slower'} than Muon

  T4: CLIPPING lambda_max CHANGES STORY FOR SGD BUT NOT MUON
      SGD  |alpha - alpha_clipped| at final step:  {clip_effect_sgd:.4f}
      Muon |alpha - alpha_clipped| at final step:  {clip_effect_muon:.4f}
      -> {"CONFIRMED" if test4_clip_sgd_more else "REJECTED"}: Clipping lambda_max changes alpha {'more for SGD' if test4_clip_sgd_more else 'more for Muon'}
""")

tests = [test1_sgd_evolves_more, test2_muon_flatter, test3_sgd_lmax_faster, test4_clip_sgd_more]
tests_passed = sum(tests)
tests_total = len(tests)


# =============================================================================
# PLOT: alpha vs step and lambda_max/lambda_median vs step
# =============================================================================

print(f"\n{'=' * 100}")
print("GENERATING PLOTS")
print("=" * 100)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('1.3c-i: Isolate lambda_max from alpha -- Track Both Independently\n'
                 f'{NUM_LAYERS}-layer linear net, dim={DIM}, {NUM_STEPS} steps',
                 fontsize=14, fontweight='bold')

    # --- Panel (a): alpha vs step (mean across layers) ---
    ax = axes[0, 0]
    ax.set_title('(a) Power-Law alpha vs Step (mean across layers)')
    ax.plot(steps_sgd, alpha_mean_sgd, 'b-o', linewidth=2.5, markersize=3, label='SGD')
    ax.plot(steps_muon, alpha_mean_muon, 'r--s', linewidth=2.5, markersize=3, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('alpha (power-law exponent)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): alpha_clipped vs step (mean across layers) ---
    ax = axes[0, 1]
    ax.set_title('(b) alpha_clipped (excl. top eigenvalue) vs Step')
    ax.plot(steps_sgd, alpha_c_mean_sgd, 'b-o', linewidth=2.5, markersize=3, label='SGD (clipped)')
    ax.plot(steps_muon, alpha_c_mean_muon, 'r--s', linewidth=2.5, markersize=3, label='Muon (clipped)')
    # Also plot unclipped for reference (thin lines)
    ax.plot(steps_sgd, alpha_mean_sgd, 'b:', linewidth=1, alpha=0.5, label='SGD (full)')
    ax.plot(steps_muon, alpha_mean_muon, 'r:', linewidth=1, alpha=0.5, label='Muon (full)')
    ax.set_xlabel('Step')
    ax.set_ylabel('alpha_clipped')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (c): lambda_max / lambda_median (outlier ratio) ---
    ax = axes[0, 2]
    ax.set_title('(c) Outlier Ratio lambda_max / lambda_median vs Step')
    ax.plot(steps_sgd, outlier_mean_sgd, 'b-o', linewidth=2.5, markersize=3, label='SGD')
    ax.plot(steps_muon, outlier_mean_muon, 'r--s', linewidth=2.5, markersize=3, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('lambda_max / lambda_median')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel (d): lambda_max trajectory ---
    ax = axes[1, 0]
    ax.set_title('(d) lambda_max vs Step (mean across layers)')
    ax.plot(steps_sgd, np.nanmean(results_sgd['lambda_max'], axis=1),
            'b-o', linewidth=2.5, markersize=3, label='SGD')
    ax.plot(steps_muon, np.nanmean(results_muon['lambda_max'], axis=1),
            'r--s', linewidth=2.5, markersize=3, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('lambda_max (top eigenvalue of W^TW)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel (e): Clipping effect over time ---
    ax = axes[1, 1]
    ax.set_title('(e) Clipping Effect |alpha - alpha_clipped| vs Step')
    clip_diff_sgd = np.nanmean(np.abs(results_sgd['alpha'] - results_sgd['alpha_clipped']), axis=1)
    clip_diff_muon = np.nanmean(np.abs(results_muon['alpha'] - results_muon['alpha_clipped']), axis=1)
    ax.plot(steps_sgd, clip_diff_sgd, 'b-o', linewidth=2.5, markersize=3, label='SGD')
    ax.plot(steps_muon, clip_diff_muon, 'r--s', linewidth=2.5, markersize=3, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('|alpha - alpha_clipped|')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel (f): Loss ---
    ax = axes[1, 2]
    ax.set_title('(f) Loss vs Step')
    ax.semilogy(steps_sgd, results_sgd['losses'], 'b-o', linewidth=2.5, markersize=3, label='SGD')
    ax.semilogy(steps_muon, results_muon['losses'], 'r--s', linewidth=2.5, markersize=3, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'alpha_vs_lambda_max.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {plot_path}")

except ImportError:
    print("\n  WARNING: matplotlib not available, skipping plots.")
    plot_path = None


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL VERDICT: ISOLATE lambda_max FROM alpha")
print("=" * 100)

if tests_passed >= 3:
    overall = "PASS"
    detail = (
        f"  {tests_passed}/{tests_total} tests pass.\n"
        "  Muon controls the BULK spectrum (alpha stays flatter) AND suppresses\n"
        "  lambda_max growth.  Clipping lambda_max has a larger effect on SGD,\n"
        "  indicating SGD's heavy tail is driven by outlier eigenvalues while\n"
        "  Muon distributes spectral energy more uniformly."
    )
elif tests_passed >= 2:
    overall = "PARTIAL PASS"
    detail = (
        f"  {tests_passed}/{tests_total} tests pass.\n"
        f"  T1 (alpha evolves differently): {'PASS' if test1_sgd_evolves_more else 'FAIL'}\n"
        f"  T2 (Muon flatter spectrum):     {'PASS' if test2_muon_flatter else 'FAIL'}\n"
        f"  T3 (SGD lmax faster):           {'PASS' if test3_sgd_lmax_faster else 'FAIL'}\n"
        f"  T4 (clipping helps SGD more):   {'PASS' if test4_clip_sgd_more else 'FAIL'}"
    )
else:
    overall = "FAIL"
    detail = (
        f"  Only {tests_passed}/{tests_total} tests pass.\n"
        f"  T1 (alpha evolves differently): {'PASS' if test1_sgd_evolves_more else 'FAIL'}\n"
        f"  T2 (Muon flatter spectrum):     {'PASS' if test2_muon_flatter else 'FAIL'}\n"
        f"  T3 (SGD lmax faster):           {'PASS' if test3_sgd_lmax_faster else 'FAIL'}\n"
        f"  T4 (clipping helps SGD more):   {'PASS' if test4_clip_sgd_more else 'FAIL'}"
    )

print(f"""
  +========================================================================+
  |  VERDICT: {overall:<63}|
  +========================================================================+
  |                                                                        |""")
for line in detail.split('\n'):
    print(f"  |  {line:<70}|")
print(f"""  |                                                                        |
  +========================================================================+
""")

print("=" * 100)
print(f"  Tests passed: {tests_passed}/{tests_total}")
print(f"  Overall: {overall}")
print("=" * 100)
