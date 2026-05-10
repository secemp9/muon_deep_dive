#!/usr/bin/env python3
"""
1.3a-i: Effective Rank -- Layer-by-Layer vs Aggregate
=====================================================
PREDICTION (from RG gauge-fixing model):
  Muon orthogonalizes gradients per layer, which should maintain high effective
  rank (near n=32) for each individual layer weight matrix. However, the product
  matrix W_product = W_6 @ ... @ W_1 may still see rank concentration because
  the PRODUCT of near-orthogonal matrices can still have spectrum that narrows
  over depth.

  SGD has no per-layer spectral control, so both per-layer AND product effective
  rank may degrade over training.

  HYPOTHESIS:
    - Muon maintains per-layer effective rank near n=32 (full rank)
    - SGD per-layer effective rank degrades over training
    - Product effective rank may still concentrate for BOTH optimizers
      (the gauge is per-layer, not per-product)

  CRITICAL CONTEXT:
    - 1.1a-i: product kappa is O(L) for Muon vs O(c^L) for SGD
    - 1.2b-i: Muon is MORE chaotic in weight space (higher Lyapunov)
    - 1.2b-ii: Muon is more orientation-biased (Q/(Q+P) = 0.60 vs 0.55)

  Effective rank = exp(H) where H = -sum(p_i * log(p_i)),
  p_i = sigma_i / sum(sigma_j), and sigma_i are singular values.
  Maximum effective rank = n (all singular values equal).
  Minimum effective rank = 1 (rank-1 matrix).

Setup: 6-layer deep linear net, 32x32, quadratic loss, 300 steps.
"""

import numpy as np
import os

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 6
NUM_STEPS = 300
BATCH_SIZE = 64
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
MEASURE_EVERY = 10
REPORT_STEPS = [0, 50, 100, 200, 300]

# Random target matrix (fixed)
W_target = np.random.randn(DIM, DIM) * 0.5

# Random input data (fixed batch)
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def compute_product_matrix(weights):
    """Compute W_L @ ... @ W_1."""
    product = np.eye(DIM)
    for W in weights:
        product = W @ product
    return product


def effective_rank(M):
    """
    Compute effective rank of matrix M.
    erank = exp(H) where H = Shannon entropy of normalized singular values.
    """
    sv = np.linalg.svd(M, compute_uv=False)
    # Remove near-zero singular values for numerical stability
    sv = sv[sv > 1e-15]
    if len(sv) == 0:
        return 1.0
    # Normalize to form a probability distribution
    p = sv / np.sum(sv)
    # Shannon entropy
    H = -np.sum(p * np.log(p))
    return np.exp(H)


def condition_number(M):
    """Compute condition number kappa = sigma_max / sigma_min."""
    sv = np.linalg.svd(M, compute_uv=False)
    if sv[-1] < 1e-15:
        return np.inf
    return sv[0] / sv[-1]


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
        for step in range(100):
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

def run_and_measure(optimizer_name, optimizer_fn, lr, num_steps):
    """
    Run optimizer for num_steps and measure effective rank + condition number
    at every MEASURE_EVERY steps.

    Returns dict with:
      steps: list of step numbers
      per_layer_erank: array (num_measurements, NUM_LAYERS)
      product_erank: array (num_measurements,)
      per_layer_kappa: array (num_measurements, NUM_LAYERS)
      product_kappa: array (num_measurements,)
      losses: array (num_measurements,)
    """
    np.random.seed(42)
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    steps_list = []
    per_layer_erank = []
    product_erank_list = []
    per_layer_kappa = []
    product_kappa_list = []
    losses = []

    def measure(step):
        steps_list.append(step)

        # Per-layer measurements
        layer_eranks = []
        layer_kappas = []
        for i in range(NUM_LAYERS):
            layer_eranks.append(effective_rank(weights[i]))
            layer_kappas.append(condition_number(weights[i]))
        per_layer_erank.append(layer_eranks)
        per_layer_kappa.append(layer_kappas)

        # Product matrix measurements
        W_prod = compute_product_matrix(weights)
        product_erank_list.append(effective_rank(W_prod))
        product_kappa_list.append(condition_number(W_prod))

        # Loss
        losses.append(compute_loss(weights, X_data, W_target))

    # Measure at step 0
    measure(0)

    for step in range(1, num_steps + 1):
        weights, velocities = optimizer_fn(weights, velocities, lr)

        # Check for divergence
        loss = compute_loss(weights, X_data, W_target)
        if np.isnan(loss) or loss > 1e10:
            print(f"    WARNING: {optimizer_name} diverged at step {step}!")
            break

        if step % MEASURE_EVERY == 0:
            measure(step)

    return {
        'steps': np.array(steps_list),
        'per_layer_erank': np.array(per_layer_erank),
        'product_erank': np.array(product_erank_list),
        'per_layer_kappa': np.array(per_layer_kappa),
        'product_kappa': np.array(product_kappa_list),
        'losses': np.array(losses),
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 100)
print("1.3a-i: EFFECTIVE RANK -- LAYER-BY-LAYER vs AGGREGATE")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer deep linear net (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"Measure every {MEASURE_EVERY} steps")
print(f"Effective rank = exp(Shannon entropy of normalized singular values)")
print(f"  Maximum possible erank = {DIM} (all singular values equal)")
print(f"  Minimum possible erank = 1 (rank-1 matrix)")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}")
print("=" * 100)

# Find stable SGD learning rate
lr_sgd = find_stable_lr_sgd()
print(f"\nSGD learning rate (max stable): {lr_sgd}")
print(f"Muon learning rate (fixed):     {LR_MUON}")

# Verify both optimizers train properly
np.random.seed(42)
w_test = init_weights(NUM_LAYERS)
loss_init = compute_loss(w_test, X_data, W_target)
print(f"\nInitial loss: {loss_init:.6e}")

# Run both optimizers
print(f"\n{'=' * 100}")
print("RUNNING OPTIMIZERS AND MEASURING EFFECTIVE RANK")
print("=" * 100)

print("\n  Running SGD...", flush=True)
results_sgd = run_and_measure('SGD', sgd_step, lr_sgd, NUM_STEPS)
print(f"    SGD final loss: {results_sgd['losses'][-1]:.6e}")

print("\n  Running Muon...", flush=True)
results_muon = run_and_measure('Muon', muon_step, LR_MUON, NUM_STEPS)
print(f"    Muon final loss: {results_muon['losses'][-1]:.6e}")


# =============================================================================
# TABLE 1: Per-Layer Effective Rank (Mean/Min/Max across layers)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 1: PER-LAYER EFFECTIVE RANK -- MEAN (MIN, MAX) ACROSS {0} LAYERS".format(NUM_LAYERS))
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD Mean':>10} {'(min':>6} {'max)':>6} | "
      f"{'Muon Mean':>10} {'(min':>6} {'max)':>6} | {'Muon-SGD':>8}")
print("-" * 80)

for step in REPORT_STEPS:
    # Find index in recorded steps
    sgd_idx = np.searchsorted(results_sgd['steps'], step)
    muon_idx = np.searchsorted(results_muon['steps'], step)

    if sgd_idx >= len(results_sgd['steps']):
        sgd_idx = len(results_sgd['steps']) - 1
    if muon_idx >= len(results_muon['steps']):
        muon_idx = len(results_muon['steps']) - 1

    sgd_eranks = results_sgd['per_layer_erank'][sgd_idx]
    muon_eranks = results_muon['per_layer_erank'][muon_idx]

    sgd_mean = np.mean(sgd_eranks)
    sgd_min = np.min(sgd_eranks)
    sgd_max = np.max(sgd_eranks)
    muon_mean = np.mean(muon_eranks)
    muon_min = np.min(muon_eranks)
    muon_max = np.max(muon_eranks)

    print(f"{step:6d} | {sgd_mean:10.2f} ({sgd_min:5.2f} {sgd_max:5.2f}) | "
          f"{muon_mean:10.2f} ({muon_min:5.2f} {muon_max:5.2f}) | {muon_mean - sgd_mean:+8.2f}")


# =============================================================================
# TABLE 2: Product Effective Rank
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 2: PRODUCT EFFECTIVE RANK (W6 @ ... @ W1)")
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD erank':>10} | {'Muon erank':>11} | {'Muon-SGD':>8} | "
      f"{'SGD/n':>6} | {'Muon/n':>6}")
print("-" * 70)

for step in REPORT_STEPS:
    sgd_idx = np.searchsorted(results_sgd['steps'], step)
    muon_idx = np.searchsorted(results_muon['steps'], step)

    if sgd_idx >= len(results_sgd['steps']):
        sgd_idx = len(results_sgd['steps']) - 1
    if muon_idx >= len(results_muon['steps']):
        muon_idx = len(results_muon['steps']) - 1

    sgd_pe = results_sgd['product_erank'][sgd_idx]
    muon_pe = results_muon['product_erank'][muon_idx]

    print(f"{step:6d} | {sgd_pe:10.2f} | {muon_pe:11.2f} | {muon_pe - sgd_pe:+8.2f} | "
          f"{sgd_pe / DIM:6.3f} | {muon_pe / DIM:6.3f}")


# =============================================================================
# TABLE 3: Per-Layer Condition Number (Mean across layers)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 3: PER-LAYER CONDITION NUMBER -- MEAN (MIN, MAX) ACROSS LAYERS")
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD Mean':>10} {'(min':>6} {'max)':>8} | "
      f"{'Muon Mean':>10} {'(min':>6} {'max)':>8}")
print("-" * 80)

for step in REPORT_STEPS:
    sgd_idx = np.searchsorted(results_sgd['steps'], step)
    muon_idx = np.searchsorted(results_muon['steps'], step)

    if sgd_idx >= len(results_sgd['steps']):
        sgd_idx = len(results_sgd['steps']) - 1
    if muon_idx >= len(results_muon['steps']):
        muon_idx = len(results_muon['steps']) - 1

    sgd_kappas = results_sgd['per_layer_kappa'][sgd_idx]
    muon_kappas = results_muon['per_layer_kappa'][muon_idx]

    sgd_mean = np.mean(sgd_kappas)
    sgd_min = np.min(sgd_kappas)
    sgd_max = np.max(sgd_kappas)
    muon_mean = np.mean(muon_kappas)
    muon_min = np.min(muon_kappas)
    muon_max = np.max(muon_kappas)

    print(f"{step:6d} | {sgd_mean:10.2f} ({sgd_min:5.2f} {sgd_max:8.2f}) | "
          f"{muon_mean:10.2f} ({muon_min:5.2f} {muon_max:8.2f})")


# =============================================================================
# TABLE 4: Product Condition Number
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 4: PRODUCT CONDITION NUMBER (W6 @ ... @ W1)")
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD kappa':>12} | {'Muon kappa':>12} | {'Ratio SGD/Muon':>15}")
print("-" * 55)

for step in REPORT_STEPS:
    sgd_idx = np.searchsorted(results_sgd['steps'], step)
    muon_idx = np.searchsorted(results_muon['steps'], step)

    if sgd_idx >= len(results_sgd['steps']):
        sgd_idx = len(results_sgd['steps']) - 1
    if muon_idx >= len(results_muon['steps']):
        muon_idx = len(results_muon['steps']) - 1

    sgd_pk = results_sgd['product_kappa'][sgd_idx]
    muon_pk = results_muon['product_kappa'][muon_idx]

    ratio = sgd_pk / muon_pk if muon_pk > 0 and np.isfinite(muon_pk) else np.nan

    sgd_str = f"{sgd_pk:12.2f}" if np.isfinite(sgd_pk) else f"{'inf':>12}"
    muon_str = f"{muon_pk:12.2f}" if np.isfinite(muon_pk) else f"{'inf':>12}"
    ratio_str = f"{ratio:15.2f}" if np.isfinite(ratio) else f"{'N/A':>15}"

    print(f"{step:6d} | {sgd_str} | {muon_str} | {ratio_str}")


# =============================================================================
# TABLE 5: Full Per-Layer Breakdown at Key Steps
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 5: PER-LAYER EFFECTIVE RANK BREAKDOWN AT KEY STEPS")
print("=" * 100)

for step in [0, 100, 300]:
    sgd_idx = np.searchsorted(results_sgd['steps'], step)
    muon_idx = np.searchsorted(results_muon['steps'], step)

    if sgd_idx >= len(results_sgd['steps']):
        sgd_idx = len(results_sgd['steps']) - 1
    if muon_idx >= len(results_muon['steps']):
        muon_idx = len(results_muon['steps']) - 1

    print(f"\n  Step {step}:")
    print(f"  {'Layer':>6} | {'SGD erank':>10} | {'Muon erank':>11} | "
          f"{'SGD kappa':>10} | {'Muon kappa':>11}")
    print("  " + "-" * 65)

    for layer in range(NUM_LAYERS):
        sgd_er = results_sgd['per_layer_erank'][sgd_idx, layer]
        muon_er = results_muon['per_layer_erank'][muon_idx, layer]
        sgd_kp = results_sgd['per_layer_kappa'][sgd_idx, layer]
        muon_kp = results_muon['per_layer_kappa'][muon_idx, layer]

        print(f"  {layer:6d} | {sgd_er:10.2f} | {muon_er:11.2f} | "
              f"{sgd_kp:10.2f} | {muon_kp:11.2f}")

    # Product row
    sgd_pe = results_sgd['product_erank'][sgd_idx]
    muon_pe = results_muon['product_erank'][muon_idx]
    sgd_pk = results_sgd['product_kappa'][sgd_idx]
    muon_pk = results_muon['product_kappa'][muon_idx]

    print("  " + "-" * 65)
    sgd_pk_str = f"{sgd_pk:10.2f}" if np.isfinite(sgd_pk) else f"{'inf':>10}"
    muon_pk_str = f"{muon_pk:11.2f}" if np.isfinite(muon_pk) else f"{'inf':>11}"
    print(f"  {'PROD':>6} | {sgd_pe:10.2f} | {muon_pe:11.2f} | "
          f"{sgd_pk_str} | {muon_pk_str}")


# =============================================================================
# KEY METRIC: erank retention ratio
# =============================================================================

print(f"\n\n{'=' * 100}")
print("EFFECTIVE RANK RETENTION ANALYSIS")
print("=" * 100)

# Per-layer retention: erank(step=300) / erank(step=0)
sgd_erank_init = np.mean(results_sgd['per_layer_erank'][0])
sgd_erank_final = np.mean(results_sgd['per_layer_erank'][-1])
muon_erank_init = np.mean(results_muon['per_layer_erank'][0])
muon_erank_final = np.mean(results_muon['per_layer_erank'][-1])

print(f"\n  Per-Layer Effective Rank (mean across layers):")
print(f"    SGD:   init={sgd_erank_init:.2f} -> final={sgd_erank_final:.2f}  "
      f"retention={sgd_erank_final / sgd_erank_init:.3f}")
print(f"    Muon:  init={muon_erank_init:.2f} -> final={muon_erank_final:.2f}  "
      f"retention={muon_erank_final / muon_erank_init:.3f}")

# Product retention
sgd_prod_init = results_sgd['product_erank'][0]
sgd_prod_final = results_sgd['product_erank'][-1]
muon_prod_init = results_muon['product_erank'][0]
muon_prod_final = results_muon['product_erank'][-1]

print(f"\n  Product Effective Rank:")
print(f"    SGD:   init={sgd_prod_init:.2f} -> final={sgd_prod_final:.2f}  "
      f"retention={sgd_prod_final / sgd_prod_init:.3f}")
print(f"    Muon:  init={muon_prod_init:.2f} -> final={muon_prod_final:.2f}  "
      f"retention={muon_prod_final / muon_prod_init:.3f}")

# Ratio of product erank to per-layer mean erank (how much does depth compress?)
print(f"\n  Depth Compression Factor (product_erank / mean_per_layer_erank):")
sgd_comp_init = sgd_prod_init / sgd_erank_init
sgd_comp_final = sgd_prod_final / sgd_erank_final
muon_comp_init = muon_prod_init / muon_erank_init
muon_comp_final = muon_prod_final / muon_erank_final

print(f"    SGD:   init={sgd_comp_init:.3f} -> final={sgd_comp_final:.3f}")
print(f"    Muon:  init={muon_comp_init:.3f} -> final={muon_comp_final:.3f}")


# =============================================================================
# PLOT: Effective Rank Over Training
# =============================================================================

print(f"\n\n{'=' * 100}")
print("GENERATING PLOTS")
print("=" * 100)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('1.3a-i: Effective Rank -- Layer-by-Layer vs Aggregate\n'
                 f'{NUM_LAYERS}-layer linear net, dim={DIM}, {NUM_STEPS} steps',
                 fontsize=14, fontweight='bold')

    # --- Panel (a): Per-layer erank over time ---
    ax = axes[0, 0]
    ax.set_title('(a) Per-Layer Effective Rank Over Training')

    # SGD: plot each layer as thin line, mean as bold
    for layer in range(NUM_LAYERS):
        ax.plot(results_sgd['steps'], results_sgd['per_layer_erank'][:, layer],
                'b-', alpha=0.25, linewidth=0.8)
    sgd_layer_mean = np.mean(results_sgd['per_layer_erank'], axis=1)
    ax.plot(results_sgd['steps'], sgd_layer_mean, 'b-', linewidth=2.5,
            label=f'SGD (mean over layers)')

    # Muon: plot each layer as thin line, mean as bold
    for layer in range(NUM_LAYERS):
        ax.plot(results_muon['steps'], results_muon['per_layer_erank'][:, layer],
                'r-', alpha=0.25, linewidth=0.8)
    muon_layer_mean = np.mean(results_muon['per_layer_erank'], axis=1)
    ax.plot(results_muon['steps'], muon_layer_mean, 'r-', linewidth=2.5,
            label=f'Muon (mean over layers)')

    ax.axhline(y=DIM, color='green', linestyle='--', alpha=0.5,
               label=f'Max erank = {DIM}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): Product erank over time ---
    ax = axes[0, 1]
    ax.set_title('(b) Product Matrix Effective Rank Over Training')

    ax.plot(results_sgd['steps'], results_sgd['product_erank'], 'b-',
            linewidth=2.5, marker='o', markersize=3,
            label='SGD (product)')
    ax.plot(results_muon['steps'], results_muon['product_erank'], 'r-',
            linewidth=2.5, marker='s', markersize=3,
            label='Muon (product)')

    ax.axhline(y=DIM, color='green', linestyle='--', alpha=0.5,
               label=f'Max erank = {DIM}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Per-layer kappa over time ---
    ax = axes[1, 0]
    ax.set_title('(c) Per-Layer Condition Number Over Training')

    for layer in range(NUM_LAYERS):
        ax.semilogy(results_sgd['steps'], results_sgd['per_layer_kappa'][:, layer],
                     'b-', alpha=0.25, linewidth=0.8)
    sgd_kappa_mean = np.mean(results_sgd['per_layer_kappa'], axis=1)
    ax.semilogy(results_sgd['steps'], sgd_kappa_mean, 'b-', linewidth=2.5,
                label='SGD (mean)')

    for layer in range(NUM_LAYERS):
        ax.semilogy(results_muon['steps'], results_muon['per_layer_kappa'][:, layer],
                     'r-', alpha=0.25, linewidth=0.8)
    muon_kappa_mean = np.mean(results_muon['per_layer_kappa'], axis=1)
    ax.semilogy(results_muon['steps'], muon_kappa_mean, 'r-', linewidth=2.5,
                label='Muon (mean)')

    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number (kappa)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (d): Product kappa over time ---
    ax = axes[1, 1]
    ax.set_title('(d) Product Matrix Condition Number Over Training')

    # Filter out inf values for plotting
    sgd_pk = results_sgd['product_kappa'].copy()
    muon_pk = results_muon['product_kappa'].copy()
    sgd_pk[~np.isfinite(sgd_pk)] = np.nan
    muon_pk[~np.isfinite(muon_pk)] = np.nan

    ax.semilogy(results_sgd['steps'], sgd_pk, 'b-',
                linewidth=2.5, marker='o', markersize=3,
                label='SGD (product)')
    ax.semilogy(results_muon['steps'], muon_pk, 'r-',
                linewidth=2.5, marker='s', markersize=3,
                label='Muon (product)')

    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number (kappa)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'effective_rank_evolution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {plot_path}")

except ImportError:
    print("\n  WARNING: matplotlib not available, skipping plots.")
    plot_path = None


# =============================================================================
# ADDITIONAL PLOT: erank comparison (per-layer vs product, side by side)
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('1.3a-i: Per-Layer vs Product Effective Rank\n'
                 'Does the gauge fix per-layer but not the product?',
                 fontsize=13, fontweight='bold')

    # --- Left: SGD ---
    ax = axes[0]
    ax.set_title('SGD')

    sgd_layer_mean = np.mean(results_sgd['per_layer_erank'], axis=1)
    sgd_layer_min = np.min(results_sgd['per_layer_erank'], axis=1)
    sgd_layer_max = np.max(results_sgd['per_layer_erank'], axis=1)

    ax.fill_between(results_sgd['steps'], sgd_layer_min, sgd_layer_max,
                     alpha=0.15, color='blue')
    ax.plot(results_sgd['steps'], sgd_layer_mean, 'b-', linewidth=2,
            label='Per-layer (mean)')
    ax.plot(results_sgd['steps'], results_sgd['product_erank'], 'b--',
            linewidth=2, label='Product')
    ax.axhline(y=DIM, color='green', linestyle=':', alpha=0.5,
               label=f'Max = {DIM}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_ylim(bottom=0, top=DIM + 2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Right: Muon ---
    ax = axes[1]
    ax.set_title('Muon')

    muon_layer_mean = np.mean(results_muon['per_layer_erank'], axis=1)
    muon_layer_min = np.min(results_muon['per_layer_erank'], axis=1)
    muon_layer_max = np.max(results_muon['per_layer_erank'], axis=1)

    ax.fill_between(results_muon['steps'], muon_layer_min, muon_layer_max,
                     alpha=0.15, color='red')
    ax.plot(results_muon['steps'], muon_layer_mean, 'r-', linewidth=2,
            label='Per-layer (mean)')
    ax.plot(results_muon['steps'], results_muon['product_erank'], 'r--',
            linewidth=2, label='Product')
    ax.axhline(y=DIM, color='green', linestyle=':', alpha=0.5,
               label=f'Max = {DIM}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_ylim(bottom=0, top=DIM + 2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = os.path.join(SCRIPT_DIR, 'erank_per_layer_vs_product.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path2}")

except ImportError:
    pass


# =============================================================================
# VERDICT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL VERDICT: EFFECTIVE RANK LAYER-BY-LAYER vs AGGREGATE")
print("=" * 100)

# Test 1: Muon maintains higher per-layer erank than SGD at step 300
sgd_final_mean_erank = np.mean(results_sgd['per_layer_erank'][-1])
muon_final_mean_erank = np.mean(results_muon['per_layer_erank'][-1])
test1_pass = muon_final_mean_erank > sgd_final_mean_erank

# Test 2: Muon per-layer erank stays near n=32 (within 80% of max)
test2_pass = muon_final_mean_erank > 0.8 * DIM

# Test 3: Product erank is lower than per-layer erank for both
# (depth compresses the spectrum)
sgd_compression = results_sgd['product_erank'][-1] / sgd_final_mean_erank
muon_compression = results_muon['product_erank'][-1] / muon_final_mean_erank
test3_pass = sgd_compression < 1.0 and muon_compression < 1.0

# Test 4: Muon product kappa is smaller than SGD product kappa
# (consistent with 1.1a-i: O(L) vs O(c^L))
sgd_final_prod_kappa = results_sgd['product_kappa'][-1]
muon_final_prod_kappa = results_muon['product_kappa'][-1]
test4_pass = muon_final_prod_kappa < sgd_final_prod_kappa

tests_passed = sum([test1_pass, test2_pass, test3_pass, test4_pass])
tests_total = 4

print(f"""
  MEASURED QUANTITIES AT STEP {NUM_STEPS}:
  ---------------------------------------------------------------
  Per-layer erank (mean):
    SGD:   {sgd_final_mean_erank:.2f}  ({sgd_final_mean_erank/DIM:.1%} of max)
    Muon:  {muon_final_mean_erank:.2f}  ({muon_final_mean_erank/DIM:.1%} of max)

  Product erank:
    SGD:   {results_sgd['product_erank'][-1]:.2f}  ({results_sgd['product_erank'][-1]/DIM:.1%} of max)
    Muon:  {results_muon['product_erank'][-1]:.2f}  ({results_muon['product_erank'][-1]/DIM:.1%} of max)

  Depth compression (product/per-layer):
    SGD:   {sgd_compression:.3f}
    Muon:  {muon_compression:.3f}

  Product condition number:
    SGD:   {sgd_final_prod_kappa:.2f}{'' if np.isfinite(sgd_final_prod_kappa) else ' (diverged!)'}
    Muon:  {muon_final_prod_kappa:.2f}{'' if np.isfinite(muon_final_prod_kappa) else ' (diverged!)'}
  ---------------------------------------------------------------

  HYPOTHESIS CHECK:
  ---------------------------------------------------------------
  T1: Muon per-layer erank > SGD per-layer erank
      Muon: {muon_final_mean_erank:.2f} vs SGD: {sgd_final_mean_erank:.2f}
      -> {"CONFIRMED" if test1_pass else "REJECTED"}

  T2: Muon per-layer erank near n={DIM} (>80% = {0.8*DIM:.1f})
      Muon: {muon_final_mean_erank:.2f}
      -> {"CONFIRMED" if test2_pass else "REJECTED"}

  T3: Product erank < per-layer erank (depth compresses)
      SGD compression:  {sgd_compression:.3f}
      Muon compression: {muon_compression:.3f}
      -> {"CONFIRMED" if test3_pass else "REJECTED"}

  T4: Muon product kappa < SGD product kappa (consistent with 1.1a-i)
      SGD: {sgd_final_prod_kappa:.2f} vs Muon: {muon_final_prod_kappa:.2f}
      -> {"CONFIRMED" if test4_pass else "REJECTED"}
  ---------------------------------------------------------------
""")

if tests_passed == 4:
    overall = "PASS"
    detail = (
        "All four tests pass:\n"
        "  1. Muon maintains higher per-layer effective rank\n"
        "  2. Muon per-layer erank is near full rank (n=32)\n"
        "  3. Product erank is lower than per-layer (depth compresses)\n"
        "  4. Muon product kappa < SGD product kappa\n"
        "\n"
        "  This confirms the hypothesis: Muon's gauge-fixing operates per-layer,\n"
        "  maintaining high effective rank at each layer, while the product\n"
        "  spectrum still concentrates due to depth."
    )
elif tests_passed >= 3:
    overall = "PARTIAL PASS"
    detail = (
        f"  {tests_passed}/4 tests pass.\n"
        f"  T1 (Muon > SGD per-layer erank):  {'PASS' if test1_pass else 'FAIL'}\n"
        f"  T2 (Muon erank near n):           {'PASS' if test2_pass else 'FAIL'}\n"
        f"  T3 (product < per-layer):         {'PASS' if test3_pass else 'FAIL'}\n"
        f"  T4 (Muon kappa < SGD kappa):      {'PASS' if test4_pass else 'FAIL'}\n"
    )
elif tests_passed >= 2:
    overall = "WEAK SIGNAL"
    detail = (
        f"  {tests_passed}/4 tests pass.\n"
        f"  T1 (Muon > SGD per-layer erank):  {'PASS' if test1_pass else 'FAIL'}\n"
        f"  T2 (Muon erank near n):           {'PASS' if test2_pass else 'FAIL'}\n"
        f"  T3 (product < per-layer):         {'PASS' if test3_pass else 'FAIL'}\n"
        f"  T4 (Muon kappa < SGD kappa):      {'PASS' if test4_pass else 'FAIL'}\n"
    )
else:
    overall = "FAIL"
    detail = (
        f"  Only {tests_passed}/4 tests pass.\n"
        f"  T1 (Muon > SGD per-layer erank):  {'PASS' if test1_pass else 'FAIL'}\n"
        f"  T2 (Muon erank near n):           {'PASS' if test2_pass else 'FAIL'}\n"
        f"  T3 (product < per-layer):         {'PASS' if test3_pass else 'FAIL'}\n"
        f"  T4 (Muon kappa < SGD kappa):      {'PASS' if test4_pass else 'FAIL'}\n"
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
