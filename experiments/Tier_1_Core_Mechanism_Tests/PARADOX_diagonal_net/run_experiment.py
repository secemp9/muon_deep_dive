#!/usr/bin/env python3
"""
Experiment 3.18: Paradox on Diagonal Nets
==========================================

HYPOTHESIS:
  The Muon Paradox (diverse weights, consistent losses) requires gauge symmetry.
  It should VANISH for diagonal networks where dim(gauge) = 0.

  Each layer is a diagonal matrix d_i in R^n. The product = element-wise product
  of diagonals. There is no O(n) gauge group -- every parameter is physical.

  "Muon" on a diagonal = Newton-Schulz ortho of diag(d) -> sign normalization,
  i.e. each element -> +/- 1.  This is coordinate-wise sign(G).

PROTOCOL:
  4-layer diagonal deep linear (n=32), 20 independent runs, 500 steps.
  Compare weight diversity ratio and loss std for Muon vs SGD.

PREDICTION:
  No paradox -- diversity ratios (func_div / weight_div) should be SIMILAR
  for Muon and SGD, because there are no gauge directions to explore.

SETUP:
  - Target: random diagonal d_target in R^n
  - Each layer: diagonal d_i in R^n  (stored as 1D vectors, not full matrices)
  - Forward: f(x) = diag(d_L * d_{L-1} * ... * d_1) @ x  (element-wise product)
  - Loss: 0.5 * ||f(X) - diag(d_target) @ X||^2 / N
  - Muon: Newton-Schulz on diag(g_i) -> sign(g_i) * ones  (orthogonal diagonal)
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
LR_MUON = 0.02
LR_SGD = 0.01
MOMENTUM = 0.9
NS_ITERS = 5
NUM_INDEPENDENT_RUNS = 20
NUM_TEST_INPUTS = 50

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed target and data
d_target = np.random.randn(DIM)  # target diagonal
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3
X_test = np.random.randn(DIM, NUM_TEST_INPUTS) * 0.3


# =============================================================================
# DIAGONAL NETWORK
# =============================================================================

def init_diag(num_layers, seed=42):
    """Initialize diagonal layers near ones for stability."""
    rng = np.random.RandomState(seed)
    diags = []
    for _ in range(num_layers):
        d = np.ones(DIM) + rng.randn(DIM) * 0.1
        diags.append(d.copy())
    return diags


def forward_diag(diags, X):
    """Forward pass: multiply X by product of diagonals."""
    prod = np.ones(DIM)
    for d in diags:
        prod = prod * d  # element-wise product
    # Apply: diag(prod) @ X = prod[:, None] * X
    return prod[:, None] * X


def compute_loss_diag(diags, X, target_diag):
    """Quadratic loss."""
    pred = forward_diag(diags, X)
    target_out = target_diag[:, None] * X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_diag(diags, X, target_diag):
    """Backprop for diagonal network."""
    num_layers = len(diags)
    N = X.shape[1]

    # Product of all diagonals
    prod = np.ones(DIM)
    for d in diags:
        prod = prod * d

    # Error: (prod - target_diag) element-wise
    err = prod - target_diag  # (DIM,)

    # Loss = 0.5 * mean_over_samples(sum_i (err_i * x_i)^2)
    # dL/d_prod_i = err_i * mean(x_i^2 across batch)  ... but let me be precise.
    # pred = prod[:, None] * X
    # target_out = target_diag[:, None] * X
    # diff = (prod - target_diag)[:, None] * X
    # loss = 0.5 * mean_cols(sum_rows(diff^2))
    # dL/dprod_i = err_i * mean(X_i^2 across batch)

    # Compute mean(X_i^2) for each dimension
    x_sq_mean = np.mean(X ** 2, axis=1)  # (DIM,)
    dL_dprod = err * x_sq_mean  # (DIM,)

    # Gradient for each layer:
    # prod = d_1 * d_2 * ... * d_L (element-wise)
    # dprod/d_{k,i} = prod_i / d_{k,i}
    grads = []
    for k in range(num_layers):
        # grad_k_i = dL_dprod_i * prod_i / d_k_i
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_k = dL_dprod * prod / (diags[k] + 1e-30)
        grads.append(grad_k)

    return grads


def newton_schulz_diagonal(g_vec, num_iters=NS_ITERS):
    """
    Newton-Schulz orthogonalization of a diagonal matrix diag(g_vec).
    A diagonal orthogonal matrix has entries +/-1.
    NS iteration on diag(g) converges to diag(sign(g)).
    We can verify: for diagonal X, A = X^T X = X^2 (diagonal),
    and 1.5*X - 0.5*X*A = 1.5*X - 0.5*X^3. Fixed points: x_i = +/-1.
    """
    norm = np.linalg.norm(g_vec)
    if norm < 1e-12:
        return g_vec.copy()
    x = g_vec / norm
    for _ in range(num_iters):
        # For diagonal: X_{k+1} = 1.5 * X_k - 0.5 * X_k^3
        x = 1.5 * x - 0.5 * x ** 3
    return x


# =============================================================================
# OPTIMIZERS
# =============================================================================

def sgd_step_diag(diags, velocities, grads, lr):
    for i in range(len(diags)):
        velocities[i] = MOMENTUM * velocities[i] + grads[i]
        diags[i] = diags[i] - lr * velocities[i]
    return diags, velocities


def muon_step_diag(diags, velocities, grads, lr):
    for i in range(len(diags)):
        ortho_grad = newton_schulz_diagonal(grads[i])
        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
        diags[i] = diags[i] - lr * velocities[i]
    return diags, velocities


# =============================================================================
# LEARNING RATE FINDER
# =============================================================================

def find_stable_lr(optimizer_fn, candidates):
    for lr in candidates:
        np.random.seed(42)
        diags = init_diag(NUM_LAYERS)
        velocities = [np.zeros(DIM) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss_diag(diags, X_data, d_target)
        stable = True
        for step in range(100):
            grads = compute_gradients_diag(diags, X_data, d_target)
            diags, velocities = optimizer_fn(diags, velocities, grads, lr)
            loss = compute_loss_diag(diags, X_data, d_target)
            if np.isnan(loss) or loss > initial_loss * 50:
                stable = False
                break
        if stable:
            return lr
    return candidates[-1]


# =============================================================================
# CONVERGENCE BASIN ANALYSIS (same as MUON_PARADOX Face 2)
# =============================================================================

def measure_convergence_basin(lr_sgd, lr_muon, num_runs, num_steps):
    """
    Run many independent initializations with each optimizer.
    Measure weight diversity, function diversity, loss diversity.
    """
    results = {}

    for opt_name, opt_fn, lr in [('SGD', sgd_step_diag, lr_sgd),
                                   ('Muon', muon_step_diag, lr_muon)]:
        final_diags_list = []
        final_functions = []
        final_losses = []

        for run_idx in range(num_runs):
            diags = init_diag(NUM_LAYERS, seed=1000 + run_idx)
            velocities = [np.zeros(DIM) for _ in range(NUM_LAYERS)]

            for step in range(num_steps):
                grads = compute_gradients_diag(diags, X_data, d_target)
                diags, velocities = opt_fn(diags, velocities, grads, lr)
                loss = compute_loss_diag(diags, X_data, d_target)
                if np.isnan(loss) or loss > 1e10:
                    break

            final_diags_list.append([d.copy() for d in diags])
            final_functions.append(forward_diag(diags, X_test).copy())
            final_losses.append(compute_loss_diag(diags, X_data, d_target))

        # Compute pairwise diversity
        n = len(final_diags_list)
        weight_dists = []
        func_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d_w = 0.0
                for k in range(NUM_LAYERS):
                    d_w += np.linalg.norm(final_diags_list[i][k] - final_diags_list[j][k]) ** 2
                weight_dists.append(np.sqrt(d_w))
                d_f = np.linalg.norm(final_functions[i] - final_functions[j], 'fro')
                d_f /= np.linalg.norm(X_test, 'fro')
                func_dists.append(d_f)

        results[opt_name] = {
            'weight_diversity_mean': np.mean(weight_dists),
            'weight_diversity_std': np.std(weight_dists),
            'func_diversity_mean': np.mean(func_dists),
            'func_diversity_std': np.std(func_dists),
            'loss_mean': np.mean(final_losses),
            'loss_std': np.std(final_losses),
            'losses': np.array(final_losses),
        }

    return results


# =============================================================================
# FULL-MATRIX REFERENCE (for comparison)
# =============================================================================

def init_weights_full(num_layers, seed=42):
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


def forward_full(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss_full(weights, X, W_target):
    pred = forward_full(weights, X)
    target_out = W_target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_full(weights, X, W_target):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())
    target_out = W_target @ X
    delta = (activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
    return grads


def newton_schulz_full(G, num_iters=NS_ITERS):
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def measure_convergence_basin_full(lr_sgd, lr_muon, num_runs, num_steps):
    W_target_full = np.diag(d_target)  # Use same target, but as full matrix

    results = {}
    for opt_name, lr in [('SGD', lr_sgd), ('Muon', lr_muon)]:
        final_weights_list = []
        final_functions = []
        final_losses = []

        for run_idx in range(num_runs):
            weights = init_weights_full(NUM_LAYERS, seed=1000 + run_idx)
            velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

            for step in range(num_steps):
                grads = compute_gradients_full(weights, X_data, W_target_full)
                if opt_name == 'SGD':
                    for i in range(NUM_LAYERS):
                        velocities[i] = MOMENTUM * velocities[i] + grads[i]
                        weights[i] = weights[i] - lr * velocities[i]
                else:
                    for i in range(NUM_LAYERS):
                        ortho_grad = newton_schulz_full(grads[i])
                        velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                        weights[i] = weights[i] - lr * velocities[i]
                loss = compute_loss_full(weights, X_data, W_target_full)
                if np.isnan(loss) or loss > 1e10:
                    break

            final_weights_list.append([w.copy() for w in weights])
            final_functions.append(forward_full(weights, X_test).copy())
            final_losses.append(compute_loss_full(weights, X_data, W_target_full))

        n = len(final_weights_list)
        weight_dists = []
        func_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d_w = 0.0
                for k in range(NUM_LAYERS):
                    d_w += np.linalg.norm(final_weights_list[i][k] - final_weights_list[j][k], 'fro') ** 2
                weight_dists.append(np.sqrt(d_w))
                d_f = np.linalg.norm(final_functions[i] - final_functions[j], 'fro')
                d_f /= np.linalg.norm(X_test, 'fro')
                func_dists.append(d_f)

        results[opt_name] = {
            'weight_diversity_mean': np.mean(weight_dists),
            'weight_diversity_std': np.std(weight_dists),
            'func_diversity_mean': np.mean(func_dists),
            'func_diversity_std': np.std(func_dists),
            'loss_mean': np.mean(final_losses),
            'loss_std': np.std(final_losses),
        }

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 90)
print("Experiment 3.18: Paradox on Diagonal Nets (dim(gauge) = 0)")
print("=" * 90)
print(f"Setup: {NUM_LAYERS}-layer diagonal net (n={DIM}), {NUM_INDEPENDENT_RUNS} runs, {NUM_STEPS} steps")
print(f"Prediction: NO paradox (diversity ratios similar for Muon and SGD)")
print("=" * 90)

# Find stable learning rates
print("\nFinding stable learning rates...")
lr_sgd = find_stable_lr(sgd_step_diag, [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])
lr_muon = find_stable_lr(muon_step_diag, [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])
print(f"  Diagonal SGD lr: {lr_sgd}")
print(f"  Diagonal Muon lr: {lr_muon}")

# Run diagonal net basin analysis
print("\n--- DIAGONAL NET (dim(gauge) = 0) ---")
diag_results = measure_convergence_basin(lr_sgd, lr_muon, NUM_INDEPENDENT_RUNS, NUM_STEPS)

for opt_name in ['SGD', 'Muon']:
    r = diag_results[opt_name]
    print(f"  {opt_name}: loss={r['loss_mean']:.6e} +/- {r['loss_std']:.6e}, "
          f"d_weight={r['weight_diversity_mean']:.6f}, d_func={r['func_diversity_mean']:.6f}")

# Run full-matrix reference
print("\n--- FULL MATRIX NET (dim(gauge) > 0) ---")
print("Finding full-matrix LRs...")

# Quick LR search for full-matrix
def find_lr_full(opt_name, candidates):
    W_target_full = np.diag(d_target)
    for lr in candidates:
        np.random.seed(42)
        weights = init_weights_full(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss_full(weights, X_data, W_target_full)
        stable = True
        for step in range(100):
            grads = compute_gradients_full(weights, X_data, W_target_full)
            if opt_name == 'SGD':
                for i in range(NUM_LAYERS):
                    velocities[i] = MOMENTUM * velocities[i] + grads[i]
                    weights[i] = weights[i] - lr * velocities[i]
            else:
                for i in range(NUM_LAYERS):
                    ortho_grad = newton_schulz_full(grads[i])
                    velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                    weights[i] = weights[i] - lr * velocities[i]
            loss = compute_loss_full(weights, X_data, W_target_full)
            if np.isnan(loss) or loss > initial_loss * 50:
                stable = False
                break
        if stable:
            return lr
    return candidates[-1]

lr_sgd_full = find_lr_full('SGD', [0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001])
lr_muon_full = find_lr_full('Muon', [0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001])
print(f"  Full SGD lr: {lr_sgd_full}")
print(f"  Full Muon lr: {lr_muon_full}")

full_results = measure_convergence_basin_full(lr_sgd_full, lr_muon_full,
                                                NUM_INDEPENDENT_RUNS, NUM_STEPS)

for opt_name in ['SGD', 'Muon']:
    r = full_results[opt_name]
    print(f"  {opt_name}: loss={r['loss_mean']:.6e} +/- {r['loss_std']:.6e}, "
          f"d_weight={r['weight_diversity_mean']:.6f}, d_func={r['func_diversity_mean']:.6f}")


# =============================================================================
# RESULTS TABLE
# =============================================================================

print(f"\n\n{'=' * 90}")
print("RESULTS: PARADOX RATIO = func_diversity / weight_diversity")
print(f"{'=' * 90}")
print(f"  (Lower ratio for Muon vs SGD = paradox present = gauge exploration)")
print()

print(f"{'Network':<20} {'Optimizer':<10} {'Weight Div':>12} {'Func Div':>12} {'Ratio':>10} {'Loss Std':>14}")
print("-" * 80)

# Diagonal
for opt_name in ['SGD', 'Muon']:
    r = diag_results[opt_name]
    wd = r['weight_diversity_mean']
    fd = r['func_diversity_mean']
    ratio = fd / wd if wd > 1e-15 else float('nan')
    print(f"{'Diagonal':<20} {opt_name:<10} {wd:>12.6f} {fd:>12.6f} {ratio:>10.6f} {r['loss_std']:>14.6e}")

# Full matrix
for opt_name in ['SGD', 'Muon']:
    r = full_results[opt_name]
    wd = r['weight_diversity_mean']
    fd = r['func_diversity_mean']
    ratio = fd / wd if wd > 1e-15 else float('nan')
    print(f"{'Full Matrix':<20} {opt_name:<10} {wd:>12.6f} {fd:>12.6f} {ratio:>10.6f} {r['loss_std']:>14.6e}")

# Compute the paradox strength = ratio_SGD / ratio_Muon (>1 means paradox present)
diag_ratio_sgd = diag_results['SGD']['func_diversity_mean'] / (diag_results['SGD']['weight_diversity_mean'] + 1e-30)
diag_ratio_muon = diag_results['Muon']['func_diversity_mean'] / (diag_results['Muon']['weight_diversity_mean'] + 1e-30)
full_ratio_sgd = full_results['SGD']['func_diversity_mean'] / (full_results['SGD']['weight_diversity_mean'] + 1e-30)
full_ratio_muon = full_results['Muon']['func_diversity_mean'] / (full_results['Muon']['weight_diversity_mean'] + 1e-30)

diag_paradox = diag_ratio_sgd / (diag_ratio_muon + 1e-30)
full_paradox = full_ratio_sgd / (full_ratio_muon + 1e-30)

print()
print(f"  Diagonal:    SGD ratio = {diag_ratio_sgd:.6f}, Muon ratio = {diag_ratio_muon:.6f}")
print(f"               Paradox strength (SGD/Muon) = {diag_paradox:.4f}")
print(f"  Full matrix: SGD ratio = {full_ratio_sgd:.6f}, Muon ratio = {full_ratio_muon:.6f}")
print(f"               Paradox strength (SGD/Muon) = {full_paradox:.4f}")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 90}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 90}")

# T1: Full matrix should show paradox (Muon ratio < SGD ratio)
t1 = full_ratio_muon < full_ratio_sgd
print(f"\nT1: Full matrix shows paradox (Muon ratio < SGD ratio)?")
print(f"    Muon={full_ratio_muon:.6f} vs SGD={full_ratio_sgd:.6f}")
print(f"    --> {'PASS' if t1 else 'FAIL'}")

# T2: Diagonal net should NOT show paradox (ratios similar)
# "Similar" means the paradox strength is close to 1.0 (within 2x)
t2 = abs(diag_paradox - 1.0) < 1.0  # paradox strength between 0 and 2
print(f"\nT2: Diagonal net does NOT show paradox (strength near 1.0)?")
print(f"    Paradox strength = {diag_paradox:.4f} (expect near 1.0)")
print(f"    --> {'PASS' if t2 else 'FAIL'}")

# T3: Full matrix paradox should be STRONGER than diagonal
t3 = full_paradox > diag_paradox
print(f"\nT3: Full matrix paradox stronger than diagonal?")
print(f"    Full={full_paradox:.4f} vs Diagonal={diag_paradox:.4f}")
print(f"    --> {'PASS' if t3 else 'FAIL'}")

# T4: In diagonal net, Muon should NOT have higher weight diversity than SGD
# (or if it does, it should also have proportionally higher func diversity)
diag_muon_higher_wd = diag_results['Muon']['weight_diversity_mean'] > diag_results['SGD']['weight_diversity_mean']
diag_muon_lower_fd = diag_results['Muon']['func_diversity_mean'] < diag_results['SGD']['func_diversity_mean']
t4_paradox_in_diag = diag_muon_higher_wd and diag_muon_lower_fd
print(f"\nT4: Diagonal net does NOT have the full paradox pattern?")
print(f"    (Muon higher weight div AND lower func div = paradox)")
print(f"    Muon higher weight div: {diag_muon_higher_wd}")
print(f"    Muon lower func div:    {diag_muon_lower_fd}")
print(f"    Full paradox pattern:   {t4_paradox_in_diag}")
print(f"    --> {'PASS (no paradox in diagonal)' if not t4_paradox_in_diag else 'FAIL (paradox found in diagonal!)'}")

total_pass = sum([t1, t2, t3, not t4_paradox_in_diag])

# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 90}")
print("FINAL VERDICT")
print(f"{'=' * 90}")

print(f"""
  HYPOTHESIS: The Muon Paradox requires gauge symmetry (dim(gauge) > 0).
  It vanishes for diagonal networks where every parameter is physical.

  Tests passed: {total_pass}/4

  Diagonal net paradox strength: {diag_paradox:.4f} (expect ~1.0)
  Full matrix paradox strength:  {full_paradox:.4f} (expect >1.0)
""")

if total_pass >= 3:
    print("  VERDICT: CONFIRMED -- Paradox requires gauge symmetry.")
    print("  The diagonal net (no gauge group) shows no paradox,")
    print("  while the full matrix net (O(n) gauge group) does.")
elif total_pass >= 2:
    print("  VERDICT: PARTIALLY CONFIRMED -- Some evidence for gauge requirement.")
else:
    print("  VERDICT: REJECTED -- Paradox appears even without gauge symmetry.")
    print("  This would mean the paradox mechanism is NOT purely gauge-theoretic.")

print("=" * 90)
