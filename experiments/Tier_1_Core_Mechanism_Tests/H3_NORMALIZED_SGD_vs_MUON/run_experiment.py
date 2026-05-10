#!/usr/bin/env python3
"""
H3: NORMALIZED SGD vs MUON — Is the Polar Factor Necessary?
=============================================================

THE QUESTION:
  If the Muon Paradox comes from discarding gradient MAGNITUDE (scale collapse),
  then simple normalized SGD (step = G/||G||_F) should show similar paradox strength
  AND convergence benefit. The polar factor would be unnecessary — just normalizing
  the gradient norm is enough.

OPTIMIZERS COMPARED (all with momentum=0.9):
  (a) SGD — baseline
  (b) Muon — ortho of momentum via Newton-Schulz (UV^T)
  (c) Normalized SGD — step = lr * M / ||M||_F  (Frobenius normalization)
  (d) Spectral-normalized SGD — step = lr * M / ||M||_op  (spectral norm)
  (e) Sign SGD — step = lr * sign(M)  (element-wise sign)

KEY TESTS:
  T1: Does normalized SGD create the paradox?
  T2: Does normalized SGD match Muon's convergence speed?
  T3: Does Muon beat normalized SGD on LOSS?
  T4: Which normalization creates the strongest paradox?

Setup: 4-layer nets (32x32), quadratic loss, 20 independent runs.
Architectures: deep linear + ReLU.
"""

import numpy as np
import os
import time

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
BATCH_SIZE = 64
MOMENTUM = 0.9
NS_ITERS = 5
NUM_INDEPENDENT_RUNS = 20
NUM_TEST_INPUTS = 50

# Fixed target and data
W_target = np.random.randn(DIM, DIM) * 0.5
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3
X_test = np.random.randn(DIM, NUM_TEST_INPUTS) * 0.3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OPTIMIZER_NAMES = ['sgd', 'muon', 'norm_sgd', 'spectral_sgd', 'sign_sgd']
OPTIMIZER_LABELS = {
    'sgd': 'SGD',
    'muon': 'Muon (UV^T)',
    'norm_sgd': 'Normalized SGD (Frob)',
    'spectral_sgd': 'Spectral-Norm SGD',
    'sign_sgd': 'Sign SGD',
}
OPTIMIZER_COLORS = {
    'sgd': '#4477AA',
    'muon': '#CC3311',
    'norm_sgd': '#228B22',
    'spectral_sgd': '#9933CC',
    'sign_sgd': '#FF8800',
}


# =============================================================================
# NETWORK DEFINITIONS
# =============================================================================

def init_weights(num_layers, seed=42):
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


# ---- DEEP LINEAR NET ----

def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss_linear(weights, X, target):
    pred = forward_linear(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_linear(weights, X, target):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())
    target_out = target @ X
    delta = (activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
    return grads


# ---- RELU NET ----

def forward_relu(weights, X):
    out = X.copy()
    for idx, W in enumerate(weights):
        out = W @ out
        if idx < len(weights) - 1:
            out = np.maximum(0, out)
    return out


def compute_loss_relu(weights, X, target):
    pred = forward_relu(weights, X)
    target_out = target @ X
    diff = pred - target_out
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_relu(weights, X, target):
    num_layers = len(weights)
    N = X.shape[1]
    pre_activations = []
    post_activations = [X.copy()]
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_activations.append(pre.copy())
        if idx < num_layers - 1:
            out = np.maximum(0, pre)
        else:
            out = pre
        post_activations.append(out.copy())
    target_out = target @ X
    delta = (post_activations[-1] - target_out) / N
    grads = []
    for i in range(num_layers - 1, -1, -1):
        G = delta @ post_activations[i].T
        grads.insert(0, G)
        if i > 0:
            delta = weights[i].T @ delta
            delta = delta * (pre_activations[i - 1] > 0).astype(float)
    return grads


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=NS_ITERS):
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def optimizer_step(weights, velocities, grads, lr, method):
    """Unified optimizer step for all 5 methods."""
    for i in range(len(weights)):
        if method == 'sgd':
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

        elif method == 'muon':
            # Muon: orthogonalize the gradient, then apply momentum
            ortho_grad = newton_schulz_orthogonalize(grads[i])
            velocities[i] = MOMENTUM * velocities[i] + ortho_grad
            weights[i] = weights[i] - lr * velocities[i]

        elif method == 'norm_sgd':
            # Standard momentum, then normalize momentum to unit Frobenius
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            v_norm = np.linalg.norm(velocities[i], 'fro')
            if v_norm > 1e-12:
                step = velocities[i] / v_norm
            else:
                step = velocities[i]
            weights[i] = weights[i] - lr * step

        elif method == 'spectral_sgd':
            # Standard momentum, then normalize to unit spectral norm
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            spec_norm = np.linalg.norm(velocities[i], ord=2)
            if spec_norm > 1e-12:
                step = velocities[i] / spec_norm
            else:
                step = velocities[i]
            weights[i] = weights[i] - lr * step

        elif method == 'sign_sgd':
            # Standard momentum, then take element-wise sign
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            step = np.sign(velocities[i])
            weights[i] = weights[i] - lr * step

    return weights, velocities


# =============================================================================
# LEARNING RATE SWEEP
# =============================================================================

def find_best_lr(method, net_type, num_steps=200):
    """
    Sweep over LR candidates for each method. Return the LR achieving the
    lowest final loss without diverging.
    """
    compute_loss_fn = compute_loss_linear if net_type == 'linear' else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == 'linear' else compute_gradients_relu

    # Different LR ranges for different methods
    if method == 'sgd':
        candidates = [0.1, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
    elif method == 'muon':
        candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
    elif method == 'norm_sgd':
        candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001, 0.0005]
    elif method == 'spectral_sgd':
        candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001, 0.0005]
    elif method == 'sign_sgd':
        # Sign SGD needs much smaller LR (each element is +/-1 times dim*dim)
        candidates = [0.005, 0.003, 0.002, 0.001, 0.0007, 0.0005, 0.0003, 0.0002, 0.0001, 0.00005]

    best_lr = candidates[-1]
    best_loss = float('inf')

    for lr_cand in candidates:
        np.random.seed(42)
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss_fn(weights, X_data, W_target)
        stable = True
        final_loss = initial_loss

        for step in range(num_steps):
            grads = compute_grad_fn(weights, X_data, W_target)
            weights, velocities = optimizer_step(weights, velocities, grads, lr_cand, method)
            loss = compute_loss_fn(weights, X_data, W_target)
            if np.isnan(loss) or loss > initial_loss * 50:
                stable = False
                break
            final_loss = loss

        if stable and final_loss < best_loss:
            best_loss = final_loss
            best_lr = lr_cand

    return best_lr, best_loss


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def run_training(weights_init, method, lr, num_steps, net_type):
    """
    Run optimizer for num_steps from weights_init.
    Returns loss curve and final weights.
    """
    compute_loss_fn = compute_loss_linear if net_type == 'linear' else compute_loss_relu
    compute_grad_fn = compute_gradients_linear if net_type == 'linear' else compute_gradients_relu

    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]
    losses = [compute_loss_fn(weights, X_data, W_target)]

    for step in range(num_steps):
        grads = compute_grad_fn(weights, X_data, W_target)
        weights, velocities = optimizer_step(weights, velocities, grads, lr, method)
        loss = compute_loss_fn(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            # Pad remaining with NaN
            for _ in range(num_steps - step - 1):
                losses.append(np.nan)
            break

    return np.array(losses), weights


# =============================================================================
# CONVERGENCE BASIN ANALYSIS (20 independent runs)
# =============================================================================

def measure_convergence_basin(method, lr, net_type, num_runs=NUM_INDEPENDENT_RUNS,
                              num_steps=NUM_STEPS):
    """
    Run num_runs independent initializations. Measure:
      - Weight diversity: mean pairwise ||W_i - W_j||_F
      - Function diversity: mean pairwise ||f(X_test;W_i) - f(X_test;W_j)||_F
      - Loss mean and std
      - Per-layer condition number at convergence
    """
    forward_fn = forward_linear if net_type == 'linear' else forward_relu
    compute_loss_fn = compute_loss_linear if net_type == 'linear' else compute_loss_relu

    final_weights_list = []
    final_functions = []
    final_losses = []
    loss_curves = []
    condition_numbers = []  # per-layer condition numbers

    for run_idx in range(num_runs):
        weights_init = init_weights(NUM_LAYERS, seed=1000 + run_idx)
        loss_curve, final_weights = run_training(
            weights_init, method, lr, num_steps, net_type)

        loss_curves.append(loss_curve)
        final_weights_list.append(final_weights)
        final_functions.append(forward_fn(final_weights, X_test).copy())
        final_losses.append(compute_loss_fn(final_weights, X_data, W_target))

        # Per-layer condition number
        cond_per_layer = []
        for W in final_weights:
            svs = np.linalg.svd(W, compute_uv=False)
            if svs[-1] > 1e-15:
                cond_per_layer.append(svs[0] / svs[-1])
            else:
                cond_per_layer.append(np.inf)
        condition_numbers.append(cond_per_layer)

    # Pairwise diversity
    n = len(final_weights_list)
    weight_dists = []
    func_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d_w = 0.0
            for k in range(NUM_LAYERS):
                d_w += np.linalg.norm(
                    final_weights_list[i][k] - final_weights_list[j][k], 'fro') ** 2
            weight_dists.append(np.sqrt(d_w))
            d_f = np.linalg.norm(
                final_functions[i] - final_functions[j], 'fro') / np.linalg.norm(X_test, 'fro')
            func_dists.append(d_f)

    # Mean condition number per layer
    cond_arr = np.array(condition_numbers)  # shape (num_runs, NUM_LAYERS)
    mean_cond = np.mean(cond_arr, axis=0)

    # Mean loss curve (handle NaN)
    max_len = max(len(lc) for lc in loss_curves)
    padded = np.full((num_runs, max_len), np.nan)
    for i, lc in enumerate(loss_curves):
        padded[i, :len(lc)] = lc
    mean_loss_curve = np.nanmean(padded, axis=0)

    return {
        'weight_diversity_mean': np.mean(weight_dists),
        'weight_diversity_std': np.std(weight_dists),
        'func_diversity_mean': np.mean(func_dists),
        'func_diversity_std': np.std(func_dists),
        'loss_mean': np.mean(final_losses),
        'loss_std': np.std(final_losses),
        'losses': np.array(final_losses),
        'weight_dists': np.array(weight_dists),
        'func_dists': np.array(func_dists),
        'mean_loss_curve': mean_loss_curve,
        'mean_cond_per_layer': mean_cond,
        'cond_all': cond_arr,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 110)
print("H3: NORMALIZED SGD vs MUON — Is the Polar Factor Necessary?")
print("=" * 110)
print(f"Setup: {NUM_LAYERS}-layer nets (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"{NUM_INDEPENDENT_RUNS} independent inits per optimizer, momentum={MOMENTUM}")
print(f"Optimizers: SGD, Muon, Normalized SGD (Frob), Spectral-Norm SGD, Sign SGD")
print("=" * 110)

all_results = {}

for net_type in ['linear', 'relu']:
    print(f"\n{'#' * 110}")
    print(f"  ARCHITECTURE: {net_type.upper()} ({NUM_LAYERS}-layer, {DIM}x{DIM})")
    print(f"{'#' * 110}")

    # Phase 1: LR sweep for each optimizer
    print(f"\n  --- Phase 1: Learning Rate Sweep ---")
    best_lrs = {}
    for method in OPTIMIZER_NAMES:
        lr, loss = find_best_lr(method, net_type, num_steps=200)
        best_lrs[method] = lr
        print(f"    {OPTIMIZER_LABELS[method]:<30} best_lr={lr:.5f}  (200-step loss={loss:.6e})")

    # Phase 2: Full training + convergence basin analysis (20 runs)
    print(f"\n  --- Phase 2: Convergence Basin ({NUM_INDEPENDENT_RUNS} runs, {NUM_STEPS} steps) ---")
    net_results = {}
    for method in OPTIMIZER_NAMES:
        t0 = time.time()
        result = measure_convergence_basin(method, best_lrs[method], net_type)
        elapsed = time.time() - t0
        net_results[method] = result
        net_results[method]['lr'] = best_lrs[method]
        print(f"    {OPTIMIZER_LABELS[method]:<30} loss={result['loss_mean']:.6e} +/- {result['loss_std']:.6e}  "
              f"d_w={result['weight_diversity_mean']:.4f}  d_f={result['func_diversity_mean']:.6f}  "
              f"({elapsed:.1f}s)")

    all_results[net_type] = net_results


# =============================================================================
# RESULTS TABLES
# =============================================================================

for net_type in ['linear', 'relu']:
    res = all_results[net_type]

    print(f"\n\n{'=' * 110}")
    print(f"RESULTS TABLE: {net_type.upper()} NET")
    print(f"{'=' * 110}")

    # ---- Convergence & Loss ----
    print(f"\n  {'Optimizer':<30} | {'LR':>8} | {'Final Loss':>14} | {'Loss Std':>14} | {'Mean Loss Curve End':>18}")
    print(f"  {'-' * 100}")
    for m in OPTIMIZER_NAMES:
        r = res[m]
        curve_end = r['mean_loss_curve'][-1] if len(r['mean_loss_curve']) > 0 else np.nan
        print(f"  {OPTIMIZER_LABELS[m]:<30} | {r['lr']:>8.5f} | {r['loss_mean']:>14.6e} | "
              f"{r['loss_std']:>14.6e} | {curve_end:>18.6e}")

    # ---- Paradox Metrics ----
    print(f"\n  {'Optimizer':<30} | {'Weight Div':>12} | {'Func Div':>12} | {'F/W Ratio':>12} | {'Loss Std':>14}")
    print(f"  {'-' * 90}")
    for m in OPTIMIZER_NAMES:
        r = res[m]
        ratio = r['func_diversity_mean'] / r['weight_diversity_mean'] if r['weight_diversity_mean'] > 1e-15 else np.nan
        print(f"  {OPTIMIZER_LABELS[m]:<30} | {r['weight_diversity_mean']:>12.4f} | {r['func_diversity_mean']:>12.6f} | "
              f"{ratio:>12.6f} | {r['loss_std']:>14.6e}")

    # ---- Per-Layer Condition Numbers ----
    print(f"\n  {'Optimizer':<30} |", end='')
    for l in range(NUM_LAYERS):
        print(f" {'Layer '+str(l):>10} |", end='')
    print(f" {'Geom Mean':>10}")
    print(f"  {'-' * (35 + 13 * NUM_LAYERS + 12)}")
    for m in OPTIMIZER_NAMES:
        r = res[m]
        conds = r['mean_cond_per_layer']
        geo_mean = np.exp(np.mean(np.log(np.clip(conds, 1e-15, None))))
        print(f"  {OPTIMIZER_LABELS[m]:<30} |", end='')
        for l in range(NUM_LAYERS):
            print(f" {conds[l]:>10.2f} |", end='')
        print(f" {geo_mean:>10.2f}")


# =============================================================================
# KEY HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 110}")
print("KEY HYPOTHESIS TESTS")
print("=" * 110)

total_pass = 0
total_tests = 0
all_test_results = {}

for net_type in ['linear', 'relu']:
    res = all_results[net_type]
    print(f"\n  {'=' * 80}")
    print(f"  {net_type.upper()} NET")
    print(f"  {'=' * 80}")

    # Helper: compute paradox ratio for a method
    def paradox_ratio(m):
        r = res[m]
        if r['weight_diversity_mean'] > 1e-15:
            return r['func_diversity_mean'] / r['weight_diversity_mean']
        return np.nan

    # Helper: paradox strength = weight_div * (1 / func_div) * (1 / loss_std)
    # Higher = stronger paradox (more weight diversity, less function diversity, less loss spread)
    def paradox_strength(m):
        r = res[m]
        if r['func_diversity_mean'] > 1e-15 and r['loss_std'] > 1e-20:
            return r['weight_diversity_mean'] / (r['func_diversity_mean'] * r['loss_std'])
        return 0.0

    # ----- T1: Does normalized SGD create the paradox? -----
    # Paradox = high weight diversity AND low loss std AND low func diversity
    # Compare to SGD baseline
    sgd_ratio = paradox_ratio('sgd')
    norm_ratio = paradox_ratio('norm_sgd')
    muon_ratio = paradox_ratio('muon')

    t1 = norm_ratio < sgd_ratio  # Lower ratio means stronger paradox
    total_tests += 1
    total_pass += t1
    print(f"\n  T1: Does Normalized SGD create the paradox?")
    print(f"      F/W Ratio: SGD={sgd_ratio:.6f}, Norm_SGD={norm_ratio:.6f}, Muon={muon_ratio:.6f}")
    print(f"      Norm SGD ratio < SGD ratio? {norm_ratio:.6f} < {sgd_ratio:.6f} --> {'PASS' if t1 else 'FAIL'}")

    # ----- T2: Does normalized SGD match Muon convergence speed? -----
    # Compare mean loss curves at 50%, 100% of steps
    muon_curve = res['muon']['mean_loss_curve']
    norm_curve = res['norm_sgd']['mean_loss_curve']
    half = len(muon_curve) // 2
    # Match = within 2x of each other at 50% and 100%
    muon_half = muon_curve[half] if half < len(muon_curve) else np.nan
    norm_half = norm_curve[half] if half < len(norm_curve) else np.nan
    muon_final = res['muon']['loss_mean']
    norm_final = res['norm_sgd']['loss_mean']
    # Within 3x is a rough match
    t2_half = (min(muon_half, norm_half) / max(muon_half, norm_half) > 0.33) if (muon_half > 1e-15 and norm_half > 1e-15) else False
    t2_final = (min(muon_final, norm_final) / max(muon_final, norm_final) > 0.33) if (muon_final > 1e-15 and norm_final > 1e-15) else False
    t2 = t2_half and t2_final
    total_tests += 1
    total_pass += t2
    print(f"\n  T2: Does Normalized SGD match Muon convergence speed?")
    print(f"      At 50% steps: Muon={muon_half:.6e}, Norm={norm_half:.6e}, ratio={min(muon_half,norm_half)/max(muon_half,norm_half):.3f}")
    print(f"      Final loss:   Muon={muon_final:.6e}, Norm={norm_final:.6e}, ratio={min(muon_final,norm_final)/max(muon_final,norm_final):.3f}")
    print(f"      --> {'PASS (comparable)' if t2 else 'FAIL (not comparable)'}")

    # ----- T3: Does Muon beat normalized SGD on LOSS? -----
    t3 = muon_final < norm_final
    total_tests += 1
    total_pass += t3
    print(f"\n  T3: Does Muon beat Normalized SGD on final loss?")
    print(f"      Muon={muon_final:.6e} vs Norm={norm_final:.6e}")
    print(f"      --> {'PASS (Muon wins)' if t3 else 'FAIL (Norm wins or tie)'}")

    # ----- T4: Which normalization creates strongest paradox? -----
    norm_methods = ['norm_sgd', 'spectral_sgd', 'sign_sgd', 'muon']
    ratios = {m: paradox_ratio(m) for m in norm_methods}
    strengths = {m: paradox_strength(m) for m in norm_methods}
    best_ratio = min(ratios, key=lambda m: ratios[m])
    best_strength = max(strengths, key=lambda m: strengths[m])
    total_tests += 1
    t4 = best_ratio == 'muon'
    total_pass += t4
    print(f"\n  T4: Which normalization creates strongest paradox?")
    print(f"      F/W Ratios (lower = stronger paradox):")
    for m in norm_methods:
        marker = " <-- BEST" if m == best_ratio else ""
        print(f"        {OPTIMIZER_LABELS[m]:<30} ratio={ratios[m]:.6f}{marker}")
    print(f"      Paradox Strength (weight_div / (func_div * loss_std)):")
    for m in norm_methods:
        marker = " <-- BEST" if m == best_strength else ""
        print(f"        {OPTIMIZER_LABELS[m]:<30} strength={strengths[m]:.2f}{marker}")
    print(f"      Muon has lowest F/W ratio? --> {'PASS' if t4 else 'FAIL'}")

    all_test_results[net_type] = {
        't1': t1, 't2': t2, 't3': t3, 't4': t4,
        'ratios': ratios, 'strengths': strengths,
    }


# =============================================================================
# THE CRITICAL COMPARISON
# =============================================================================

print(f"\n\n{'=' * 110}")
print("THE CRITICAL COMPARISON: Is the Polar Factor Necessary?")
print("=" * 110)

for net_type in ['linear', 'relu']:
    res = all_results[net_type]
    tr = all_test_results[net_type]

    print(f"\n  --- {net_type.upper()} NET ---")

    muon_loss = res['muon']['loss_mean']
    norm_loss = res['norm_sgd']['loss_mean']
    spec_loss = res['spectral_sgd']['loss_mean']

    muon_ratio = tr['ratios']['muon']
    norm_ratio = tr['ratios']['norm_sgd']
    spec_ratio = tr['ratios']['spectral_sgd']
    sgd_ratio = res['sgd']['func_diversity_mean'] / res['sgd']['weight_diversity_mean'] if res['sgd']['weight_diversity_mean'] > 1e-15 else np.nan

    # Check if norm/spectral SGD match Muon on BOTH paradox AND loss
    norm_matches_paradox = norm_ratio < sgd_ratio * 0.8  # At least 20% better than SGD
    norm_matches_loss = abs(norm_loss - muon_loss) / max(muon_loss, 1e-15) < 0.5  # Within 50%
    spec_matches_paradox = spec_ratio < sgd_ratio * 0.8
    spec_matches_loss = abs(spec_loss - muon_loss) / max(muon_loss, 1e-15) < 0.5

    any_matches_both = (norm_matches_paradox and norm_matches_loss) or (spec_matches_paradox and spec_matches_loss)
    any_matches_paradox_only = (norm_matches_paradox or spec_matches_paradox) and not any_matches_both

    print(f"\n    SGD baseline F/W ratio:        {sgd_ratio:.6f}")
    print(f"    Muon F/W ratio:                {muon_ratio:.6f}  loss={muon_loss:.6e}")
    print(f"    Norm SGD F/W ratio:            {norm_ratio:.6f}  loss={norm_loss:.6e}")
    print(f"    Spectral SGD F/W ratio:        {spec_ratio:.6f}  loss={spec_loss:.6e}")
    print(f"    Sign SGD F/W ratio:            {tr['ratios']['sign_sgd']:.6f}  loss={res['sign_sgd']['loss_mean']:.6e}")

    print(f"\n    Norm SGD paradox?  {norm_matches_paradox}  (ratio < 80% of SGD)")
    print(f"    Norm SGD loss match?  {norm_matches_loss}  (within 50% of Muon)")
    print(f"    Spec SGD paradox?  {spec_matches_paradox}")
    print(f"    Spec SGD loss match?  {spec_matches_loss}")

    if any_matches_both:
        print(f"\n    ==> CONCLUSION: Polar factor is UNNECESSARY.")
        print(f"        Simple normalization suffices for both paradox and convergence.")
        print(f"        The key mechanism is scale removal, not orthogonal projection.")
    elif any_matches_paradox_only or (norm_matches_paradox and not norm_matches_loss) or (spec_matches_paradox and not spec_matches_loss):
        print(f"\n    ==> CONCLUSION: Normalization creates the paradox but DIRECTIONAL QUALITY matters.")
        print(f"        The polar factor provides better convergence than simple normalization.")
        print(f"        Scale removal creates gauge exploration; ortho projection provides gradient quality.")
    else:
        print(f"\n    ==> CONCLUSION: Polar factor is doing something QUALITATIVELY DIFFERENT.")
        print(f"        Simple normalization does NOT replicate the Muon paradox or convergence.")
        print(f"        The orthogonal projection is essential, not just the normalization.")


# =============================================================================
# COMPREHENSIVE NUMBERS DUMP
# =============================================================================

print(f"\n\n{'=' * 110}")
print("COMPREHENSIVE NUMBERS (for paper)")
print("=" * 110)

for net_type in ['linear', 'relu']:
    res = all_results[net_type]
    print(f"\n  {net_type.upper()} NET:")
    print(f"  {'Method':<30} | {'Loss':>12} | {'Loss Std':>12} | {'W Div':>10} | {'F Div':>12} | "
          f"{'F/W Ratio':>10} | {'Cond(geom)':>10} | {'Paradox Str':>12}")
    print(f"  {'-' * 125}")
    for m in OPTIMIZER_NAMES:
        r = res[m]
        ratio = r['func_diversity_mean'] / r['weight_diversity_mean'] if r['weight_diversity_mean'] > 1e-15 else np.nan
        conds = r['mean_cond_per_layer']
        geo_cond = np.exp(np.mean(np.log(np.clip(conds, 1e-15, None))))
        if r['func_diversity_mean'] > 1e-15 and r['loss_std'] > 1e-20:
            pstr = r['weight_diversity_mean'] / (r['func_diversity_mean'] * r['loss_std'])
        else:
            pstr = 0.0
        print(f"  {OPTIMIZER_LABELS[m]:<30} | {r['loss_mean']:>12.6e} | {r['loss_std']:>12.6e} | "
              f"{r['weight_diversity_mean']:>10.4f} | {r['func_diversity_mean']:>12.6f} | "
              f"{ratio:>10.6f} | {geo_cond:>10.2f} | {pstr:>12.2f}")


# =============================================================================
# PLOTS
# =============================================================================

print(f"\n\n{'=' * 110}")
print("GENERATING PLOTS")
print("=" * 110)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for net_type in ['linear', 'relu']:
        res = all_results[net_type]

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(
            f'H3: Normalized SGD vs Muon ({net_type.upper()} net)\n'
            f'{NUM_LAYERS}-layer, dim={DIM}, {NUM_STEPS} steps, {NUM_INDEPENDENT_RUNS} runs',
            fontsize=14, fontweight='bold')

        # ---- (a) Loss Curves ----
        ax = axes[0, 0]
        ax.set_title('Mean Loss Curves (Best LR)')
        for m in OPTIMIZER_NAMES:
            curve = res[m]['mean_loss_curve']
            ax.semilogy(np.arange(len(curve)), curve, color=OPTIMIZER_COLORS[m],
                        linewidth=2, label=f"{OPTIMIZER_LABELS[m]} (lr={res[m]['lr']:.4f})")
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- (b) Final Loss Bar Chart ----
        ax = axes[0, 1]
        ax.set_title('Final Loss (mean +/- std)')
        x_pos = np.arange(len(OPTIMIZER_NAMES))
        means = [res[m]['loss_mean'] for m in OPTIMIZER_NAMES]
        stds = [res[m]['loss_std'] for m in OPTIMIZER_NAMES]
        colors = [OPTIMIZER_COLORS[m] for m in OPTIMIZER_NAMES]
        bars = ax.bar(x_pos, means, yerr=stds, color=colors, edgecolor='black',
                      capsize=3, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([OPTIMIZER_LABELS[m] for m in OPTIMIZER_NAMES],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Final Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{means[i]:.2e}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        # ---- (c) Paradox Ratio (F/W) ----
        ax = axes[0, 2]
        ax.set_title('Paradox Ratio: Func_Div / Weight_Div\n(Lower = stronger paradox)')
        ratios = []
        for m in OPTIMIZER_NAMES:
            r = res[m]
            if r['weight_diversity_mean'] > 1e-15:
                ratios.append(r['func_diversity_mean'] / r['weight_diversity_mean'])
            else:
                ratios.append(0)
        bars = ax.bar(x_pos, ratios, color=colors, edgecolor='black', width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([OPTIMIZER_LABELS[m] for m in OPTIMIZER_NAMES],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Func Div / Weight Div')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{ratios[i]:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # ---- (d) Weight Diversity ----
        ax = axes[1, 0]
        ax.set_title('Weight Diversity (pairwise)')
        wdivs = [res[m]['weight_diversity_mean'] for m in OPTIMIZER_NAMES]
        bars = ax.bar(x_pos, wdivs, color=colors, edgecolor='black', width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([OPTIMIZER_LABELS[m] for m in OPTIMIZER_NAMES],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean Pairwise ||W_i - W_j||_F')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{wdivs[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # ---- (e) Function Diversity ----
        ax = axes[1, 1]
        ax.set_title('Function Diversity (pairwise)')
        fdivs = [res[m]['func_diversity_mean'] for m in OPTIMIZER_NAMES]
        bars = ax.bar(x_pos, fdivs, color=colors, edgecolor='black', width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([OPTIMIZER_LABELS[m] for m in OPTIMIZER_NAMES],
                           rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean Pairwise ||f_i - f_j||_F / ||X||_F')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{fdivs[i]:.5f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        # ---- (f) Condition Number ----
        ax = axes[1, 2]
        ax.set_title('Per-Layer Condition Number (mean)')
        x_layers = np.arange(NUM_LAYERS)
        width_bar = 0.15
        for idx, m in enumerate(OPTIMIZER_NAMES):
            conds = res[m]['mean_cond_per_layer']
            offset = (idx - len(OPTIMIZER_NAMES) / 2 + 0.5) * width_bar
            ax.bar(x_layers + offset, conds, width_bar, color=OPTIMIZER_COLORS[m],
                   edgecolor='black', label=OPTIMIZER_LABELS[m])
        ax.set_xticks(x_layers)
        ax.set_xticklabels([f'Layer {l}' for l in range(NUM_LAYERS)])
        ax.set_ylabel('Condition Number')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(SCRIPT_DIR, f'h3_normalized_sgd_vs_muon_{net_type}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {plot_path}")

    # ---- Combined summary ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('H3: Is the Polar Factor Necessary?\n'
                 'Paradox Ratio (F_div / W_div) — Lower = Stronger Paradox',
                 fontsize=14, fontweight='bold')

    for idx, net_type in enumerate(['linear', 'relu']):
        res = all_results[net_type]
        ax = axes[idx]
        ax.set_title(f'{net_type.upper()} Net')

        ratios = []
        for m in OPTIMIZER_NAMES:
            r = res[m]
            if r['weight_diversity_mean'] > 1e-15:
                ratios.append(r['func_diversity_mean'] / r['weight_diversity_mean'])
            else:
                ratios.append(0)

        x_pos = np.arange(len(OPTIMIZER_NAMES))
        colors = [OPTIMIZER_COLORS[m] for m in OPTIMIZER_NAMES]
        bars = ax.bar(x_pos, ratios, color=colors, edgecolor='black', width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([OPTIMIZER_LABELS[m] for m in OPTIMIZER_NAMES],
                           rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Func Diversity / Weight Diversity')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{ratios[i]:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add loss info as text
        loss_text = "Final Losses:\n"
        for m in OPTIMIZER_NAMES:
            loss_text += f"  {OPTIMIZER_LABELS[m]}: {res[m]['loss_mean']:.2e}\n"
        ax.text(0.98, 0.98, loss_text, transform=ax.transAxes, fontsize=7,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    combined_path = os.path.join(SCRIPT_DIR, 'h3_combined_summary.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined plot saved: {combined_path}")

except ImportError:
    print("  WARNING: matplotlib not available, skipping plots.")


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 110}")
print("FINAL VERDICT")
print("=" * 110)

for net_type in ['linear', 'relu']:
    tr = all_test_results[net_type]
    res = all_results[net_type]
    print(f"\n  {net_type.upper()} NET:")
    tests = [('T1', tr['t1'], 'Norm SGD creates paradox (lower F/W ratio than SGD)'),
             ('T2', tr['t2'], 'Norm SGD matches Muon convergence speed'),
             ('T3', tr['t3'], 'Muon beats Norm SGD on final loss'),
             ('T4', tr['t4'], 'Muon has strongest paradox (lowest F/W ratio)')]
    for tname, tval, desc in tests:
        print(f"    {tname}: {'PASS' if tval else 'FAIL'}  -- {desc}")
    n_pass = sum(1 for _, v, _ in tests if v)
    print(f"    Total: {n_pass}/4 tests passed")

all_pass = sum(1 for net_type in ['linear', 'relu']
               for v in [all_test_results[net_type]['t1'], all_test_results[net_type]['t2'],
                         all_test_results[net_type]['t3'], all_test_results[net_type]['t4']] if v)

print(f"\n  Overall: {all_pass}/8 tests passed across both architectures")

# Determine the paper thesis
print(f"\n  {'=' * 80}")
print(f"  PAPER THESIS DETERMINATION:")
print(f"  {'=' * 80}")

# Check the dominant pattern
both_paradox = all(all_test_results[nt]['t1'] for nt in ['linear', 'relu'])
both_conv = all(all_test_results[nt]['t2'] for nt in ['linear', 'relu'])
muon_wins_loss = all(all_test_results[nt]['t3'] for nt in ['linear', 'relu'])
muon_wins_paradox = all(all_test_results[nt]['t4'] for nt in ['linear', 'relu'])

if both_paradox and both_conv:
    print(f"  Normalization alone creates the paradox AND matches Muon convergence.")
    print(f"  ==> The polar factor is NOT necessary. Scale removal is the key mechanism.")
elif both_paradox and muon_wins_loss:
    print(f"  Normalization creates the paradox BUT Muon wins on loss.")
    print(f"  ==> The paradox comes from scale removal, but the polar factor provides")
    print(f"      superior gradient DIRECTION quality for faster convergence.")
    print(f"  ==> TWO-MECHANISM STORY: scale removal for gauge exploration,")
    print(f"      orthogonal projection for convergence quality.")
elif both_paradox:
    print(f"  Normalization creates the paradox. Mixed results on convergence.")
    print(f"  ==> Scale removal is sufficient for the paradox mechanism.")
elif muon_wins_paradox:
    print(f"  Only Muon creates strong paradox. The polar factor is essential.")
    print(f"  ==> The orthogonal projection does something qualitatively different")
    print(f"      from simple normalization.")
else:
    print(f"  Mixed results. The picture is more nuanced than any simple thesis.")

print(f"\n{'=' * 110}")
print("EXPERIMENT COMPLETE")
print(f"{'=' * 110}")
