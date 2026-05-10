#!/usr/bin/env python3
"""
2.20: cos(ortho_k(G), Newton direction) vs k
=============================================

HYPOTHESIS: The cosine similarity between ortho_k(G) and the Newton direction
-H^{-1}g peaks at k=3-5 then decreases. k=5 approximates natural gradient
BETTER than exact ortho (k=inf).

CONTEXT:
  - Muon uses k=5 Newton-Schulz iterations to approximate the orthogonal polar factor
  - The Newton direction -H^{-1}g is the optimal local step accounting for curvature
  - If approximate ortho (k=5) aligns BETTER with Newton than exact ortho (k=20),
    it means the warm (inexact) orthogonalization preserves curvature information
    that gets destroyed by full orthogonalization

SETUP:
  - 2-layer deep linear net (4x4, 32 total params)
  - Random target. Train to a point where loss > 0 but gradients are nonzero
  - Compute full 32x32 Hessian via finite differences
  - Compute Newton direction d_newton = -H_pinv @ g
  - For each k in {0,1,2,3,4,5,7,10,15,20}:
      Apply k NS iterations to get ortho_k(G) for each weight matrix
      Flatten to 32-vector d_muon_k
      Compute cos(d_muon_k, d_newton)
  - Repeat at 10 different training steps
  - Also compare SGD direction and Adam direction as baselines
"""

import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4                    # 4x4 matrices => 16 params per layer
NUM_LAYERS = 2             # 2-layer deep linear net => 32 total params
N_PARAMS = NUM_LAYERS * DIM * DIM  # 32
HESSIAN_EPS = 1e-5         # Finite difference step for Hessian
NS_K_VALUES = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]

# Training steps at which to measure alignment
MEASUREMENT_STEPS = [10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500]
TRAIN_LR = 0.005           # Small LR for SGD pre-training (keep gradients nonzero)
MOMENTUM = 0.9
NUM_SEEDS = 5              # Average over seeds for robustness

# Adam hyperparameters (for baseline comparison)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


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
    """Backprop through deep linear net. Returns list of gradient matrices."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    # Forward pass -- store activations
    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    # Output error
    Y_pred = activations[-1]
    delta = (Y_pred - Y_target) / batch_size

    # Backward pass
    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


# =============================================================================
# WEIGHT VECTOR UTILITIES
# =============================================================================

def weights_to_vector(weights):
    """Flatten all weight matrices into a single vector."""
    return np.concatenate([W.flatten() for W in weights])


def vector_to_weights(vec, dim, num_layers):
    """Unflatten a vector back into list of weight matrices."""
    weights = []
    idx = 0
    for _ in range(num_layers):
        size = dim * dim
        W = vec[idx:idx + size].reshape(dim, dim)
        weights.append(W)
        idx += size
    return weights


def grads_to_vector(grads):
    """Flatten gradient matrices into a single vector."""
    return np.concatenate([g.flatten() for g in grads])


# =============================================================================
# NEWTON-SCHULZ ITERATION
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters):
    """Apply Newton-Schulz iteration to approximate orthogonal polar factor.

    k=0 means just the normalized gradient (no NS iterations).
    """
    norm = np.linalg.norm(G, ord='fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X @ X.T
        X = 1.5 * X - 0.5 * A @ X

    return X


# =============================================================================
# FULL HESSIAN COMPUTATION VIA FINITE DIFFERENCES
# =============================================================================

def compute_gradient_vector(weights, X, Y_target):
    """Return gradient as a flat vector."""
    grads = compute_gradients(weights, X, Y_target)
    return grads_to_vector(grads)


def compute_full_hessian(weights, X, Y_target, eps=HESSIAN_EPS):
    """Compute full Hessian via central finite differences on the gradient."""
    theta = weights_to_vector(weights)
    n_params = len(theta)

    H = np.zeros((n_params, n_params))

    for i in range(n_params):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps

        w_plus = vector_to_weights(theta_plus, DIM, NUM_LAYERS)
        w_minus = vector_to_weights(theta_minus, DIM, NUM_LAYERS)

        grad_plus = compute_gradient_vector(w_plus, X, Y_target)
        grad_minus = compute_gradient_vector(w_minus, X, Y_target)

        H[:, i] = (grad_plus - grad_minus) / (2 * eps)

    # Symmetrize
    H = 0.5 * (H + H.T)
    return H


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def cosine_sim(a, b):
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return np.dot(a, b) / (na * nb)


# =============================================================================
# COMPUTE MUON DIRECTION FOR A GIVEN k
# =============================================================================

def compute_muon_direction(grads, k):
    """Apply k NS iterations to each gradient matrix, then flatten.

    k=0 returns normalized gradient (no NS, just Frobenius normalization per layer).
    """
    direction_parts = []
    for G in grads:
        G_orth = newton_schulz_orthogonalize(G, num_iters=k)
        direction_parts.append(G_orth.flatten())
    return np.concatenate(direction_parts)


# =============================================================================
# ADAM DIRECTION (stateless snapshot: uses current m,v state)
# =============================================================================

def compute_adam_direction(m_state, v_state, t):
    """Compute Adam update direction from current moment estimates."""
    m_hat = m_state / (1 - ADAM_BETA1 ** t)
    v_hat = v_state / (1 - ADAM_BETA2 ** t)
    direction = m_hat / (np.sqrt(v_hat) + ADAM_EPS)
    return direction


# =============================================================================
# MAIN MEASUREMENT AT ONE TRAINING CHECKPOINT
# =============================================================================

def measure_alignment(weights, X, Y_target, adam_m, adam_v, adam_t):
    """At current weights, compute Newton direction and cosine with ortho_k.

    Returns dict mapping k -> cosine, plus SGD and Adam cosines.
    """
    # 1. Compute gradient (flat vector)
    grads = compute_gradients(weights, X, Y_target)
    g = grads_to_vector(grads)
    g_norm = np.linalg.norm(g)

    if g_norm < 1e-15:
        # Gradient vanished -- skip
        return None

    # 2. Compute full Hessian
    H = compute_full_hessian(weights, X, Y_target)

    # 3. Newton direction via pseudoinverse (handles singular/gauge directions)
    H_pinv = np.linalg.pinv(H, rcond=1e-10)
    d_newton = -H_pinv @ g
    d_newton_norm = np.linalg.norm(d_newton)

    if d_newton_norm < 1e-15:
        return None

    # 4. Cosine with ortho_k(G) for each k
    cos_by_k = {}
    for k in NS_K_VALUES:
        d_muon_k = compute_muon_direction(grads, k)
        # Muon direction is negative (descent)
        cos_by_k[k] = cosine_sim(-d_muon_k, d_newton)

    # 5. SGD direction = -g (steepest descent)
    cos_sgd = cosine_sim(-g, d_newton)

    # 6. Adam direction
    if adam_t > 0:
        adam_dir = compute_adam_direction(adam_m, adam_v, adam_t)
        cos_adam = cosine_sim(-adam_dir, d_newton)
    else:
        cos_adam = cos_sgd  # At t=0, Adam = SGD

    # 7. Also compute Hessian condition info for context
    eigenvalues = np.linalg.eigvalsh(H)
    lambda_max = np.max(np.abs(eigenvalues))
    n_significant = np.sum(np.abs(eigenvalues) > 0.01 * lambda_max) if lambda_max > 1e-15 else 0

    return {
        'cos_by_k': cos_by_k,
        'cos_sgd': cos_sgd,
        'cos_adam': cos_adam,
        'loss': compute_loss(weights, X, Y_target),
        'grad_norm': g_norm,
        'newton_norm': d_newton_norm,
        'hessian_cond': lambda_max / max(np.min(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-12])), 1e-15) if np.any(np.abs(eigenvalues) > 1e-12) else np.inf,
        'hessian_rank': int(n_significant),
    }


# =============================================================================
# TRAINING LOOP WITH MEASUREMENT
# =============================================================================

def run_single_seed(seed):
    """Train with SGD, measure alignment at specified steps."""
    rng = np.random.RandomState(seed)

    # Generate random target
    W_target = [rng.randn(DIM, DIM) * 0.5 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, 32) * 0.5  # 32 data points

    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target

    # Initialize weights
    weights = init_weights(DIM, NUM_LAYERS, seed + 1000)

    # SGD momentum buffer
    velocities = [np.zeros_like(W) for W in weights]

    # Adam state (tracked in flat param space for direction computation)
    adam_m = np.zeros(N_PARAMS)
    adam_v = np.zeros(N_PARAMS)
    adam_t = 0

    # Check initial loss
    initial_loss = compute_loss(weights, X, Y_target)

    measurements = []
    max_step = max(MEASUREMENT_STEPS) + 1

    for step in range(1, max_step + 1):
        # Compute gradients
        grads = compute_gradients(weights, X, Y_target)
        g_flat = grads_to_vector(grads)

        # Update Adam state (for direction comparison only)
        adam_t += 1
        adam_m = ADAM_BETA1 * adam_m + (1 - ADAM_BETA1) * g_flat
        adam_v = ADAM_BETA2 * adam_v + (1 - ADAM_BETA2) * (g_flat ** 2)

        # SGD + momentum update
        for i in range(len(weights)):
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - TRAIN_LR * velocities[i]

        # Measure at specified steps
        if step in MEASUREMENT_STEPS:
            loss_now = compute_loss(weights, X, Y_target)
            if loss_now < 1e-12:
                # Loss too small, gradients effectively zero
                print(f"    Step {step}: loss={loss_now:.2e} (too small, skipping)")
                continue

            result = measure_alignment(weights, X, Y_target, adam_m, adam_v, adam_t)
            if result is not None:
                result['step'] = step
                measurements.append(result)
                print(f"    Step {step:5d}: loss={result['loss']:.4e}, "
                      f"|g|={result['grad_norm']:.4e}, "
                      f"H_rank={result['hessian_rank']}, "
                      f"cos(k=5,Newt)={result['cos_by_k'][5]:+.4f}")
            else:
                print(f"    Step {step:5d}: gradient or Newton direction vanished, skipping")

    return measurements


# =============================================================================
# AGGREGATION AND REPORTING
# =============================================================================

def aggregate_results(all_measurements):
    """Aggregate cosine measurements across seeds and steps.

    Returns:
      - avg_cos_by_k: dict k -> mean cosine across all measurement points
      - std_cos_by_k: dict k -> std cosine
      - avg_cos_sgd: mean cosine of SGD direction
      - avg_cos_adam: mean cosine of Adam direction
      - per_step_data: dict step -> {k -> [cosines across seeds]}
    """
    # Flatten all measurements
    all_cos_by_k = {k: [] for k in NS_K_VALUES}
    all_cos_sgd = []
    all_cos_adam = []

    per_step_data = {}

    for seed_measurements in all_measurements:
        for m in seed_measurements:
            step = m['step']
            if step not in per_step_data:
                per_step_data[step] = {k: [] for k in NS_K_VALUES}
                per_step_data[step]['sgd'] = []
                per_step_data[step]['adam'] = []

            for k in NS_K_VALUES:
                all_cos_by_k[k].append(m['cos_by_k'][k])
                per_step_data[step][k].append(m['cos_by_k'][k])

            all_cos_sgd.append(m['cos_sgd'])
            all_cos_adam.append(m['cos_adam'])
            per_step_data[step]['sgd'].append(m['cos_sgd'])
            per_step_data[step]['adam'].append(m['cos_adam'])

    avg_cos_by_k = {k: np.mean(all_cos_by_k[k]) for k in NS_K_VALUES}
    std_cos_by_k = {k: np.std(all_cos_by_k[k]) for k in NS_K_VALUES}

    return {
        'avg_cos_by_k': avg_cos_by_k,
        'std_cos_by_k': std_cos_by_k,
        'avg_cos_sgd': np.mean(all_cos_sgd),
        'std_cos_sgd': np.std(all_cos_sgd),
        'avg_cos_adam': np.mean(all_cos_adam),
        'std_cos_adam': np.std(all_cos_adam),
        'per_step_data': per_step_data,
        'n_measurements': len(all_cos_sgd),
    }


def print_separator(char='=', width=100):
    print(char * width)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print_separator('#')
    print("  EXPERIMENT 2.20: cos(ortho_k(G), Newton direction) vs k")
    print("  Does Muon's approximate ortho (k=5) align BETTER with Newton than exact ortho?")
    print_separator('#')
    print()
    print(f"  Network: {NUM_LAYERS}-layer deep linear, {DIM}x{DIM} = {N_PARAMS} params")
    print(f"  Training: SGD+momentum (lr={TRAIN_LR}, mom={MOMENTUM})")
    print(f"  Measurement steps: {MEASUREMENT_STEPS}")
    print(f"  NS k values: {NS_K_VALUES}")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"  Hessian: central finite differences (eps={HESSIAN_EPS})")
    print()

    all_measurements = []

    for seed_idx in range(NUM_SEEDS):
        seed = 42 + seed_idx * 137
        print_separator('-')
        print(f"  Seed {seed_idx+1}/{NUM_SEEDS} (seed={seed})")
        print_separator('-')
        measurements = run_single_seed(seed)
        all_measurements.append(measurements)
        print()

    # Aggregate
    agg = aggregate_results(all_measurements)

    # ==========================================================================
    # TABLE 1: Overall average cos(ortho_k, Newton) vs k
    # ==========================================================================
    print_separator('=')
    print("  TABLE 1: cos(ortho_k(G), Newton direction) averaged over ALL measurement points")
    print(f"  ({agg['n_measurements']} measurements = {NUM_SEEDS} seeds x up to {len(MEASUREMENT_STEPS)} steps)")
    print_separator('=')
    print()
    print(f"  {'k':<6s} | {'cos(ortho_k, Newton)':<24s} | {'std':<10s} | {'bar':<40s}")
    print(f"  {'-'*6}-+-{'-'*24}-+-{'-'*10}-+-{'-'*40}")

    best_k = max(NS_K_VALUES, key=lambda k: agg['avg_cos_by_k'][k])
    max_cos = max(agg['avg_cos_by_k'].values())

    for k in NS_K_VALUES:
        cos_val = agg['avg_cos_by_k'][k]
        std_val = agg['std_cos_by_k'][k]
        # Visual bar (scale 0 to max_cos)
        bar_len = int(35 * max(0, cos_val) / max(max_cos, 0.01))
        bar = '#' * bar_len
        marker = " <<< PEAK" if k == best_k else ""
        print(f"  k={k:<3d} | {cos_val:>+.6f}                | {std_val:.6f} | {bar}{marker}")

    print()
    print(f"  SGD direction (steepest descent):")
    print(f"         | {agg['avg_cos_sgd']:>+.6f}                | {agg['std_cos_sgd']:.6f} |")
    print(f"  Adam direction:")
    print(f"         | {agg['avg_cos_adam']:>+.6f}                | {agg['std_cos_adam']:.6f} |")
    print()

    # ==========================================================================
    # TABLE 2: Per-step breakdown
    # ==========================================================================
    print_separator('=')
    print("  TABLE 2: cos(ortho_k, Newton) per training step (averaged over seeds)")
    print_separator('=')
    print()

    # Header
    header = f"  {'step':<6s}"
    for k in NS_K_VALUES:
        header += f" | k={k:<3d} "
    header += f" | {'SGD':>7s} | {'Adam':>7s}"
    print(header)
    print(f"  {'-'*6}" + "".join([f"-+-{'-'*7}" for _ in NS_K_VALUES]) + f"-+-{'-'*7}-+-{'-'*7}")

    sorted_steps = sorted(agg['per_step_data'].keys())
    for step in sorted_steps:
        sd = agg['per_step_data'][step]
        row = f"  {step:<6d}"
        for k in NS_K_VALUES:
            if len(sd[k]) > 0:
                row += f" | {np.mean(sd[k]):+.4f}"
            else:
                row += f" |    N/A"
        if len(sd['sgd']) > 0:
            row += f" | {np.mean(sd['sgd']):+.4f}"
            row += f" | {np.mean(sd['adam']):+.4f}"
        else:
            row += f" |    N/A |    N/A"
        print(row)

    print()

    # ==========================================================================
    # HYPOTHESIS TEST
    # ==========================================================================
    print_separator('*')
    print("  HYPOTHESIS TEST")
    print_separator('*')
    print()

    # Test 1: Does cosine peak at k=3-5?
    cos_vals = [(k, agg['avg_cos_by_k'][k]) for k in NS_K_VALUES]
    peak_k, peak_cos = max(cos_vals, key=lambda x: x[1])

    print(f"  1. Peak cosine occurs at k={peak_k} (cos={peak_cos:+.6f})")
    if 3 <= peak_k <= 5:
        print(f"     >>> CONFIRMED: Peak is in predicted range k=3-5")
    elif 1 <= peak_k <= 7:
        print(f"     >>> PARTIAL: Peak at k={peak_k} is near but not exactly k=3-5")
    else:
        print(f"     >>> NOT CONFIRMED: Peak at k={peak_k} is outside predicted range k=3-5")
    print()

    # Test 2: Is k=5 better than k=20?
    cos_5 = agg['avg_cos_by_k'][5]
    cos_20 = agg['avg_cos_by_k'][20]
    print(f"  2. cos(ortho_5, Newton) = {cos_5:+.6f}")
    print(f"     cos(ortho_20, Newton) = {cos_20:+.6f}")
    print(f"     Difference: {cos_5 - cos_20:+.6f}")
    if cos_5 > cos_20:
        print(f"     >>> CONFIRMED: k=5 is CLOSER to Newton than k=20 (exact ortho)")
        print(f"     Interpretation: Approximate ortho preserves curvature info that exact ortho destroys")
    else:
        print(f"     >>> NOT CONFIRMED: k=20 is closer to Newton than k=5")
    print()

    # Test 3: Does cosine decline after peak?
    if peak_k < max(NS_K_VALUES):
        post_peak = [(k, agg['avg_cos_by_k'][k]) for k in NS_K_VALUES if k > peak_k]
        if len(post_peak) > 0:
            last_cos = post_peak[-1][1]
            print(f"  3. After peak (k={peak_k}): cos declines from {peak_cos:+.6f} to {last_cos:+.6f}")
            if last_cos < peak_cos:
                print(f"     >>> CONFIRMED: Cosine decreases after peak (over-orthogonalization hurts)")
            else:
                print(f"     >>> NOT CONFIRMED: Cosine does not decrease after peak")
    else:
        print(f"  3. Peak is at max k tested -- cannot assess decline")
    print()

    # Test 4: Muon vs baselines
    cos_sgd = agg['avg_cos_sgd']
    cos_adam = agg['avg_cos_adam']
    print(f"  4. Baseline comparisons:")
    print(f"     cos(SGD,  Newton) = {cos_sgd:+.6f}")
    print(f"     cos(Adam, Newton) = {cos_adam:+.6f}")
    print(f"     cos(k=5,  Newton) = {cos_5:+.6f}")
    print(f"     cos(k={peak_k} [best], Newton) = {peak_cos:+.6f}")
    print()
    if cos_5 > cos_sgd:
        print(f"     Muon (k=5) is CLOSER to Newton than SGD by {cos_5 - cos_sgd:+.6f}")
    else:
        print(f"     SGD is closer to Newton than Muon (k=5)")
    if cos_5 > cos_adam:
        print(f"     Muon (k=5) is CLOSER to Newton than Adam by {cos_5 - cos_adam:+.6f}")
    else:
        print(f"     Adam is closer to Newton than Muon (k=5)")
    print()

    # ==========================================================================
    # INTERPRETATION
    # ==========================================================================
    print_separator('#')
    print("  INTERPRETATION")
    print_separator('#')
    print()
    if peak_k <= 7 and cos_5 > cos_20:
        print("  The data supports the hypothesis: Muon's approximate orthogonalization (k~5)")
        print("  is NOT a compromise -- it is BETTER than exact orthogonalization for aligning")
        print("  with the Newton direction. This happens because:")
        print("  - At k=0: direction is raw gradient (captures curvature but is poorly conditioned)")
        print("  - At k~5: direction balances gauge-fixing with curvature preservation")
        print("  - At k=20: direction converges to exact orthogonal polar factor,")
        print("    which aggressively projects away curvature information")
        print()
        print("  This is consistent with the RG interpretation: the NS flow has an 'optimal")
        print("  stopping point' where it has removed gauge redundancy without over-projecting")
        print("  curvature-relevant gradient components.")
    elif peak_k <= 7:
        print("  Partial support: The peak IS in the low-k regime, suggesting approximate ortho")
        print("  captures something that exact ortho misses. However, the decline from k=5 to k=20")
        print("  is not significant.")
    else:
        print("  The hypothesis is NOT supported in this setting. Exact orthogonalization")
        print("  does not destroy Newton alignment. The benefit of NS iterations may be")
        print("  purely computational (convergence to orthogonal factor) rather than an")
        print("  information-theoretic sweet spot.")
    print()
    print_separator('#')
    print()


if __name__ == '__main__':
    main()
