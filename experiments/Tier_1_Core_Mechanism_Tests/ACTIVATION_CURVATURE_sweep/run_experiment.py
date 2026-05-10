#!/usr/bin/env python3
"""
Experiment 2.13: Activation Curvature Theory
==============================================

HYPOTHESIS:
  The ortho penalty recovery percentage (how much of Muon's advantage
  SGD+penalty recovers) depends on activation curvature. Higher curvature
  activations make the landscape harder for the penalty approach, so
  recovery % decreases as activation curvature increases.

SETUP:
  - 6 activations: Linear, ReLU, LeakyReLU(0.1), GELU, Tanh, Sigmoid
  - 4-layer net, width 32, 500 steps
  - Four optimizers:
    (a) SGD -- baseline
    (b) Muon -- orthogonalized gradient steps via Newton-Schulz
    (c) SGD + weight ortho penalty (lambda=0.003) on ||W^T W - I||_F^2
    (d) SGD + step ortho penalty -- blend SGD direction with its polar factor:
        step = (1-alpha)*g + alpha*polar(g), sweep alpha to find best

  - Compute recovery %:
      recovery = (loss_SGD - loss_method) / (loss_SGD - loss_Muon) * 100
  - Estimate mean |f''(x)| numerically for each activation
  - Key test: is recovery % monotonically decreasing with curvature?
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

WIDTH = 32
DEPTH = 4
NUM_STEPS = 500
LR_SGD = 0.01
LR_MUON = 0.02
ORTHO_LAMBDA = 0.003
NS_ITERS = 5
BATCH_SIZE = 64
INPUT_DIM = 32
OUTPUT_DIM = 32
SEED = 42

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def act_linear(x):
    return x.copy()

def act_relu(x):
    return np.maximum(0, x)

def act_leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def act_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def act_tanh(x):
    return np.tanh(x)

def act_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def dact_linear(x):
    return np.ones_like(x)

def dact_relu(x):
    return (x > 0).astype(float)

def dact_leaky_relu(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)

def dact_gelu(x):
    eps = 1e-5
    return (act_gelu(x + eps) - act_gelu(x - eps)) / (2 * eps)

def dact_tanh(x):
    return 1.0 - np.tanh(x)**2

def dact_sigmoid(x):
    s = act_sigmoid(x)
    return s * (1.0 - s)


ACTIVATIONS = {
    'Linear':        (act_linear, dact_linear),
    'ReLU':          (act_relu, dact_relu),
    'LeakyReLU(0.1)':(act_leaky_relu, dact_leaky_relu),
    'GELU':          (act_gelu, dact_gelu),
    'Tanh':          (act_tanh, dact_tanh),
    'Sigmoid':       (act_sigmoid, dact_sigmoid),
}

# =============================================================================
# COMPUTE MEAN |f''(x)| NUMERICALLY
# =============================================================================

def estimate_mean_second_derivative(act_fn, n_samples=10000, seed=42):
    """Numerically estimate mean |f''(x)| over typical inputs N(0,1)."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples)
    h = 1e-4
    fpp = (act_fn(x + h) - 2.0 * act_fn(x) + act_fn(x - h)) / (h * h)
    return np.mean(np.abs(fpp))


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(num_layers, width, seed):
    rng = np.random.RandomState(seed)
    weights = []
    for _ in range(num_layers):
        std = np.sqrt(2.0 / (width + width))
        W = rng.randn(width, width) * std
        weights.append(W.copy())
    return weights


def forward(weights, X, act_fn):
    activations = [X.copy()]
    pre_activations = []
    out = X.copy()
    for W in weights:
        z = W @ out
        pre_activations.append(z)
        out = act_fn(z)
        activations.append(out)
    return activations, pre_activations


def compute_loss(weights, X, Y_target, act_fn):
    activations, _ = forward(weights, X, act_fn)
    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)


def compute_gradients(weights, X, Y_target, act_fn, dact_fn):
    num_layers = len(weights)
    batch_size = X.shape[1]
    activations, pre_activations = forward(weights, X, act_fn)
    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    delta = diff / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        act_deriv = dact_fn(pre_activations[l])
        delta_z = delta * act_deriv
        grads[l] = delta_z @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta_z
    return grads


def ortho_penalty_gradient(W):
    """Gradient of ||W^T W - I||_F^2."""
    WtW = W.T @ W
    I = np.eye(W.shape[0])
    return 4.0 * W @ (WtW - I)


def newton_schulz_orthogonalize(G, num_iters=5):
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm
    for _ in range(num_iters):
        A = X.T @ X
        X = (15.0 / 8.0) * X - (10.0 / 8.0) * X @ A + (3.0 / 8.0) * X @ A @ A
    return X


# =============================================================================
# TRAINING ROUTINES
# =============================================================================

def safe_loss(weights, X, Y, act_fn):
    loss = compute_loss(weights, X, Y, act_fn)
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss


def train_sgd(weights, X, Y, num_steps, lr, act_fn, dact_fn):
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)


def train_muon(weights, X, Y, num_steps, lr, act_fn, dact_fn, ns_iters=5):
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            G_orth = newton_schulz_orthogonalize(grads[i], ns_iters)
            weights[i] -= lr * G_orth
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)


def train_sgd_ortho_penalty(weights, X, Y, num_steps, lr, lam, act_fn, dact_fn):
    """SGD with weight orthogonality penalty."""
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            pen_grad = ortho_penalty_gradient(weights[i])
            weights[i] -= lr * (grads[i] + lam * pen_grad)
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)


def train_partial_ortho(weights, X, Y, num_steps, lr, alpha, act_fn, dact_fn, ns_iters=5):
    """Blend SGD with Muon: step = (1-alpha)*G + alpha*ortho(G).
    alpha=0 is pure SGD, alpha=1 is pure Muon direction."""
    weights = [W.copy() for W in weights]
    for step in range(num_steps):
        grads = compute_gradients(weights, X, Y, act_fn, dact_fn)
        for i in range(len(weights)):
            G = grads[i]
            G_orth = newton_schulz_orthogonalize(G, ns_iters)
            # Scale G_orth to match G's norm for fair blending
            gn = np.linalg.norm(G, 'fro')
            on = np.linalg.norm(G_orth, 'fro')
            if on > 1e-12:
                G_orth_scaled = G_orth * (gn / on)
            else:
                G_orth_scaled = G_orth
            blended = (1 - alpha) * G + alpha * G_orth_scaled
            weights[i] -= lr * blended
        if step % 50 == 0 and safe_loss(weights, X, Y, act_fn) > 1e8:
            return weights, 1e10
    return weights, safe_loss(weights, X, Y, act_fn)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    X = rng.randn(INPUT_DIM, BATCH_SIZE) * 0.5
    Y = rng.randn(OUTPUT_DIM, BATCH_SIZE) * 0.3

    print("=" * 90)
    print("Experiment 2.13: Activation Curvature Theory")
    print("=" * 90)
    print()
    print("HYPOTHESIS: Ortho penalty recovery % decreases with activation curvature.")
    print("  Higher |f''| -> harder for penalty to match Muon's benefit.")
    print()
    print(f"Config: {DEPTH}-layer, width={WIDTH}, {NUM_STEPS} steps")
    print(f"  lr_sgd={LR_SGD}, lr_muon={LR_MUON}, ortho_lambda={ORTHO_LAMBDA}")
    print()

    # =========================================================================
    # STEP 1: Activation curvatures
    # =========================================================================
    print("=" * 90)
    print("STEP 1: Activation Second Derivatives (mean |f''(x)|)")
    print("=" * 90)
    print()

    curvatures = {}
    for name, (act_fn, _) in ACTIVATIONS.items():
        curv = estimate_mean_second_derivative(act_fn)
        curvatures[name] = curv
        print(f"  {name:<18}: mean|f''| = {curv:.6f}")

    # =========================================================================
    # STEP 2: Train all combinations
    # =========================================================================
    print()
    print("=" * 90)
    print("STEP 2: Training")
    print("=" * 90)
    print()

    results = {}
    for name, (act_fn, dact_fn) in ACTIVATIONS.items():
        print(f"  {name}:", flush=True)
        weights_init = init_weights(DEPTH, WIDTH, seed=SEED)

        # (a) SGD
        _, loss_sgd = train_sgd(weights_init, X, Y, NUM_STEPS, LR_SGD, act_fn, dact_fn)
        print(f"    SGD     = {loss_sgd:.6f}")

        # (b) Muon
        _, loss_muon = train_muon(weights_init, X, Y, NUM_STEPS, LR_MUON, act_fn, dact_fn, NS_ITERS)
        print(f"    Muon    = {loss_muon:.6f}")

        # (c) SGD + weight ortho penalty (lambda=0.003)
        _, loss_pen = train_sgd_ortho_penalty(
            weights_init, X, Y, NUM_STEPS, LR_SGD, ORTHO_LAMBDA, act_fn, dact_fn
        )
        print(f"    Penalty = {loss_pen:.6f}")

        # (d) Partial ortho -- sweep alpha to find best
        best_alpha = 0
        best_loss_partial = loss_sgd
        alpha_sweep = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        for alpha in alpha_sweep:
            _, loss_partial = train_partial_ortho(
                weights_init, X, Y, NUM_STEPS, LR_SGD, alpha, act_fn, dact_fn, NS_ITERS
            )
            if loss_partial < best_loss_partial:
                best_loss_partial = loss_partial
                best_alpha = alpha
        print(f"    PartOrth= {best_loss_partial:.6f} (alpha={best_alpha})")

        # Recovery calculations
        gap = loss_sgd - loss_muon
        if gap > 1e-8 and loss_muon < loss_sgd:
            rec_pen = (loss_sgd - loss_pen) / gap * 100.0
            rec_partial = (loss_sgd - best_loss_partial) / gap * 100.0
        else:
            rec_pen = 0.0
            rec_partial = 0.0

        results[name] = {
            'loss_sgd': loss_sgd,
            'loss_muon': loss_muon,
            'loss_penalty': loss_pen,
            'loss_partial': best_loss_partial,
            'best_alpha': best_alpha,
            'recovery_pen': rec_pen,
            'recovery_partial': rec_partial,
            'curvature': curvatures[name],
        }

    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print()
    print("=" * 90)
    print("RESULTS TABLE (sorted by curvature)")
    print("=" * 90)
    header_curv = 'mean|f"|'
    print(f"{'Activation':<18} {header_curv:>8} {'SGD':>9} {'Muon':>9} "
          f"{'Penalty':>9} {'PartOrth':>9} {'Rec_pen%':>9} {'Rec_part%':>10}")
    print("-" * 90)

    sorted_names = sorted(results.keys(), key=lambda n: results[n]['curvature'])
    for name in sorted_names:
        r = results[name]
        print(f"{name:<18} {r['curvature']:>8.4f} {r['loss_sgd']:>9.5f} "
              f"{r['loss_muon']:>9.5f} {r['loss_penalty']:>9.5f} "
              f"{r['loss_partial']:>9.5f} {r['recovery_pen']:>9.1f} "
              f"{r['recovery_partial']:>10.1f}")

    # =========================================================================
    # RECOVERY vs CURVATURE (using partial ortho -- the more meaningful metric)
    # =========================================================================
    print()
    print("=" * 90)
    print("RECOVERY vs CURVATURE (partial ortho blend)")
    print("=" * 90)
    print()

    max_abs_rec = max(abs(results[n]['recovery_partial']) for n in sorted_names)
    scale = 50.0 / max(max_abs_rec, 1)

    for name in sorted_names:
        r = results[name]
        rec = r['recovery_partial']
        if rec >= 0:
            bar = '#' * max(0, int(rec * scale))
            print(f"  {name:<18} curv={r['curvature']:.4f}  rec={rec:>7.1f}%  |{bar}")
        else:
            bar = '-' * max(0, int(-rec * scale))
            print(f"  {name:<18} curv={r['curvature']:.4f}  rec={rec:>7.1f}%  |{bar} (neg)")

    # =========================================================================
    # MONOTONICITY ANALYSIS
    # =========================================================================
    print()
    print("=" * 90)
    print("MONOTONICITY ANALYSIS (using partial ortho recovery)")
    print("=" * 90)
    print()

    curvs = np.array([results[n]['curvature'] for n in sorted_names])
    recovs = np.array([results[n]['recovery_partial'] for n in sorted_names])

    # Count concordant pairs (skip tied curvatures)
    n_pairs = 0
    n_correct = 0
    inversions = []
    for i in range(len(sorted_names)):
        for j in range(i + 1, len(sorted_names)):
            ci = results[sorted_names[i]]['curvature']
            cj = results[sorted_names[j]]['curvature']
            ri = results[sorted_names[i]]['recovery_partial']
            rj = results[sorted_names[j]]['recovery_partial']
            if abs(ci - cj) < 1e-8:
                continue
            n_pairs += 1
            if ri >= rj:
                n_correct += 1
            else:
                inversions.append((sorted_names[i], sorted_names[j],
                                   ci, cj, ri, rj))

    concordance = n_correct / n_pairs if n_pairs > 0 else 0

    print(f"  Concordant pairs (lower curv -> higher rec): {n_correct}/{n_pairs} ({concordance:.0%})")
    if inversions:
        print(f"  Inversions (higher curv has HIGHER recovery):")
        for n1, n2, c1, c2, r1, r2 in inversions:
            print(f"    {n1}(c={c1:.4f},r={r1:.1f}%) vs {n2}(c={c2:.4f},r={r2:.1f}%)")

    # Spearman rank correlation
    def rank_array(arr):
        temp = sorted(range(len(arr)), key=lambda k: arr[k])
        ranks = [0.0] * len(arr)
        for rank_val, idx in enumerate(temp):
            ranks[idx] = rank_val + 1.0
        return ranks

    curv_ranks = rank_array(curvs.tolist())
    rec_ranks = rank_array(recovs.tolist())
    n = len(curv_ranks)
    d_sq = sum((cr - rr) ** 2 for cr, rr in zip(curv_ranks, rec_ranks))
    spearman_rho = 1 - 6 * d_sq / (n * (n * n - 1))

    print(f"\n  Spearman rho (curvature vs recovery): {spearman_rho:.4f}")
    print(f"  (Negative = higher curvature -> lower recovery)")

    # =========================================================================
    # HYPOTHESIS TESTS
    # =========================================================================
    print()
    print("=" * 90)
    print("HYPOTHESIS TESTS")
    print("=" * 90)

    # Test 1: Muon beats SGD for most activations
    muon_wins = sum(1 for n in ACTIVATIONS if results[n]['loss_muon'] < results[n]['loss_sgd'])
    test1_pass = muon_wins >= len(ACTIVATIONS) * 0.5
    print(f"\n  Test 1: Muon beats SGD for >= 50% of activations")
    print(f"    Muon wins: {muon_wins}/{len(ACTIVATIONS)}  [{'PASS' if test1_pass else 'FAIL'}]")

    # Test 2: Partial ortho recovers SOME advantage for at least 3 activations
    positive_rec = sum(1 for n in ACTIVATIONS if results[n]['recovery_partial'] > 10)
    test2_pass = positive_rec >= 3
    print(f"\n  Test 2: Partial ortho recovery > 10% for >= 3 activations")
    print(f"    Count: {positive_rec}/{len(ACTIVATIONS)}  [{'PASS' if test2_pass else 'FAIL'}]")

    # Test 3: Negative Spearman correlation
    test3_pass = spearman_rho < -0.3
    print(f"\n  Test 3: Spearman rho < -0.3 (negative correlation)")
    print(f"    rho = {spearman_rho:.4f}  [{'PASS' if test3_pass else 'FAIL'}]")

    # Test 4: Concordance >= 60%
    test4_pass = concordance >= 0.60
    print(f"\n  Test 4: Concordance >= 60% (monotonic decreasing trend)")
    print(f"    Concordance = {concordance:.0%}  [{'PASS' if test4_pass else 'FAIL'}]")

    # Test 5: Separation between zero-curv and nonzero-curv groups
    zero_curv = [n for n in sorted_names if results[n]['curvature'] < 0.01]
    nonzero_curv = [n for n in sorted_names if results[n]['curvature'] >= 0.01]
    if zero_curv and nonzero_curv:
        mean_zero_rec = np.mean([results[n]['recovery_partial'] for n in zero_curv])
        mean_nz_rec = np.mean([results[n]['recovery_partial'] for n in nonzero_curv])
        test5_pass = mean_zero_rec > mean_nz_rec
        print(f"\n  Test 5: Zero-curv activations have higher mean recovery")
        print(f"    Zero-curv mean = {mean_zero_rec:.1f}%, nonzero mean = {mean_nz_rec:.1f}%"
              f"  [{'PASS' if test5_pass else 'FAIL'}]")
    else:
        test5_pass = False
        print(f"\n  Test 5: Not enough groups  [FAIL]")

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    print()
    print("=" * 90)

    core_pass = (test3_pass or test4_pass) and test1_pass
    direction_pass = spearman_rho < 0

    if core_pass:
        print("OVERALL: PASS")
        print("  Recovery % decreases with activation curvature.")
        print(f"  Spearman rho = {spearman_rho:.4f}, concordance = {concordance:.0%}")
    elif direction_pass and test1_pass:
        print("OVERALL: PARTIAL PASS")
        print("  Correlation direction is correct (negative rho) but not strongly monotonic.")
        print(f"  Spearman rho = {spearman_rho:.4f}, concordance = {concordance:.0%}")
    else:
        print("OVERALL: FAIL")
        if not test1_pass:
            print("  Muon did not consistently beat SGD across activations.")
        if not direction_pass:
            print(f"  Correlation POSITIVE (rho={spearman_rho:.4f}), opposite to prediction.")
        if not (test3_pass or test4_pass):
            print(f"  Correlation too weak (rho={spearman_rho:.4f}, conc={concordance:.0%})")

    print("=" * 90)


if __name__ == "__main__":
    run_experiment()
