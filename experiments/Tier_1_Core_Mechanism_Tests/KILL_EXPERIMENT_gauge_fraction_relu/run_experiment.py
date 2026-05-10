#!/usr/bin/env python3
"""
KILL EXPERIMENT: Gradient Gauge Fraction in ReLU MLP
=====================================================

THE FALSIFICATION TEST for the gauge theory of Muon.

Train a 6-layer ReLU MLP (width 32) on random regression (100 samples).
At each step, decompose the gradient of each layer into gauge and physical
components.

DECOMPOSITION (from Axiom 0.9a):
  Let W be a weight matrix, G its gradient, Q = ortho(W) = UV^T (polar factor).
  The Muon update direction is Q (up to sign).

  G can be decomposed into:
    G = G_tangent + G_normal

  where G_tangent is the component in the tangent space of the Stiefel manifold
  at Q, and G_normal is the normal component.

  For Q orthogonal, the tangent space at Q consists of matrices Z such that
  Q^T Z + Z^T Q is antisymmetric, i.e., Q^T Z = -Z^T Q is skew-symmetric.
  Equivalently: Z = Q A + Q_perp B where A is skew-symmetric.

  The normal component is: G_normal = Q * sym(Q^T G)
  where sym(M) = (M + M^T) / 2.

  The gauge fraction = ||G_normal||_F^2 / ||G||_F^2

  INTERPRETATION:
  - G_normal points in the "gauge" direction (scaling/stretching, not rotation)
  - G_tangent points in the "physical" direction (rotation on the Stiefel manifold)
  - Muon (ortho projection) effectively REMOVES G_normal and keeps G_tangent

  The gauge fraction tells us what fraction of the gradient is "wasted" on
  non-rotational (gauge) directions that Muon ignores.

PREDICTION FROM THEORY:
  - Linear nets: ~50% gauge fraction (half the gradient is gauge)
  - ReLU nets: 15-35% (reduced from 50% by ReLU breaking exact gauge symmetry)
  - If <5%: the gauge theory is DEAD for nonlinear nets

SETUP:
  - 6-layer ReLU MLP, width 32
  - Random regression: 100 input-output pairs, X ~ N(0,1), Y ~ N(0,1)
  - 500 training steps with SGD (to measure the natural gradient decomposition)
  - Also measure for Muon for comparison
  - Per-layer gauge fraction tracked every step
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
np.random.seed(SEED)

INPUT_DIM = 32
HIDDEN_DIM = 32
OUTPUT_DIM = 32
NUM_LAYERS = 6  # 6 weight matrices (including input and output projections)
NUM_SAMPLES = 100
NUM_STEPS = 500
LR_SGD = 0.003
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# RELU MLP
# =============================================================================

def init_weights(rng):
    """Initialize 6-layer MLP: all layers are HIDDEN_DIM x HIDDEN_DIM for simplicity."""
    weights = []
    for i in range(NUM_LAYERS):
        # He initialization
        std = np.sqrt(2.0 / HIDDEN_DIM)
        W = rng.randn(HIDDEN_DIM, HIDDEN_DIM) * std
        weights.append(W.copy())
    return weights


def forward_relu(weights, X):
    """Forward pass: ReLU between layers, linear output."""
    out = X.copy()
    pre_acts = []
    post_acts = [X.copy()]
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre.copy())
        if idx < len(weights) - 1:
            out = np.maximum(0, pre)
        else:
            out = pre  # No ReLU on last layer
        post_acts.append(out.copy())
    return out, pre_acts, post_acts


def compute_loss(weights, X, Y):
    pred, _, _ = forward_relu(weights, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients(weights, X, Y):
    """Backprop through ReLU MLP."""
    num_layers = len(weights)
    N = X.shape[1]

    pred, pre_acts, post_acts = forward_relu(weights, X)
    delta = (pred - Y) / N

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ post_acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
            # ReLU derivative
            delta = delta * (pre_acts[l - 1] > 0).astype(float)

    return grads


# =============================================================================
# GAUGE DECOMPOSITION
# =============================================================================

def compute_polar_factor(W):
    """Compute Q = UV^T from SVD of W."""
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    return U @ Vt


def gauge_decomposition(G, W):
    """
    Decompose gradient G into tangent (physical) and normal (gauge) components
    at the Stiefel manifold point Q = ortho(W).

    G_normal = Q @ sym(Q^T @ G)  (the gauge/normal component)
    G_tangent = G - G_normal     (the physical/tangent component)

    Returns:
        gauge_fraction: ||G_normal||^2 / ||G||^2
        G_normal: the gauge component
        G_tangent: the physical component
    """
    Q = compute_polar_factor(W)

    QtG = Q.T @ G
    sym_QtG = 0.5 * (QtG + QtG.T)  # symmetric part
    G_normal = Q @ sym_QtG

    G_tangent = G - G_normal

    G_norm_sq = np.sum(G ** 2)
    G_normal_norm_sq = np.sum(G_normal ** 2)

    if G_norm_sq < 1e-30:
        return 0.0, G_normal, G_tangent

    gauge_fraction = G_normal_norm_sq / G_norm_sq
    return gauge_fraction, G_normal, G_tangent


# =============================================================================
# ALSO DO LINEAR NET FOR REFERENCE
# =============================================================================

def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss_linear(weights, X, Y):
    pred = forward_linear(weights, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients_linear(weights, X, Y):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())
    delta = (activations[-1] - Y) / N
    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


# =============================================================================
# NEWTON-SCHULZ
# =============================================================================

def newton_schulz_ortho(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-12:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# TRAINING WITH GAUGE TRACKING
# =============================================================================

def train_with_gauge_tracking(net_type, weights_init, X, Y, lr, optimizer='sgd',
                               n_steps=NUM_STEPS):
    """
    Train and track gauge fraction per layer at each step.
    Returns losses, gauge_fractions (shape: [n_steps, num_layers]).
    """
    weights = [w.copy() for w in weights_init]
    velocities = [np.zeros_like(w) for w in weights]
    num_layers = len(weights)

    if net_type == 'relu':
        loss_fn = compute_loss
        grad_fn = compute_gradients
    else:
        loss_fn = compute_loss_linear
        grad_fn = compute_gradients_linear

    losses = []
    gauge_fractions = []  # [step, layer]

    for step in range(n_steps):
        loss = loss_fn(weights, X, Y)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            losses.extend([1e10] * (n_steps - step - 1))
            gauge_fractions.extend([[0.0] * num_layers] * (n_steps - step - 1))
            break

        grads = grad_fn(weights, X, Y)

        # Compute gauge fraction for each layer
        step_fractions = []
        for l in range(num_layers):
            gf, _, _ = gauge_decomposition(grads[l], weights[l])
            step_fractions.append(gf)
        gauge_fractions.append(step_fractions)

        # Optimizer step
        if optimizer == 'sgd':
            for i in range(num_layers):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] = weights[i] - lr * velocities[i]
        elif optimizer == 'muon':
            for i in range(num_layers):
                ortho_grad = newton_schulz_ortho(grads[i])
                velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                weights[i] = weights[i] - lr * velocities[i]

    return np.array(losses), np.array(gauge_fractions)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 85)
print("KILL EXPERIMENT: Gradient Gauge Fraction in ReLU MLP")
print("=" * 85)
print(f"Setup: {NUM_LAYERS}-layer ReLU MLP (width={HIDDEN_DIM})")
print(f"Data: {NUM_SAMPLES} random regression samples")
print(f"Steps: {NUM_STEPS}")
print(f"Seeds: {NUM_SEEDS}")
print()
print("PREDICTION: ReLU gauge fraction 15-35%. If <5%, gauge theory is DEAD.")
print("=" * 85)

# Collect results
all_relu_sgd_gf = []      # [seed, step, layer]
all_relu_muon_gf = []
all_linear_sgd_gf = []
all_linear_muon_gf = []
all_relu_sgd_loss = []
all_relu_muon_loss = []
all_linear_sgd_loss = []
all_linear_muon_loss = []

for seed_idx in range(NUM_SEEDS):
    run_seed = SEED + seed_idx * 31
    rng = np.random.RandomState(run_seed)
    print(f"\n--- Seed {run_seed} ---")

    # Random data
    X = rng.randn(INPUT_DIM, NUM_SAMPLES) * 0.3
    Y = rng.randn(OUTPUT_DIM, NUM_SAMPLES) * 0.3

    # Initialize weights (same for both optimizers)
    weights_init = init_weights(rng)

    # ReLU MLP with SGD
    print("  Training ReLU MLP with SGD...", flush=True)
    relu_sgd_loss, relu_sgd_gf = train_with_gauge_tracking(
        'relu', weights_init, X, Y, LR_SGD, 'sgd')
    all_relu_sgd_gf.append(relu_sgd_gf)
    all_relu_sgd_loss.append(relu_sgd_loss)

    # ReLU MLP with Muon
    print("  Training ReLU MLP with Muon...", flush=True)
    relu_muon_loss, relu_muon_gf = train_with_gauge_tracking(
        'relu', weights_init, X, Y, LR_MUON, 'muon')
    all_relu_muon_gf.append(relu_muon_gf)
    all_relu_muon_loss.append(relu_muon_loss)

    # Linear net with SGD (reference)
    print("  Training Linear net with SGD...", flush=True)
    linear_sgd_loss, linear_sgd_gf = train_with_gauge_tracking(
        'linear', weights_init, X, Y, LR_SGD, 'sgd')
    all_linear_sgd_gf.append(linear_sgd_gf)
    all_linear_sgd_loss.append(linear_sgd_loss)

    # Linear net with Muon (reference)
    print("  Training Linear net with Muon...", flush=True)
    linear_muon_loss, linear_muon_gf = train_with_gauge_tracking(
        'linear', weights_init, X, Y, LR_MUON, 'muon')
    all_linear_muon_gf.append(linear_muon_gf)
    all_linear_muon_loss.append(linear_muon_loss)

    # Print summary for this seed
    relu_sgd_mean_gf = np.mean(relu_sgd_gf) * 100
    relu_muon_mean_gf = np.mean(relu_muon_gf) * 100
    linear_sgd_mean_gf = np.mean(linear_sgd_gf) * 100
    linear_muon_mean_gf = np.mean(linear_muon_gf) * 100

    print(f"  Mean gauge fraction (across all layers/steps):")
    print(f"    ReLU  SGD:  {relu_sgd_mean_gf:.1f}%   Muon: {relu_muon_mean_gf:.1f}%")
    print(f"    Linear SGD: {linear_sgd_mean_gf:.1f}%   Muon: {linear_muon_mean_gf:.1f}%")


# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("AGGREGATE RESULTS")
print(f"{'=' * 85}")

# Combine across seeds: shape (num_seeds, steps, layers)
relu_sgd_gf_all = np.array(all_relu_sgd_gf)
relu_muon_gf_all = np.array(all_relu_muon_gf)
linear_sgd_gf_all = np.array(all_linear_sgd_gf)
linear_muon_gf_all = np.array(all_linear_muon_gf)

# Mean across seeds
relu_sgd_gf_mean = np.mean(relu_sgd_gf_all, axis=0)   # (steps, layers)
relu_muon_gf_mean = np.mean(relu_muon_gf_all, axis=0)
linear_sgd_gf_mean = np.mean(linear_sgd_gf_all, axis=0)
linear_muon_gf_mean = np.mean(linear_muon_gf_all, axis=0)


# =============================================================================
# PER-LAYER GAUGE FRACTION TABLE
# =============================================================================

print(f"\n{'=' * 85}")
print("PER-LAYER GAUGE FRACTION (%, averaged over all steps and seeds)")
print(f"{'=' * 85}")

print(f"\n{'Layer':>6} {'ReLU SGD':>10} {'ReLU Muon':>10} {'Linear SGD':>12} {'Linear Muon':>12}")
print("-" * 55)

for l in range(NUM_LAYERS):
    r_sgd = np.mean(relu_sgd_gf_mean[:, l]) * 100
    r_muon = np.mean(relu_muon_gf_mean[:, l]) * 100
    li_sgd = np.mean(linear_sgd_gf_mean[:, l]) * 100
    li_muon = np.mean(linear_muon_gf_mean[:, l]) * 100
    print(f"{l+1:>6} {r_sgd:>10.1f}% {r_muon:>10.1f}% {li_sgd:>11.1f}% {li_muon:>11.1f}%")

# Overall
print("-" * 55)
r_sgd_all = np.mean(relu_sgd_gf_mean) * 100
r_muon_all = np.mean(relu_muon_gf_mean) * 100
li_sgd_all = np.mean(linear_sgd_gf_mean) * 100
li_muon_all = np.mean(linear_muon_gf_mean) * 100
print(f"{'ALL':>6} {r_sgd_all:>10.1f}% {r_muon_all:>10.1f}% {li_sgd_all:>11.1f}% {li_muon_all:>11.1f}%")


# =============================================================================
# GAUGE FRACTION OVER TIME
# =============================================================================

print(f"\n\n{'=' * 85}")
print("GAUGE FRACTION OVER TIME (%, averaged over layers and seeds)")
print(f"{'=' * 85}")

print(f"\n{'Step':>6} {'ReLU SGD':>10} {'ReLU Muon':>10} {'Linear SGD':>12} {'Linear Muon':>12}")
print("-" * 55)

snapshot_steps = [0, 10, 25, 50, 100, 200, 300, 400, 499]
for s in snapshot_steps:
    if s < relu_sgd_gf_mean.shape[0]:
        r_sgd = np.mean(relu_sgd_gf_mean[s, :]) * 100
        r_muon = np.mean(relu_muon_gf_mean[s, :]) * 100
        li_sgd = np.mean(linear_sgd_gf_mean[s, :]) * 100
        li_muon = np.mean(linear_muon_gf_mean[s, :]) * 100
        print(f"{s:>6} {r_sgd:>10.1f}% {r_muon:>10.1f}% {li_sgd:>11.1f}% {li_muon:>11.1f}%")


# =============================================================================
# EARLY vs LATE COMPARISON
# =============================================================================

print(f"\n\n{'=' * 85}")
print("EARLY vs LATE GAUGE FRACTION")
print(f"{'=' * 85}")

early_steps = slice(0, 50)
late_steps = slice(400, 500)

for name, data in [("ReLU SGD", relu_sgd_gf_mean),
                    ("ReLU Muon", relu_muon_gf_mean),
                    ("Linear SGD", linear_sgd_gf_mean),
                    ("Linear Muon", linear_muon_gf_mean)]:
    early = np.mean(data[early_steps, :]) * 100
    late_end = min(500, data.shape[0])
    late_start = max(0, late_end - 100)
    late = np.mean(data[late_start:late_end, :]) * 100
    print(f"  {name:<15}: early (0-50) = {early:.1f}%,  late (400-500) = {late:.1f}%")


# =============================================================================
# THE KILL TEST
# =============================================================================

print(f"\n\n{'=' * 85}")
print("THE KILL TEST: Is gauge fraction substantial in ReLU nets?")
print(f"{'=' * 85}")

relu_sgd_overall = np.mean(relu_sgd_gf_all) * 100
relu_muon_overall = np.mean(relu_muon_gf_all) * 100
linear_sgd_overall = np.mean(linear_sgd_gf_all) * 100
linear_muon_overall = np.mean(linear_muon_gf_all) * 100

print(f"""
  ReLU MLP gauge fraction (SGD training):  {relu_sgd_overall:.1f}%
  ReLU MLP gauge fraction (Muon training): {relu_muon_overall:.1f}%
  Linear net gauge fraction (SGD):         {linear_sgd_overall:.1f}%
  Linear net gauge fraction (Muon):        {linear_muon_overall:.1f}%
""")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"{'=' * 85}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 85}")

# T1: Linear net gauge fraction should be ~50% (theory baseline)
t1 = 30.0 < linear_sgd_overall < 70.0
print(f"\nT1: Linear net gauge fraction in 30-70% range (expect ~50%)?")
print(f"    Linear SGD: {linear_sgd_overall:.1f}%")
print(f"    --> {'PASS' if t1 else 'FAIL'}")

# T2: ReLU net gauge fraction > 5% (gauge theory is alive)
t2 = relu_sgd_overall > 5.0
print(f"\nT2: ReLU gauge fraction > 5% (gauge theory alive)?")
print(f"    ReLU SGD: {relu_sgd_overall:.1f}%")
print(f"    --> {'PASS' if t2 else 'FAIL -- GAUGE THEORY IS DEAD FOR NONLINEAR NETS'}")

# T3: ReLU gauge fraction > 15% (strong gauge signal)
t3 = relu_sgd_overall > 15.0
print(f"\nT3: ReLU gauge fraction > 15% (strong signal)?")
print(f"    ReLU SGD: {relu_sgd_overall:.1f}%")
print(f"    --> {'PASS' if t3 else 'FAIL'}")

# T4: ReLU gauge fraction < linear (ReLU breaks some gauge symmetry)
t4 = relu_sgd_overall < linear_sgd_overall
print(f"\nT4: ReLU gauge fraction < linear (ReLU breaks gauge symmetry)?")
print(f"    ReLU: {relu_sgd_overall:.1f}%, Linear: {linear_sgd_overall:.1f}%")
print(f"    --> {'PASS' if t4 else 'FAIL'}")

# T5: Gauge fraction is non-trivial (>10%) in at least 4/6 layers for ReLU
per_layer_relu = [np.mean(relu_sgd_gf_mean[:, l]) * 100 for l in range(NUM_LAYERS)]
n_nontrivial = sum(1 for gf in per_layer_relu if gf > 10.0)
t5 = n_nontrivial >= 4
print(f"\nT5: Gauge fraction >10% in at least 4/6 layers?")
print(f"    Per-layer: {[f'{gf:.1f}%' for gf in per_layer_relu]}")
print(f"    Layers with >10%: {n_nontrivial}/6")
print(f"    --> {'PASS' if t5 else 'FAIL'}")

# T6: Gauge fraction persists late in training (>5% at step 400+)
late_relu = np.mean(relu_sgd_gf_mean[max(0, relu_sgd_gf_mean.shape[0]-100):, :]) * 100
t6 = late_relu > 5.0
print(f"\nT6: Gauge fraction persists late in training (>5% at step 400+)?")
print(f"    Late ReLU SGD: {late_relu:.1f}%")
print(f"    --> {'PASS' if t6 else 'FAIL'}")


# =============================================================================
# FINAL VERDICT
# =============================================================================

total_pass = sum([t1, t2, t3, t4, t5, t6])

print(f"\n\n{'=' * 85}")
print("FINAL VERDICT: KILL EXPERIMENT")
print(f"{'=' * 85}")

print(f"""
  THE QUESTION: Does the gradient have a substantial gauge component
  in nonlinear (ReLU) networks?

  If YES (gauge fraction 15-35%): The gauge theory extends to nonlinear nets.
    Muon's orthogonal projection is removing a meaningful gauge component.
  If NO (gauge fraction <5%): The gauge theory is DEAD for nonlinear nets.
    Muon works for a different reason than gauge fixing.

  RESULTS:
    ReLU gauge fraction: {relu_sgd_overall:.1f}% (SGD), {relu_muon_overall:.1f}% (Muon)
    Linear reference:    {linear_sgd_overall:.1f}% (SGD), {linear_muon_overall:.1f}% (Muon)

  Tests passed: {total_pass}/6
""")

if relu_sgd_overall < 5.0:
    print("  *** KILL CONFIRMED: GAUGE THEORY IS DEAD FOR NONLINEAR NETS ***")
    print("  The gradient gauge fraction is negligible in ReLU networks.")
    print("  Muon's advantage must come from a different mechanism.")
elif relu_sgd_overall < 15.0:
    print("  WEAK SIGNAL: Gauge fraction is present but small.")
    print("  The gauge theory has limited applicability to nonlinear nets.")
    print("  Other mechanisms likely dominate Muon's advantage.")
elif relu_sgd_overall < 35.0:
    print("  GAUGE THEORY SURVIVES: Substantial gauge fraction in ReLU nets.")
    print("  The gauge-fixing interpretation of Muon extends to nonlinear networks.")
    print("  ReLU reduces but does not eliminate gauge symmetry.")
else:
    print("  STRONG GAUGE SIGNAL: ReLU gauge fraction is surprisingly high.")
    print("  The near-orthogonal weight structure preserves gauge symmetry")
    print("  even through nonlinear activations.")

print(f"\n{'=' * 85}")
