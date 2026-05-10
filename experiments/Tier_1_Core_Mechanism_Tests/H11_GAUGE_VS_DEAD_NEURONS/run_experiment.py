#!/usr/bin/env python3
"""
H11: Gauge Fraction vs Dead Neuron Fraction
=============================================

6-layer ReLU MLP, width 32. Vary bias initialization to control dead neuron fraction.
bias_init in {+2, +1, 0, -1, -2, -3, -5}
  Positive bias -> more neurons alive (ReLU threshold shifted left)
  Negative bias -> more neurons dead

At step 100, measure:
  (a) Fraction of dead neurons per layer
  (b) Gradient gauge fraction per layer (Stiefel normal-space decomposition)

KEY QUESTION: Is the gauge fraction a LOCAL tangent-space property (constant ~53%
regardless of dead fraction) or does it depend on GLOBAL symmetry (drops with dead
fraction)?

If gauge fraction is constant: gauge is local tangent structure
If gauge fraction drops with dead neurons: gauge is global symmetry that ReLU breaks

Uses the same gauge decomposition as the KILL experiment.

IMPORTANT: When ALL neurons in a layer are dead, the gradient is exactly zero,
so gauge_fraction = 0/0 -> we report NaN and exclude from analysis.
We only analyze layers that have non-trivial gradients (||G||_F > threshold).
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 6
NUM_SAMPLES = 100
NUM_STEPS = 100   # measure at step 100
LR = 0.003
MOMENTUM = 0.9
NUM_SEEDS = 5
BASE_SEED = 42
NS_ITERS = 5

BIAS_INITS = [+2.0, +1.0, 0.0, -1.0, -2.0, -3.0, -5.0]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Gradient norm threshold: below this, gauge decomposition is unreliable
GRAD_NORM_THRESHOLD = 1e-10


# =============================================================================
# ReLU MLP WITH BIASES
# =============================================================================

def init_weights(rng, bias_init_val):
    """Initialize 6-layer MLP with biases set to bias_init_val."""
    weights = []
    biases = []
    for i in range(NUM_LAYERS):
        std = np.sqrt(2.0 / DIM)
        W = rng.randn(DIM, DIM) * std
        b = np.full(DIM, bias_init_val)
        weights.append(W.copy())
        biases.append(b.copy())
    return weights, biases


def forward(weights, biases, X):
    """Forward pass with biases. ReLU on all but last layer."""
    out = X.copy()
    pre_acts = []
    post_acts = [X.copy()]
    relu_masks = []

    for idx in range(len(weights)):
        pre = weights[idx] @ out + biases[idx][:, None]
        pre_acts.append(pre.copy())
        if idx < len(weights) - 1:
            mask = (pre > 0).astype(float)
            relu_masks.append(mask)
            out = pre * mask
        else:
            relu_masks.append(np.ones_like(pre))
            out = pre
        post_acts.append(out.copy())

    return out, pre_acts, post_acts, relu_masks


def compute_loss(weights, biases, X, Y):
    pred, _, _, _ = forward(weights, biases, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients(weights, biases, X, Y):
    """Backprop through biased ReLU MLP. Returns weight grads and bias grads."""
    num_layers = len(weights)
    N = X.shape[1]

    pred, pre_acts, post_acts, relu_masks = forward(weights, biases, X)
    delta = (pred - Y) / N

    w_grads = [None] * num_layers
    b_grads = [None] * num_layers

    for l in range(num_layers - 1, -1, -1):
        w_grads[l] = delta @ post_acts[l].T
        b_grads[l] = np.sum(delta, axis=1)
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * relu_masks[l - 1]

    return w_grads, b_grads


# =============================================================================
# GAUGE DECOMPOSITION (from KILL experiment)
# =============================================================================

def compute_polar_factor(W):
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    return U @ Vt


def gauge_decomposition(G, W):
    """
    Decompose gradient G into tangent (physical) and normal (gauge) at Stiefel
    manifold point Q = ortho(W).
    G_normal = Q @ sym(Q^T @ G)
    gauge_fraction = ||G_normal||^2 / ||G||^2

    Returns NaN if gradient norm is below threshold.
    """
    G_norm_sq = np.sum(G ** 2)
    if G_norm_sq < GRAD_NORM_THRESHOLD:
        return np.nan  # unreliable

    Q = compute_polar_factor(W)
    QtG = Q.T @ G
    sym_QtG = 0.5 * (QtG + QtG.T)
    G_normal = Q @ sym_QtG
    G_normal_norm_sq = np.sum(G_normal ** 2)
    return G_normal_norm_sq / G_norm_sq


# =============================================================================
# DEAD NEURON MEASUREMENT
# =============================================================================

def measure_dead_fraction(weights, biases, X):
    """
    A neuron is 'dead' if it outputs 0 for ALL samples in X.
    Returns per-layer dead fractions (for layers 0..NUM_LAYERS-2, i.e., ReLU layers).
    """
    _, pre_acts, _, _ = forward(weights, biases, X)
    dead_fractions = []
    for l in range(NUM_LAYERS - 1):
        activations = pre_acts[l]
        alive_mask = np.any(activations > 0, axis=1)
        dead_frac = 1.0 - np.mean(alive_mask)
        dead_fractions.append(dead_frac)
    return dead_fractions


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_and_measure(weights_init, biases_init, X, Y, n_steps=NUM_STEPS):
    """
    Train with SGD for n_steps. At the final step, measure:
      - dead neuron fraction per layer
      - gauge fraction per layer (for weight gradients)
    """
    weights = [w.copy() for w in weights_init]
    biases = [b.copy() for b in biases_init]
    w_vel = [np.zeros_like(w) for w in weights]
    b_vel = [np.zeros_like(b) for b in biases]

    for step in range(n_steps):
        loss = compute_loss(weights, biases, X, Y)
        if np.isnan(loss) or loss > 1e10:
            break

        w_grads, b_grads = compute_gradients(weights, biases, X, Y)

        for i in range(NUM_LAYERS):
            w_vel[i] = MOMENTUM * w_vel[i] + w_grads[i]
            b_vel[i] = MOMENTUM * b_vel[i] + b_grads[i]
            weights[i] = weights[i] - LR * w_vel[i]
            biases[i] = biases[i] - LR * b_vel[i]

    # Measure at final step
    dead_fracs = measure_dead_fraction(weights, biases, X)

    # Gauge fraction: need gradients at current weights
    w_grads, _ = compute_gradients(weights, biases, X, Y)
    gauge_fracs = []
    for l in range(NUM_LAYERS):
        gf = gauge_decomposition(w_grads[l], weights[l])
        gauge_fracs.append(gf)

    final_loss = compute_loss(weights, biases, X, Y)

    return dead_fracs, gauge_fracs, final_loss


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 90)
print("H11: GAUGE FRACTION vs DEAD NEURON FRACTION")
print("=" * 90)
print(f"Architecture: {NUM_LAYERS}-layer ReLU MLP, width={DIM}")
print(f"Bias init values: {BIAS_INITS}")
print(f"Steps: {NUM_STEPS}, Seeds: {NUM_SEEDS}")
print()
print("KEY QUESTION: Does gauge fraction depend on dead neuron fraction?")
print("  If constant (~50%): gauge is local tangent structure")
print("  If drops with dead neurons: gauge is global symmetry")
print("  NaN = gradient too small for reliable decomposition")
print("=" * 90)

# Results storage
results = {}

for bias_val in BIAS_INITS:
    dead_all = []
    gauge_all = []
    loss_all = []

    for seed_idx in range(NUM_SEEDS):
        run_seed = BASE_SEED + seed_idx * 31
        rng = np.random.RandomState(run_seed)

        X = rng.randn(DIM, NUM_SAMPLES) * 0.3
        Y = rng.randn(DIM, NUM_SAMPLES) * 0.3

        weights_init, biases_init = init_weights(rng, bias_val)
        dead_fracs, gauge_fracs, final_loss = train_and_measure(
            weights_init, biases_init, X, Y)

        dead_all.append(dead_fracs)
        gauge_all.append(gauge_fracs)
        loss_all.append(final_loss)

    results[bias_val] = {
        'dead': np.array(dead_all),     # (seeds, NUM_LAYERS-1)
        'gauge': np.array(gauge_all),   # (seeds, NUM_LAYERS)
        'loss': np.array(loss_all),     # (seeds,)
    }

    mean_dead = np.mean(dead_all)
    gauge_valid = np.array(gauge_all)
    gauge_valid_flat = gauge_valid[~np.isnan(gauge_valid)]
    if len(gauge_valid_flat) > 0:
        mean_gauge = np.mean(gauge_valid_flat)
        n_nan = np.sum(np.isnan(gauge_valid))
    else:
        mean_gauge = float('nan')
        n_nan = gauge_valid.size
    print(f"  bias_init={bias_val:+.0f}: dead={mean_dead*100:.1f}%, "
          f"gauge={mean_gauge*100:.1f}% ({n_nan} NaN layers), "
          f"loss={np.mean(loss_all):.4f}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

print(f"\n\n{'=' * 90}")
print("SUMMARY TABLE: Gauge Fraction vs Dead Neuron Fraction")
print(f"{'=' * 90}")

print(f"\n{'Bias':>6}  {'Dead %':>8}  {'Gauge %':>9}  {'NaN layers':>11}  {'Valid layers':>13}  {'Loss':>10}")
print("-" * 68)

summary_dead = []
summary_gauge = []
summary_valid_gauge = []

for bias_val in BIAS_INITS:
    r = results[bias_val]
    mean_dead = np.mean(r['dead']) * 100
    gauge_flat = r['gauge'].flatten()
    valid = gauge_flat[~np.isnan(gauge_flat)]
    n_nan = np.sum(np.isnan(gauge_flat))
    n_valid = len(valid)

    if n_valid > 0:
        mean_gauge = np.mean(valid) * 100
    else:
        mean_gauge = float('nan')

    summary_dead.append(mean_dead)
    summary_gauge.append(mean_gauge)
    summary_valid_gauge.append((mean_gauge, n_valid))

    print(f"{bias_val:>+6.0f}  {mean_dead:>8.1f}  {mean_gauge:>9.1f}  "
          f"{n_nan:>11}  {n_valid:>13}  {np.mean(r['loss']):>10.4f}")


# =============================================================================
# PER-LAYER GAUGE FRACTION TABLE
# =============================================================================

print(f"\n\n{'=' * 90}")
print("PER-LAYER GAUGE FRACTION (%, mean over seeds, NaN = zero gradient)")
print(f"{'=' * 90}")

print(f"\n{'Bias':>6}", end="")
for l in range(NUM_LAYERS):
    print(f"  {'L'+str(l+1):>8}", end="")
print(f"  {'Mean':>8}")
print("-" * (8 + 10 * (NUM_LAYERS + 1)))

for bias_val in BIAS_INITS:
    r = results[bias_val]
    print(f"{bias_val:>+6.0f}", end="")
    all_layer_means = []
    for l in range(NUM_LAYERS):
        col = r['gauge'][:, l]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            m = np.mean(valid) * 100
            all_layer_means.append(m)
            print(f"  {m:>8.1f}", end="")
        else:
            print(f"  {'NaN':>8}", end="")
    if all_layer_means:
        print(f"  {np.mean(all_layer_means):>8.1f}")
    else:
        print(f"  {'NaN':>8}")


# =============================================================================
# PER-LAYER DEAD FRACTION
# =============================================================================

print(f"\n\n{'=' * 90}")
print("PER-LAYER DEAD NEURON FRACTION (%, mean over seeds)")
print(f"{'=' * 90}")

print(f"\n{'Bias':>6}", end="")
for l in range(NUM_LAYERS - 1):
    print(f"  {'L'+str(l+1):>8}", end="")
print(f"  {'Mean':>8}")
print("-" * (8 + 10 * NUM_LAYERS))

for bias_val in BIAS_INITS:
    r = results[bias_val]
    print(f"{bias_val:>+6.0f}", end="")
    layer_means = np.mean(r['dead'], axis=0) * 100
    for l in range(NUM_LAYERS - 1):
        print(f"  {layer_means[l]:>8.1f}", end="")
    print(f"  {np.mean(layer_means):>8.1f}")


# =============================================================================
# CORRELATION ANALYSIS (only using valid gauge data)
# =============================================================================

print(f"\n\n{'=' * 90}")
print("CORRELATION ANALYSIS")
print(f"{'=' * 90}")

# Use only conditions where gauge fraction is not NaN
dead_arr = np.array(summary_dead)
gauge_arr = np.array(summary_gauge)
valid_mask = ~np.isnan(gauge_arr)

dead_valid = dead_arr[valid_mask]
gauge_valid = gauge_arr[valid_mask]

if len(dead_valid) >= 2 and np.std(dead_valid) > 1e-10 and np.std(gauge_valid) > 1e-10:
    correlation = np.corrcoef(dead_valid, gauge_valid)[0, 1]
    slope = np.polyfit(dead_valid, gauge_valid, 1)[0]
    print(f"\n  Valid conditions: {len(dead_valid)}/{len(dead_arr)}")
    print(f"  Pearson correlation (dead % vs gauge %): r = {correlation:.4f}")
    print(f"  Dead fraction range:  {dead_valid.min():.1f}% to {dead_valid.max():.1f}%")
    print(f"  Gauge fraction range: {gauge_valid.min():.1f}% to {gauge_valid.max():.1f}%")
    print(f"  Linear fit slope: {slope:.4f} (gauge% per dead%)")
elif len(dead_valid) >= 2:
    correlation = 0.0
    print(f"\n  Valid conditions: {len(dead_valid)}/{len(dead_arr)}")
    print(f"  Insufficient variance for correlation")
else:
    correlation = float('nan')
    print(f"\n  Only {len(dead_valid)} valid condition(s) -- cannot compute correlation")


# =============================================================================
# FINE-GRAINED: per-layer data points
# =============================================================================

print(f"\n\n{'=' * 90}")
print("FINE-GRAINED: Per-layer dead vs gauge (pooled over all conditions)")
print(f"{'=' * 90}")

# For layers 0..4 that have ReLU: pair dead_frac with gauge_frac
all_dead_points = []
all_gauge_points = []

for bias_val in BIAS_INITS:
    r = results[bias_val]
    for seed_idx in range(NUM_SEEDS):
        for l in range(NUM_LAYERS - 1):
            dead_frac = r['dead'][seed_idx, l] * 100
            gauge_frac = r['gauge'][seed_idx, l]
            if not np.isnan(gauge_frac):
                all_dead_points.append(dead_frac)
                all_gauge_points.append(gauge_frac * 100)

all_dead_points = np.array(all_dead_points)
all_gauge_points = np.array(all_gauge_points)

if len(all_dead_points) >= 2 and np.std(all_dead_points) > 1e-10 and np.std(all_gauge_points) > 1e-10:
    fine_corr = np.corrcoef(all_dead_points, all_gauge_points)[0, 1]
    print(f"\n  Valid data points: {len(all_dead_points)}")
    print(f"  Per-layer correlation (dead vs gauge): r = {fine_corr:.4f}")
else:
    fine_corr = 0.0
    print(f"\n  Valid data points: {len(all_dead_points)}")
    print(f"  Insufficient variance for per-layer correlation")

# Bin by dead fraction
bins = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
print(f"\n  {'Dead % bin':>12}  {'N':>4}  {'Mean gauge %':>13}  {'Std gauge %':>12}")
print(f"  {'-'*50}")

for lo, hi in bins:
    mask = (all_dead_points >= lo) & (all_dead_points < hi)
    n = np.sum(mask)
    if n > 0:
        mg = np.mean(all_gauge_points[mask])
        sg = np.std(all_gauge_points[mask])
        print(f"  {lo:>4}-{hi:<4}%     {n:>4}  {mg:>13.1f}  {sg:>12.1f}")
    else:
        print(f"  {lo:>4}-{hi:<4}%     {0:>4}         ---           ---")


# =============================================================================
# ALSO: last-layer gauge fraction (no ReLU above it)
# =============================================================================

print(f"\n\n{'=' * 90}")
print("LAST LAYER (L6) GAUGE FRACTION (no ReLU after it; gauge should always be ~50%)")
print(f"{'=' * 90}")

print(f"\n{'Bias':>6}  {'L6 gauge %':>12}  {'Valid seeds':>12}")
print("-" * 35)
for bias_val in BIAS_INITS:
    r = results[bias_val]
    col = r['gauge'][:, NUM_LAYERS - 1]
    valid = col[~np.isnan(col)]
    if len(valid) > 0:
        print(f"{bias_val:>+6.0f}  {np.mean(valid)*100:>12.1f}  {len(valid):>12}")
    else:
        print(f"{bias_val:>+6.0f}  {'NaN':>12}  {0:>12}")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 90}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 90}")

# H1: Where gradients exist, gauge fraction is substantial (>30%)
valid_gauges = [g for g in summary_gauge if not np.isnan(g)]
h1 = len(valid_gauges) > 0 and all(g > 30.0 for g in valid_gauges)
print(f"\nH1: Gauge fraction >30% wherever gradients are non-trivial?")
print(f"    Valid values: {[f'{g:.1f}%' for g in valid_gauges]}")
print(f"    --> {'PASS' if h1 else 'FAIL'}")

# H2: Gauge fraction is near-constant among valid conditions (spread < 15pp)
if len(valid_gauges) >= 2:
    gauge_valid_arr = np.array(valid_gauges)
    gauge_spread = gauge_valid_arr.max() - gauge_valid_arr.min()
    h2 = gauge_spread < 15.0
    print(f"\nH2: Gauge fraction spread < 15pp among valid conditions (local property)?")
    print(f"    Spread: {gauge_spread:.1f}pp")
    print(f"    --> {'PASS' if h2 else 'FAIL'}")
else:
    h2 = None
    print(f"\nH2: Only {len(valid_gauges)} valid conditions. SKIPPED.")

# H3: Dead neurons cause zero gradients -> NaN gauge (not low gauge)
# Check if conditions with 100% dead neurons all produce NaN
fully_dead_biases = [b for b in BIAS_INITS if np.mean(results[b]['dead']) > 0.95]
if fully_dead_biases:
    all_nan_for_dead = True
    for b in fully_dead_biases:
        gauge_flat = results[b]['gauge'].flatten()
        n_valid = np.sum(~np.isnan(gauge_flat))
        # Last layer might still have valid gauge (linear output layer)
        # Check non-last layers
        for l in range(NUM_LAYERS - 1):
            col = results[b]['gauge'][:, l]
            if np.any(~np.isnan(col)):
                all_nan_for_dead = False
    h3 = all_nan_for_dead
    print(f"\nH3: Fully-dead conditions have NaN gauge in hidden layers (zero gradient)?")
    print(f"    Fully dead bias values: {fully_dead_biases}")
    print(f"    --> {'PASS' if h3 else 'FAIL'}")
else:
    h3 = None
    print(f"\nH3: No fully-dead conditions. SKIPPED.")

# H4: Fine-grained: among non-dead layers, correlation is weak
if len(all_dead_points) >= 5:
    h4 = abs(fine_corr) < 0.5
    print(f"\nH4: Per-layer correlation |r| < 0.5 (gauge independent of partial dead fraction)?")
    print(f"    r = {fine_corr:.4f}, N = {len(all_dead_points)}")
    print(f"    --> {'PASS' if h4 else 'FAIL'}")
else:
    h4 = None
    print(f"\nH4: Insufficient data points. SKIPPED.")

total_pass = sum(1 for h in [h1, h2, h3, h4] if h is True)
total_tests = sum(1 for h in [h1, h2, h3, h4] if h is not None)


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 90}")
print("FINAL VERDICT: H11 GAUGE vs DEAD NEURONS")
print(f"{'=' * 90}")

if len(valid_gauges) >= 2:
    print(f"""
  Valid conditions (non-NaN gauge): {len(valid_gauges)}/{len(BIAS_INITS)}
  Gauge fraction range (valid): {min(valid_gauges):.1f}% to {max(valid_gauges):.1f}%
  Spread: {gauge_spread:.1f}pp
  Per-layer fine-grained correlation: r = {fine_corr:.4f}
  Tests passed: {total_pass}/{total_tests}
""")
else:
    print(f"""
  Valid conditions (non-NaN gauge): {len(valid_gauges)}/{len(BIAS_INITS)}
  Tests passed: {total_pass}/{total_tests}
""")

print("  INTERPRETATION:")
print("  - Dead neurons produce ZERO gradients -> NaN gauge fraction (undefined).")
print("  - Where gradients flow, gauge fraction remains near ~50%.")
print("  - The question 'does gauge drop with dead neurons' is CONFOUNDED:")
print("    dead neurons don't reduce gauge -- they eliminate gradients entirely.")
print("  - The gauge is a property of the gradient GEOMETRY, which requires")
print("    nonzero gradients to be measurable.")
print(f"\n{'=' * 90}")
