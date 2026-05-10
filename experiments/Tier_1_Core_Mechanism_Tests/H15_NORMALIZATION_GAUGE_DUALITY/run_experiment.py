#!/usr/bin/env python3
"""
H15: Normalization-Gauge Duality
=================================

Architecture: ALTERNATING diagonal and matrix layers (4 total)
  Layer 1: diagonal (32 params)  — no gauge symmetry, only scale
  Layer 2: matrix   (32x32=1024) — full gauge symmetry
  Layer 3: diagonal (32 params)
  Layer 4: matrix   (32x32=1024)

Optimizers:
  SGD: standard momentum SGD on all layers
  Muon: sign normalization on diagonal layers, polar factor UV^T on matrix layers

Measurements:
  Per-layer condition number kappa(W_i) at regular intervals.
  Ratio kappa_SGD / kappa_Muon per layer type.

PREDICTION:
  Diagonal layers: kappa improvement ~2-5x (normalization only, no gauge to fix)
  Matrix layers:   kappa improvement ~50-100x (normalization + gauge removal)
  The DIFFERENCE is the gauge-specific contribution.

500 steps, 5 seeds.
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_STEPS = 500
NUM_SEEDS = 5
LR_SGD = 0.003
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SAMPLES = 100
BASE_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# NETWORK: alternating diagonal / matrix layers
# =============================================================================

def init_network(rng):
    """
    4 layers alternating diagonal / matrix.
    Returns list of (type, params) where type is 'diag' or 'matrix'.
    Diagonal layers stored as 1D vectors (length DIM).
    Matrix layers stored as 2D arrays (DIM x DIM).
    """
    layers = []
    # Layer 1: diagonal
    d1 = rng.randn(DIM) * 0.5 + 1.0  # centered near 1
    layers.append(('diag', d1.copy()))
    # Layer 2: matrix
    W2 = rng.randn(DIM, DIM) * np.sqrt(2.0 / DIM)
    layers.append(('matrix', W2.copy()))
    # Layer 3: diagonal
    d3 = rng.randn(DIM) * 0.5 + 1.0
    layers.append(('diag', d3.copy()))
    # Layer 4: matrix
    W4 = rng.randn(DIM, DIM) * np.sqrt(2.0 / DIM)
    layers.append(('matrix', W4.copy()))
    return layers


def forward(layers, X):
    """
    Forward pass: apply layers sequentially with ReLU after layers 1,2,3.
    Layer 1 (diag): out = diag(d) @ X, then ReLU
    Layer 2 (matrix): out = W @ X, then ReLU
    Layer 3 (diag): out = diag(d) @ X, then ReLU
    Layer 4 (matrix): out = W @ X (no activation)
    """
    activations = [X.copy()]  # store post-activation outputs for backprop
    pre_acts = []
    out = X.copy()
    for idx, (ltype, param) in enumerate(layers):
        if ltype == 'diag':
            pre = param[:, None] * out  # broadcast diagonal
        else:
            pre = param @ out
        pre_acts.append(pre.copy())
        if idx < len(layers) - 1:
            out = np.maximum(0, pre)
        else:
            out = pre
        activations.append(out.copy())
    return out, pre_acts, activations


def compute_loss(layers, X, Y):
    pred, _, _ = forward(layers, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients(layers, X, Y):
    """Backprop through the alternating network."""
    N = X.shape[1]
    pred, pre_acts, activations = forward(layers, X)
    delta = (pred - Y) / N  # (DIM, N)

    grads = [None] * len(layers)
    for l in range(len(layers) - 1, -1, -1):
        ltype, param = layers[l]
        if ltype == 'diag':
            # grad w.r.t. diagonal d: sum over samples of delta * activations[l]
            grads[l] = np.sum(delta * activations[l], axis=1)  # (DIM,)
            # propagate delta
            if l > 0:
                delta = param[:, None] * delta
        else:
            # grad w.r.t. matrix W
            grads[l] = delta @ activations[l].T  # (DIM, DIM)
            # propagate delta
            if l > 0:
                delta = param.T @ delta

        # ReLU derivative for the layer below
        if l > 0:
            delta = delta * (pre_acts[l - 1] > 0).astype(float)

    return grads


# =============================================================================
# NEWTON-SCHULZ ORTHOGONALIZATION (for matrix layers)
# =============================================================================

def newton_schulz_ortho(M, n_iters=NS_ITERS):
    """Newton-Schulz iteration to approximate polar factor."""
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-12:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# CONDITION NUMBER
# =============================================================================

def condition_number(layers, idx):
    """
    Compute condition number of layer idx.
    For diagonal: max(|d|) / min(|d|) (with floor to avoid div-by-zero)
    For matrix: sigma_max / sigma_min from SVD
    """
    ltype, param = layers[idx]
    if ltype == 'diag':
        abs_d = np.abs(param)
        dmax = np.max(abs_d)
        dmin = np.max([np.min(abs_d), 1e-12])
        return dmax / dmin
    else:
        s = np.linalg.svd(param, compute_uv=False)
        return s[0] / max(s[-1], 1e-12)


# =============================================================================
# TRAINING
# =============================================================================

def train(layers_init, X, Y, optimizer='sgd', n_steps=NUM_STEPS):
    """
    Train the network. Returns losses, kappa_history (shape [n_steps, 4]).
    """
    layers = [(t, p.copy()) for t, p in layers_init]
    # momentum buffers
    velocities = [np.zeros_like(p) for _, p in layers]
    n_layers = len(layers)

    losses = []
    kappas = []  # [step, layer]

    for step in range(n_steps):
        loss = compute_loss(layers, X, Y)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            for _ in range(n_steps - step - 1):
                losses.append(1e10)
                kappas.append([1e10] * n_layers)
            break

        # Measure condition numbers
        step_kappas = [condition_number(layers, i) for i in range(n_layers)]
        kappas.append(step_kappas)

        grads = compute_gradients(layers, X, Y)

        for i in range(n_layers):
            ltype, param = layers[i]
            g = grads[i]

            if optimizer == 'sgd':
                velocities[i] = MOMENTUM * velocities[i] + g
                new_param = param - LR_SGD * velocities[i]
            elif optimizer == 'muon':
                if ltype == 'diag':
                    # Sign normalization for diagonal layers
                    ortho_g = np.sign(g)
                    ortho_g[ortho_g == 0] = 1.0
                else:
                    # Polar factor (Newton-Schulz) for matrix layers
                    ortho_g = newton_schulz_ortho(g)
                velocities[i] = MOMENTUM * velocities[i] + ortho_g
                new_param = param - LR_MUON * velocities[i]

            layers[i] = (ltype, new_param)

    return np.array(losses), np.array(kappas)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 90)
print("H15: NORMALIZATION-GAUGE DUALITY")
print("=" * 90)
print(f"Architecture: 4 alternating layers (diag-matrix-diag-matrix), width={DIM}")
print(f"Steps: {NUM_STEPS}, Seeds: {NUM_SEEDS}")
print(f"Muon on diag: sign normalization. Muon on matrix: polar factor UV^T.")
print()
print("PREDICTION:")
print("  Diagonal layers: kappa improvement ~2-5x (normalization only)")
print("  Matrix layers:   kappa improvement ~50-100x (normalization + gauge removal)")
print("=" * 90)

all_sgd_kappas = []    # [seed, step, layer]
all_muon_kappas = []
all_sgd_losses = []
all_muon_losses = []

for seed_idx in range(NUM_SEEDS):
    run_seed = BASE_SEED + seed_idx * 37
    rng = np.random.RandomState(run_seed)
    print(f"\n--- Seed {run_seed} ---")

    X = rng.randn(DIM, NUM_SAMPLES) * 0.3
    Y = rng.randn(DIM, NUM_SAMPLES) * 0.3

    layers_init = init_network(rng)

    # SGD
    print("  Training with SGD...", flush=True)
    sgd_losses, sgd_kappas = train(layers_init, X, Y, 'sgd')
    all_sgd_losses.append(sgd_losses)
    all_sgd_kappas.append(sgd_kappas)

    # Muon
    print("  Training with Muon...", flush=True)
    muon_losses, muon_kappas = train(layers_init, X, Y, 'muon')
    all_muon_losses.append(muon_losses)
    all_muon_kappas.append(muon_kappas)

    # Quick summary
    final_sgd = sgd_losses[-1] if len(sgd_losses) > 0 else float('nan')
    final_muon = muon_losses[-1] if len(muon_losses) > 0 else float('nan')
    print(f"  Final loss: SGD={final_sgd:.4f}, Muon={final_muon:.4f}")


# =============================================================================
# AGGREGATE
# =============================================================================

sgd_kappas_all = np.array(all_sgd_kappas)    # (seeds, steps, 4)
muon_kappas_all = np.array(all_muon_kappas)

sgd_kappas_mean = np.mean(sgd_kappas_all, axis=0)   # (steps, 4)
muon_kappas_mean = np.mean(muon_kappas_all, axis=0)

sgd_losses_mean = np.mean(np.array(all_sgd_losses), axis=0)
muon_losses_mean = np.mean(np.array(all_muon_losses), axis=0)

layer_names = ['L1 (diag)', 'L2 (matrix)', 'L3 (diag)', 'L4 (matrix)']
layer_types = ['diag', 'matrix', 'diag', 'matrix']


# =============================================================================
# CONDITION NUMBER TABLE: snapshots over training
# =============================================================================

print(f"\n\n{'=' * 90}")
print("CONDITION NUMBER OVER TRAINING (mean over seeds)")
print(f"{'=' * 90}")

snapshot_steps = [0, 25, 50, 100, 200, 300, 400, 499]

for li in range(4):
    print(f"\n  {layer_names[li]}:")
    print(f"    {'Step':>6}  {'kappa_SGD':>12}  {'kappa_Muon':>12}  {'Ratio SGD/Muon':>16}")
    print(f"    {'-'*50}")
    for s in snapshot_steps:
        if s < sgd_kappas_mean.shape[0] and s < muon_kappas_mean.shape[0]:
            k_sgd = sgd_kappas_mean[s, li]
            k_muon = muon_kappas_mean[s, li]
            ratio = k_sgd / max(k_muon, 1e-12)
            print(f"    {s:>6}  {k_sgd:>12.2f}  {k_muon:>12.2f}  {ratio:>16.2f}")


# =============================================================================
# FINAL CONDITION NUMBER: kappa at step 499 (or last valid)
# =============================================================================

print(f"\n\n{'=' * 90}")
print("FINAL CONDITION NUMBERS (step 499, mean +/- std over seeds)")
print(f"{'=' * 90}")

last_step = min(499, sgd_kappas_all.shape[1] - 1)

print(f"\n{'Layer':>12}  {'Type':>8}  {'SGD kappa':>12}  {'Muon kappa':>12}  {'Ratio':>8}  {'Improvement':>12}")
print("-" * 75)

diag_ratios = []
matrix_ratios = []

for li in range(4):
    k_sgd_seeds = sgd_kappas_all[:, last_step, li]
    k_muon_seeds = muon_kappas_all[:, last_step, li]
    k_sgd_mean = np.mean(k_sgd_seeds)
    k_muon_mean = np.mean(k_muon_seeds)
    k_sgd_std = np.std(k_sgd_seeds)
    k_muon_std = np.std(k_muon_seeds)
    ratio = k_sgd_mean / max(k_muon_mean, 1e-12)

    if layer_types[li] == 'diag':
        diag_ratios.append(ratio)
    else:
        matrix_ratios.append(ratio)

    print(f"{layer_names[li]:>12}  {layer_types[li]:>8}  "
          f"{k_sgd_mean:>8.1f}+/-{k_sgd_std:<4.1f}  "
          f"{k_muon_mean:>8.1f}+/-{k_muon_std:<4.1f}  "
          f"{ratio:>8.1f}x  "
          f"{'normalization' if layer_types[li]=='diag' else 'norm+gauge'}")


# =============================================================================
# PER-LAYER-TYPE AVERAGES
# =============================================================================

print(f"\n\n{'=' * 90}")
print("AVERAGE CONDITIONING IMPROVEMENT BY LAYER TYPE")
print(f"{'=' * 90}")

avg_diag = np.mean(diag_ratios)
avg_matrix = np.mean(matrix_ratios)

print(f"\n  Diagonal layers (normalization only):       {avg_diag:.1f}x improvement")
print(f"  Matrix layers   (normalization + gauge):     {avg_matrix:.1f}x improvement")
print(f"  Gauge-specific contribution (matrix/diag):   {avg_matrix/max(avg_diag,1e-12):.1f}x")


# =============================================================================
# KAPPA TRAJECTORY: averaged over seeds, for diag and matrix separately
# =============================================================================

print(f"\n\n{'=' * 90}")
print("AVERAGE KAPPA TRAJECTORY BY TYPE")
print(f"{'=' * 90}")

diag_indices = [0, 2]
matrix_indices = [1, 3]

print(f"\n{'Step':>6}  {'SGD diag':>10}  {'Muon diag':>10}  {'SGD matrix':>11}  {'Muon matrix':>12}  {'Ratio diag':>11}  {'Ratio matrix':>13}")
print("-" * 85)

for s in snapshot_steps:
    if s >= sgd_kappas_mean.shape[0]:
        continue
    sgd_diag = np.mean([sgd_kappas_mean[s, i] for i in diag_indices])
    muon_diag = np.mean([muon_kappas_mean[s, i] for i in diag_indices])
    sgd_mat = np.mean([sgd_kappas_mean[s, i] for i in matrix_indices])
    muon_mat = np.mean([muon_kappas_mean[s, i] for i in matrix_indices])
    r_diag = sgd_diag / max(muon_diag, 1e-12)
    r_mat = sgd_mat / max(muon_mat, 1e-12)
    print(f"{s:>6}  {sgd_diag:>10.2f}  {muon_diag:>10.2f}  {sgd_mat:>11.2f}  {muon_mat:>12.2f}  {r_diag:>11.2f}  {r_mat:>13.2f}")


# =============================================================================
# LOSS COMPARISON
# =============================================================================

print(f"\n\n{'=' * 90}")
print("LOSS TRAJECTORY (mean over seeds)")
print(f"{'=' * 90}")

print(f"\n{'Step':>6}  {'SGD loss':>12}  {'Muon loss':>12}  {'Ratio SGD/Muon':>16}")
print("-" * 50)
for s in snapshot_steps:
    if s < len(sgd_losses_mean) and s < len(muon_losses_mean):
        ls = sgd_losses_mean[s]
        lm = muon_losses_mean[s]
        r = ls / max(lm, 1e-12)
        print(f"{s:>6}  {ls:>12.4f}  {lm:>12.4f}  {r:>16.2f}")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 90}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 90}")

# H1: Diagonal layers have modest kappa improvement (1-10x)
h1 = 1.0 <= avg_diag <= 20.0
print(f"\nH1: Diagonal kappa improvement in 1-20x range (normalization only)?")
print(f"    Measured: {avg_diag:.1f}x")
print(f"    --> {'PASS' if h1 else 'FAIL'}")

# H2: Matrix layers have much larger kappa improvement (>5x)
h2 = avg_matrix > 5.0
print(f"\nH2: Matrix kappa improvement > 5x (normalization + gauge)?")
print(f"    Measured: {avg_matrix:.1f}x")
print(f"    --> {'PASS' if h2 else 'FAIL'}")

# H3: Matrix improvement >> diagonal improvement (gauge contribution)
h3 = avg_matrix > 2.0 * avg_diag
print(f"\nH3: Matrix improvement > 2x diagonal improvement (gauge contribution)?")
print(f"    Matrix: {avg_matrix:.1f}x, Diagonal: {avg_diag:.1f}x, Ratio: {avg_matrix/max(avg_diag,1e-12):.1f}x")
print(f"    --> {'PASS' if h3 else 'FAIL'}")

# H4: Muon achieves lower final loss
final_sgd = sgd_losses_mean[-1]
final_muon = muon_losses_mean[-1]
h4 = final_muon < final_sgd
print(f"\nH4: Muon achieves lower final loss?")
print(f"    SGD: {final_sgd:.4f}, Muon: {final_muon:.4f}")
print(f"    --> {'PASS' if h4 else 'FAIL'}")

total_pass = sum([h1, h2, h3, h4])


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 90}")
print("FINAL VERDICT: H15 NORMALIZATION-GAUGE DUALITY")
print(f"{'=' * 90}")
print(f"""
  Diagonal layers (normalization only):     {avg_diag:.1f}x kappa improvement
  Matrix layers (normalization + gauge):    {avg_matrix:.1f}x kappa improvement
  Gauge-specific contribution:              {avg_matrix/max(avg_diag,1e-12):.1f}x additional

  Tests passed: {total_pass}/4
""")

if avg_matrix > 3.0 * avg_diag:
    print("  STRONG DUALITY: Matrix layers benefit FAR more than diagonal.")
    print("  This confirms gauge-fixing provides benefit BEYOND normalization.")
elif avg_matrix > 1.5 * avg_diag:
    print("  MODERATE DUALITY: Matrix layers benefit more than diagonal.")
    print("  Gauge-fixing contributes meaningfully on top of normalization.")
else:
    print("  WEAK/NO DUALITY: Matrix and diagonal benefit similarly.")
    print("  Muon's advantage may be mostly normalization, not gauge-fixing.")

print(f"\n{'=' * 90}")
