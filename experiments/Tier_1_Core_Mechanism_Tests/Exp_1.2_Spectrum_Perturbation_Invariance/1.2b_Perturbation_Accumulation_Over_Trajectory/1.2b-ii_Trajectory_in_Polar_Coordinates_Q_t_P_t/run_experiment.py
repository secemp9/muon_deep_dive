#!/usr/bin/env python3
"""
1.2b-ii: Trajectory in Polar Coordinates -- (Q_t, P_t) Decomposition
=====================================================================

At each step, decompose W_t = Q_t P_t (polar decomposition).
  Q_t lives on the Stiefel manifold (orthogonal matrices).
  P_t lives in Sym+ (symmetric positive semi-definite matrices).

Track how much of each weight update changes Q vs P:
  ||DeltaQ|| / ||DeltaW||   (orientation change fraction)
  ||DeltaP|| / ||DeltaW||   (spectrum change fraction)

Also track cumulative drift from initialization:
  ||Q_t - Q_0||_F   (total orientation drift)
  ||P_t - P_0||_F   (total spectrum drift)

HYPOTHESIS (to be tested, may be wrong given 1.2b-i results):
  Under Muon: ||DeltaQ||/||DeltaW|| ~ 1, ||DeltaP||/||DeltaW|| ~ 0
    (movement is predominantly in orientation, not spectrum)
  Under SGD:  both ratios are O(1)
    (movement in both orientation and spectrum)

CRITICAL CONTEXT from 1.2b-i:
  - Muon is MORE chaotic in weight space (higher Lyapunov) but more stable
    in loss space
  - The benefit is DIRECTIONAL, not stability
  - Neither optimizer distinguishes gauge from physical in Lyapunov terms
  - So this experiment may show something different from what was hypothesized

Setup: 4-layer deep linear net, 32x32, quadratic loss, 300 steps.
"""

import numpy as np
import os

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 300
BATCH_SIZE = 64
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5

# Report steps
REPORT_STEPS = [50, 100, 200, 300]

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


def polar_decomposition(W):
    """
    Compute the polar decomposition W = Q P.
    Q is orthogonal (or unitary), P is symmetric positive semi-definite.
    Uses SVD: W = U S V^T => Q = U V^T, P = V S V^T.
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    Q = U @ Vt
    P = Vt.T @ np.diag(S) @ Vt
    return Q, P


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


# =============================================================================
# OPTIMIZER STEP FUNCTIONS
# =============================================================================

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
# POLAR COORDINATE TRACKING ENGINE
# =============================================================================

def run_and_track_polar(optimizer, lr, num_steps):
    """
    Run optimizer for num_steps, at each step decompose each layer's W_t = Q_t P_t.
    Track per-step and cumulative polar decomposition metrics.

    Returns dict with per-layer and averaged metrics.
    """
    np.random.seed(42)
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros_like(w) for w in weights]

    # Initial polar decompositions
    Q_prev = []
    P_prev = []
    Q_init = []
    P_init = []
    for i in range(NUM_LAYERS):
        Q, P = polar_decomposition(weights[i])
        Q_prev.append(Q.copy())
        P_prev.append(P.copy())
        Q_init.append(Q.copy())
        P_init.append(P.copy())

    # Per-layer tracking arrays
    # Per-step ratios: ||DeltaQ||/||DeltaW|| and ||DeltaP||/||DeltaW||
    dQ_ratio = np.zeros((NUM_LAYERS, num_steps))  # ||DeltaQ||/||DeltaW||
    dP_ratio = np.zeros((NUM_LAYERS, num_steps))  # ||DeltaP||/||DeltaW||

    # Per-step absolute norms
    dQ_norm = np.zeros((NUM_LAYERS, num_steps))
    dP_norm = np.zeros((NUM_LAYERS, num_steps))
    dW_norm = np.zeros((NUM_LAYERS, num_steps))

    # Cumulative drift from initialization
    cum_Q_drift = np.zeros((NUM_LAYERS, num_steps))
    cum_P_drift = np.zeros((NUM_LAYERS, num_steps))
    cum_W_drift = np.zeros((NUM_LAYERS, num_steps))

    # Loss tracking
    losses = np.zeros(num_steps + 1)
    losses[0] = compute_loss(weights, X_data, W_target)

    W_prev = [w.copy() for w in weights]
    W_init = [w.copy() for w in weights]

    for step in range(num_steps):
        # Take optimizer step
        if optimizer == 'sgd':
            weights, velocities = sgd_step(weights, velocities, lr)
        elif optimizer == 'muon':
            weights, velocities = muon_step(weights, velocities, lr)

        losses[step + 1] = compute_loss(weights, X_data, W_target)

        # Check for divergence
        if np.isnan(losses[step + 1]) or losses[step + 1] > 1e10:
            print(f"    WARNING: {optimizer} diverged at step {step + 1}")
            # Fill remaining with NaN
            dQ_ratio[:, step:] = np.nan
            dP_ratio[:, step:] = np.nan
            dQ_norm[:, step:] = np.nan
            dP_norm[:, step:] = np.nan
            dW_norm[:, step:] = np.nan
            cum_Q_drift[:, step:] = np.nan
            cum_P_drift[:, step:] = np.nan
            cum_W_drift[:, step:] = np.nan
            losses[step + 1:] = np.nan
            break

        for i in range(NUM_LAYERS):
            # Polar decomposition of current weight
            Q_curr, P_curr = polar_decomposition(weights[i])

            # Step differences
            delta_W = weights[i] - W_prev[i]
            delta_Q = Q_curr - Q_prev[i]
            delta_P = P_curr - P_prev[i]

            norm_dW = np.linalg.norm(delta_W, 'fro')
            norm_dQ = np.linalg.norm(delta_Q, 'fro')
            norm_dP = np.linalg.norm(delta_P, 'fro')

            dW_norm[i, step] = norm_dW
            dQ_norm[i, step] = norm_dQ
            dP_norm[i, step] = norm_dP

            if norm_dW > 1e-15:
                dQ_ratio[i, step] = norm_dQ / norm_dW
                dP_ratio[i, step] = norm_dP / norm_dW
            else:
                dQ_ratio[i, step] = np.nan
                dP_ratio[i, step] = np.nan

            # Cumulative drift from initialization
            cum_Q_drift[i, step] = np.linalg.norm(Q_curr - Q_init[i], 'fro')
            cum_P_drift[i, step] = np.linalg.norm(P_curr - P_init[i], 'fro')
            cum_W_drift[i, step] = np.linalg.norm(weights[i] - W_init[i], 'fro')

            # Update previous
            Q_prev[i] = Q_curr.copy()
            P_prev[i] = P_curr.copy()

        W_prev = [w.copy() for w in weights]

    return {
        'dQ_ratio': dQ_ratio,       # (NUM_LAYERS, num_steps)
        'dP_ratio': dP_ratio,
        'dQ_norm': dQ_norm,
        'dP_norm': dP_norm,
        'dW_norm': dW_norm,
        'cum_Q_drift': cum_Q_drift,
        'cum_P_drift': cum_P_drift,
        'cum_W_drift': cum_W_drift,
        'losses': losses,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 100)
print("1.2b-ii: TRAJECTORY IN POLAR COORDINATES -- (Q_t, P_t) DECOMPOSITION")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer deep linear net (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}")
print(f"Report at steps: {REPORT_STEPS}")
print("=" * 100)

# Find stable SGD learning rate
lr_sgd = find_stable_lr_sgd()
print(f"\nSGD learning rate (max stable): {lr_sgd}")
print(f"Muon learning rate (fixed):     {LR_MUON}")

# Quick sanity check
np.random.seed(42)
w_test = init_weights(NUM_LAYERS)
loss_init = compute_loss(w_test, X_data, W_target)
print(f"\nInitial loss: {loss_init:.6e}")

# Verify polar decomposition works
print("\nVerifying polar decomposition on initial weights...")
for i, W in enumerate(w_test):
    Q, P = polar_decomposition(W)
    recon_err = np.linalg.norm(W - Q @ P, 'fro') / np.linalg.norm(W, 'fro')
    Q_orth_err = np.linalg.norm(Q.T @ Q - np.eye(DIM), 'fro')
    P_sym_err = np.linalg.norm(P - P.T, 'fro')
    P_eigmin = np.min(np.linalg.eigvalsh(P))
    print(f"  Layer {i}: recon_err={recon_err:.2e}, Q_orth_err={Q_orth_err:.2e}, "
          f"P_sym_err={P_sym_err:.2e}, P_min_eig={P_eigmin:.4f}")


# =============================================================================
# RUN BOTH OPTIMIZERS
# =============================================================================

print(f"\n{'=' * 100}")
print("RUNNING OPTIMIZERS AND TRACKING POLAR DECOMPOSITION")
print("=" * 100)

print("\n  Running SGD...", flush=True)
results_sgd = run_and_track_polar('sgd', lr_sgd, NUM_STEPS)
print(f"    Final loss: {results_sgd['losses'][-1]:.6e}")

print("\n  Running Muon...", flush=True)
results_muon = run_and_track_polar('muon', LR_MUON, NUM_STEPS)
print(f"    Final loss: {results_muon['losses'][-1]:.6e}")


# =============================================================================
# TABLE 1: PER-STEP RATIOS AT REPORT STEPS (averaged across layers)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 1: PER-STEP ORIENTATION vs SPECTRUM CHANGE RATIOS")
print("        (averaged across layers, averaged over 10-step window around report step)")
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD ||dQ||/||dW||':>18} | {'SGD ||dP||/||dW||':>18} | "
      f"{'Muon ||dQ||/||dW||':>19} | {'Muon ||dP||/||dW||':>19}")
print("-" * 90)

for step in REPORT_STEPS:
    # Average over a window of 10 steps centered on the report step
    lo = max(0, step - 5)
    hi = min(NUM_STEPS, step + 5)

    sgd_dQ_r = np.nanmean(results_sgd['dQ_ratio'][:, lo:hi])
    sgd_dP_r = np.nanmean(results_sgd['dP_ratio'][:, lo:hi])
    muon_dQ_r = np.nanmean(results_muon['dQ_ratio'][:, lo:hi])
    muon_dP_r = np.nanmean(results_muon['dP_ratio'][:, lo:hi])

    print(f"{step:6d} | {sgd_dQ_r:18.6f} | {sgd_dP_r:18.6f} | "
          f"{muon_dQ_r:19.6f} | {muon_dP_r:19.6f}")

# Overall averages
sgd_dQ_overall = np.nanmean(results_sgd['dQ_ratio'])
sgd_dP_overall = np.nanmean(results_sgd['dP_ratio'])
muon_dQ_overall = np.nanmean(results_muon['dQ_ratio'])
muon_dP_overall = np.nanmean(results_muon['dP_ratio'])

print("-" * 90)
print(f"{'ALL':>6} | {sgd_dQ_overall:18.6f} | {sgd_dP_overall:18.6f} | "
      f"{muon_dQ_overall:19.6f} | {muon_dP_overall:19.6f}")


# =============================================================================
# TABLE 2: CUMULATIVE DRIFT AT REPORT STEPS (averaged across layers)
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 2: CUMULATIVE DRIFT FROM INITIALIZATION (averaged across layers)")
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD ||Q-Q0||':>14} | {'SGD ||P-P0||':>14} | {'SGD ||W-W0||':>14} | "
      f"{'Muon ||Q-Q0||':>14} | {'Muon ||P-P0||':>14} | {'Muon ||W-W0||':>14}")
print("-" * 105)

for step in REPORT_STEPS:
    idx = step - 1  # 0-indexed
    sgd_Qd = np.nanmean(results_sgd['cum_Q_drift'][:, idx])
    sgd_Pd = np.nanmean(results_sgd['cum_P_drift'][:, idx])
    sgd_Wd = np.nanmean(results_sgd['cum_W_drift'][:, idx])
    muon_Qd = np.nanmean(results_muon['cum_Q_drift'][:, idx])
    muon_Pd = np.nanmean(results_muon['cum_P_drift'][:, idx])
    muon_Wd = np.nanmean(results_muon['cum_W_drift'][:, idx])

    print(f"{step:6d} | {sgd_Qd:14.6f} | {sgd_Pd:14.6f} | {sgd_Wd:14.6f} | "
          f"{muon_Qd:14.6f} | {muon_Pd:14.6f} | {muon_Wd:14.6f}")


# =============================================================================
# TABLE 3: RATIO OF CUMULATIVE Q DRIFT TO P DRIFT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 3: RATIO OF CUMULATIVE Q-DRIFT TO P-DRIFT (averaged across layers)")
print("         Q-drift / P-drift > 1 means more orientation change than spectrum change")
print("=" * 100)

print(f"\n{'Step':>6} | {'SGD Q/P ratio':>14} | {'Muon Q/P ratio':>15} | {'SGD Q/(Q+P)':>12} | {'Muon Q/(Q+P)':>13}")
print("-" * 72)

for step in REPORT_STEPS:
    idx = step - 1
    sgd_Qd = np.nanmean(results_sgd['cum_Q_drift'][:, idx])
    sgd_Pd = np.nanmean(results_sgd['cum_P_drift'][:, idx])
    muon_Qd = np.nanmean(results_muon['cum_Q_drift'][:, idx])
    muon_Pd = np.nanmean(results_muon['cum_P_drift'][:, idx])

    sgd_qp = sgd_Qd / sgd_Pd if sgd_Pd > 1e-15 else np.inf
    muon_qp = muon_Qd / muon_Pd if muon_Pd > 1e-15 else np.inf
    sgd_frac = sgd_Qd / (sgd_Qd + sgd_Pd) if (sgd_Qd + sgd_Pd) > 1e-15 else np.nan
    muon_frac = muon_Qd / (muon_Qd + muon_Pd) if (muon_Qd + muon_Pd) > 1e-15 else np.nan

    print(f"{step:6d} | {sgd_qp:14.4f} | {muon_qp:15.4f} | {sgd_frac:12.4f} | {muon_frac:13.4f}")


# =============================================================================
# TABLE 4: PER-LAYER BREAKDOWN AT STEP 300
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 4: PER-LAYER BREAKDOWN AT STEP 300")
print("=" * 100)

print(f"\n  SGD (lr={lr_sgd}):")
print(f"  {'Layer':>6} | {'||dQ||/||dW||':>14} | {'||dP||/||dW||':>14} | "
      f"{'cum ||Q-Q0||':>14} | {'cum ||P-P0||':>14} | {'Q/(Q+P)':>10}")
print("  " + "-" * 85)

for layer in range(NUM_LAYERS):
    dQr = np.nanmean(results_sgd['dQ_ratio'][layer, :])
    dPr = np.nanmean(results_sgd['dP_ratio'][layer, :])
    Qd = results_sgd['cum_Q_drift'][layer, -1]
    Pd = results_sgd['cum_P_drift'][layer, -1]
    frac = Qd / (Qd + Pd) if (Qd + Pd) > 1e-15 else np.nan
    print(f"  {layer:6d} | {dQr:14.6f} | {dPr:14.6f} | {Qd:14.6f} | {Pd:14.6f} | {frac:10.4f}")

print(f"\n  Muon (lr={LR_MUON}):")
print(f"  {'Layer':>6} | {'||dQ||/||dW||':>14} | {'||dP||/||dW||':>14} | "
      f"{'cum ||Q-Q0||':>14} | {'cum ||P-P0||':>14} | {'Q/(Q+P)':>10}")
print("  " + "-" * 85)

for layer in range(NUM_LAYERS):
    dQr = np.nanmean(results_muon['dQ_ratio'][layer, :])
    dPr = np.nanmean(results_muon['dP_ratio'][layer, :])
    Qd = results_muon['cum_Q_drift'][layer, -1]
    Pd = results_muon['cum_P_drift'][layer, -1]
    frac = Qd / (Qd + Pd) if (Qd + Pd) > 1e-15 else np.nan
    print(f"  {layer:6d} | {dQr:14.6f} | {dPr:14.6f} | {Qd:14.6f} | {Pd:14.6f} | {frac:10.4f}")


# =============================================================================
# TABLE 5: EARLY vs LATE TRAINING COMPARISON
# =============================================================================

print(f"\n\n{'=' * 100}")
print("TABLE 5: EARLY (steps 1-50) vs LATE (steps 250-300) TRAINING DYNAMICS")
print("=" * 100)

early_slice = slice(0, 50)
late_slice = slice(250, 300)

print(f"\n{'Phase':>10} | {'SGD dQ/dW':>12} | {'SGD dP/dW':>12} | "
      f"{'Muon dQ/dW':>12} | {'Muon dP/dW':>12}")
print("-" * 68)

for phase_name, sl in [('Early', early_slice), ('Late', late_slice)]:
    sgd_dQr = np.nanmean(results_sgd['dQ_ratio'][:, sl])
    sgd_dPr = np.nanmean(results_sgd['dP_ratio'][:, sl])
    muon_dQr = np.nanmean(results_muon['dQ_ratio'][:, sl])
    muon_dPr = np.nanmean(results_muon['dP_ratio'][:, sl])
    print(f"{phase_name:>10} | {sgd_dQr:12.6f} | {sgd_dPr:12.6f} | "
          f"{muon_dQr:12.6f} | {muon_dPr:12.6f}")


# =============================================================================
# PLOT: CUMULATIVE Q AND P DRIFT OVER TIME
# =============================================================================

print(f"\n\n{'=' * 100}")
print("GENERATING PLOTS")
print("=" * 100)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('1.2b-ii: Trajectory in Polar Coordinates -- (Q, P) Decomposition\n'
                 f'{NUM_LAYERS}-layer linear net, dim={DIM}, {NUM_STEPS} steps',
                 fontsize=14, fontweight='bold')

    t_axis = np.arange(NUM_STEPS)

    # --- Panel (a): Cumulative Q drift ---
    ax = axes[0, 0]
    ax.set_title('(a) Cumulative Orientation Drift ||Q_t - Q_0||')
    for layer in range(NUM_LAYERS):
        ax.plot(t_axis, results_sgd['cum_Q_drift'][layer, :],
                'b-', alpha=0.3, linewidth=0.8)
        ax.plot(t_axis, results_muon['cum_Q_drift'][layer, :],
                'r-', alpha=0.3, linewidth=0.8)
    # Layer-averaged
    ax.plot(t_axis, np.mean(results_sgd['cum_Q_drift'], axis=0),
            'b-', linewidth=2.5, label='SGD (avg)')
    ax.plot(t_axis, np.mean(results_muon['cum_Q_drift'], axis=0),
            'r-', linewidth=2.5, label='Muon (avg)')
    ax.set_xlabel('Step')
    ax.set_ylabel('||Q_t - Q_0||_F')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel (b): Cumulative P drift ---
    ax = axes[0, 1]
    ax.set_title('(b) Cumulative Spectrum Drift ||P_t - P_0||')
    for layer in range(NUM_LAYERS):
        ax.plot(t_axis, results_sgd['cum_P_drift'][layer, :],
                'b-', alpha=0.3, linewidth=0.8)
        ax.plot(t_axis, results_muon['cum_P_drift'][layer, :],
                'r-', alpha=0.3, linewidth=0.8)
    ax.plot(t_axis, np.mean(results_sgd['cum_P_drift'], axis=0),
            'b-', linewidth=2.5, label='SGD (avg)')
    ax.plot(t_axis, np.mean(results_muon['cum_P_drift'], axis=0),
            'r-', linewidth=2.5, label='Muon (avg)')
    ax.set_xlabel('Step')
    ax.set_ylabel('||P_t - P_0||_F')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Q drift / (Q drift + P drift) over time ---
    ax = axes[0, 2]
    ax.set_title('(c) Orientation Fraction: ||Q-Q0|| / (||Q-Q0|| + ||P-P0||)')
    sgd_Q_avg = np.mean(results_sgd['cum_Q_drift'], axis=0)
    sgd_P_avg = np.mean(results_sgd['cum_P_drift'], axis=0)
    muon_Q_avg = np.mean(results_muon['cum_Q_drift'], axis=0)
    muon_P_avg = np.mean(results_muon['cum_P_drift'], axis=0)

    sgd_frac = sgd_Q_avg / (sgd_Q_avg + sgd_P_avg + 1e-15)
    muon_frac = muon_Q_avg / (muon_Q_avg + muon_P_avg + 1e-15)

    ax.plot(t_axis, sgd_frac, 'b-', linewidth=2.5, label='SGD')
    ax.plot(t_axis, muon_frac, 'r-', linewidth=2.5, label='Muon')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Equal Q/P')
    ax.set_xlabel('Step')
    ax.set_ylabel('Q fraction')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel (d): Per-step ||dQ||/||dW|| ---
    ax = axes[1, 0]
    ax.set_title('(d) Per-Step ||dQ|| / ||dW|| (layer-averaged)')

    # Smooth with rolling window
    window = 10
    sgd_dQr_avg = np.nanmean(results_sgd['dQ_ratio'], axis=0)
    muon_dQr_avg = np.nanmean(results_muon['dQ_ratio'], axis=0)

    # Rolling mean
    sgd_dQr_smooth = np.convolve(sgd_dQr_avg, np.ones(window)/window, mode='valid')
    muon_dQr_smooth = np.convolve(muon_dQr_avg, np.ones(window)/window, mode='valid')

    ax.plot(np.arange(len(sgd_dQr_smooth)), sgd_dQr_smooth,
            'b-', linewidth=2, label='SGD')
    ax.plot(np.arange(len(muon_dQr_smooth)), muon_dQr_smooth,
            'r-', linewidth=2, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('||dQ|| / ||dW||')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel (e): Per-step ||dP||/||dW|| ---
    ax = axes[1, 1]
    ax.set_title('(e) Per-Step ||dP|| / ||dW|| (layer-averaged)')

    sgd_dPr_avg = np.nanmean(results_sgd['dP_ratio'], axis=0)
    muon_dPr_avg = np.nanmean(results_muon['dP_ratio'], axis=0)

    sgd_dPr_smooth = np.convolve(sgd_dPr_avg, np.ones(window)/window, mode='valid')
    muon_dPr_smooth = np.convolve(muon_dPr_avg, np.ones(window)/window, mode='valid')

    ax.plot(np.arange(len(sgd_dPr_smooth)), sgd_dPr_smooth,
            'b-', linewidth=2, label='SGD')
    ax.plot(np.arange(len(muon_dPr_smooth)), muon_dPr_smooth,
            'r-', linewidth=2, label='Muon')
    ax.set_xlabel('Step')
    ax.set_ylabel('||dP|| / ||dW||')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel (f): Loss curves ---
    ax = axes[1, 2]
    ax.set_title('(f) Training Loss')
    ax.semilogy(np.arange(NUM_STEPS + 1), results_sgd['losses'], 'b-',
                linewidth=2, label=f'SGD (lr={lr_sgd})')
    ax.semilogy(np.arange(NUM_STEPS + 1), results_muon['losses'], 'r-',
                linewidth=2, label=f'Muon (lr={LR_MUON})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'polar_trajectory_decomposition.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to: {plot_path}")

except ImportError:
    print("\n  WARNING: matplotlib not available, skipping plots.")
    plot_path = None


# =============================================================================
# ADDITIONAL PLOT: PHASE PORTRAIT (Q drift vs P drift)
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('1.2b-ii: Phase Portrait -- Cumulative Q Drift vs P Drift',
                 fontsize=13, fontweight='bold')

    # Panel (a): All layers
    ax = axes[0]
    ax.set_title('(a) Per-Layer Phase Portrait')
    for layer in range(NUM_LAYERS):
        ax.plot(results_sgd['cum_P_drift'][layer, :],
                results_sgd['cum_Q_drift'][layer, :],
                'b-', alpha=0.5, linewidth=1, label=f'SGD L{layer}' if layer == 0 else None)
        ax.plot(results_muon['cum_P_drift'][layer, :],
                results_muon['cum_Q_drift'][layer, :],
                'r-', alpha=0.5, linewidth=1, label=f'Muon L{layer}' if layer == 0 else None)
        # Mark endpoints
        ax.plot(results_sgd['cum_P_drift'][layer, -1],
                results_sgd['cum_Q_drift'][layer, -1],
                'bx', markersize=8)
        ax.plot(results_muon['cum_P_drift'][layer, -1],
                results_muon['cum_Q_drift'][layer, -1],
                'rx', markersize=8)

    # Diagonal reference line
    max_val = max(
        np.nanmax(results_sgd['cum_Q_drift']),
        np.nanmax(results_sgd['cum_P_drift']),
        np.nanmax(results_muon['cum_Q_drift']),
        np.nanmax(results_muon['cum_P_drift']),
    )
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Q=P (equal)')
    ax.set_xlabel('Cumulative ||P_t - P_0||_F (spectrum drift)')
    ax.set_ylabel('Cumulative ||Q_t - Q_0||_F (orientation drift)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b): Averaged
    ax = axes[1]
    ax.set_title('(b) Layer-Averaged Phase Portrait')
    sgd_Q_avg = np.mean(results_sgd['cum_Q_drift'], axis=0)
    sgd_P_avg = np.mean(results_sgd['cum_P_drift'], axis=0)
    muon_Q_avg = np.mean(results_muon['cum_Q_drift'], axis=0)
    muon_P_avg = np.mean(results_muon['cum_P_drift'], axis=0)

    ax.plot(sgd_P_avg, sgd_Q_avg, 'b-', linewidth=2.5, label='SGD')
    ax.plot(muon_P_avg, muon_Q_avg, 'r-', linewidth=2.5, label='Muon')
    ax.plot(sgd_P_avg[-1], sgd_Q_avg[-1], 'bx', markersize=12, markeredgewidth=3)
    ax.plot(muon_P_avg[-1], muon_Q_avg[-1], 'rx', markersize=12, markeredgewidth=3)
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Q=P (equal)')
    ax.set_xlabel('Cumulative ||P_t - P_0||_F (spectrum drift)')
    ax.set_ylabel('Cumulative ||Q_t - Q_0||_F (orientation drift)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = os.path.join(SCRIPT_DIR, 'polar_phase_portrait.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to: {plot_path2}")

except ImportError:
    pass


# =============================================================================
# VERDICT
# =============================================================================

print(f"\n\n{'=' * 100}")
print("FINAL ANALYSIS: POLAR COORDINATE TRAJECTORY DECOMPOSITION")
print("=" * 100)

# Compute key summary statistics
sgd_dQr_all = np.nanmean(results_sgd['dQ_ratio'])
sgd_dPr_all = np.nanmean(results_sgd['dP_ratio'])
muon_dQr_all = np.nanmean(results_muon['dQ_ratio'])
muon_dPr_all = np.nanmean(results_muon['dP_ratio'])

sgd_cumQ_final = np.nanmean(results_sgd['cum_Q_drift'][:, -1])
sgd_cumP_final = np.nanmean(results_sgd['cum_P_drift'][:, -1])
muon_cumQ_final = np.nanmean(results_muon['cum_Q_drift'][:, -1])
muon_cumP_final = np.nanmean(results_muon['cum_P_drift'][:, -1])

sgd_Q_frac_final = sgd_cumQ_final / (sgd_cumQ_final + sgd_cumP_final + 1e-15)
muon_Q_frac_final = muon_cumQ_final / (muon_cumQ_final + muon_cumP_final + 1e-15)

print(f"""
  SUMMARY STATISTICS (averaged across layers, over all {NUM_STEPS} steps):
  ---------------------------------------------------------------
  PER-STEP RATIOS:
    SGD:   ||dQ||/||dW|| = {sgd_dQr_all:.6f},  ||dP||/||dW|| = {sgd_dPr_all:.6f}
    Muon:  ||dQ||/||dW|| = {muon_dQr_all:.6f},  ||dP||/||dW|| = {muon_dPr_all:.6f}

  CUMULATIVE DRIFT AT STEP {NUM_STEPS}:
    SGD:   ||Q-Q0|| = {sgd_cumQ_final:.6f},  ||P-P0|| = {sgd_cumP_final:.6f},  Q/(Q+P) = {sgd_Q_frac_final:.4f}
    Muon:  ||Q-Q0|| = {muon_cumQ_final:.6f},  ||P-P0|| = {muon_cumP_final:.6f},  Q/(Q+P) = {muon_Q_frac_final:.4f}

  FINAL LOSSES:
    SGD:   {results_sgd['losses'][-1]:.6e}
    Muon:  {results_muon['losses'][-1]:.6e}
  ---------------------------------------------------------------
""")

# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

print("  HYPOTHESIS TESTS:")
print("  ---------------------------------------------------------------")

# Test 1: Muon's per-step dQ/dW > SGD's per-step dQ/dW
# (Muon changes orientation more per step)
test1 = muon_dQr_all > sgd_dQr_all
print(f"  T1: Muon ||dQ||/||dW|| > SGD ||dQ||/||dW||")
print(f"      Muon={muon_dQr_all:.6f} vs SGD={sgd_dQr_all:.6f}")
print(f"      -> {'YES' if test1 else 'NO'}")

# Test 2: Muon's per-step dP/dW < SGD's per-step dP/dW
# (Muon changes spectrum less per step)
test2 = muon_dPr_all < sgd_dPr_all
print(f"\n  T2: Muon ||dP||/||dW|| < SGD ||dP||/||dW||")
print(f"      Muon={muon_dPr_all:.6f} vs SGD={sgd_dPr_all:.6f}")
print(f"      -> {'YES' if test2 else 'NO'}")

# Test 3: Muon's cumulative Q fraction > SGD's cumulative Q fraction
# (Muon's total movement is more orientation-dominated)
test3 = muon_Q_frac_final > sgd_Q_frac_final
print(f"\n  T3: Muon Q/(Q+P) > SGD Q/(Q+P) at step {NUM_STEPS}")
print(f"      Muon={muon_Q_frac_final:.4f} vs SGD={sgd_Q_frac_final:.4f}")
print(f"      -> {'YES' if test3 else 'NO'}")

# Test 4: Muon has higher Q/P ratio overall
muon_QP_ratio = muon_cumQ_final / (muon_cumP_final + 1e-15)
sgd_QP_ratio = sgd_cumQ_final / (sgd_cumP_final + 1e-15)
test4 = muon_QP_ratio > sgd_QP_ratio
print(f"\n  T4: Muon cumulative Q/P ratio > SGD cumulative Q/P ratio")
print(f"      Muon={muon_QP_ratio:.4f} vs SGD={sgd_QP_ratio:.4f}")
print(f"      -> {'YES' if test4 else 'NO'}")

# Test 5: Muon's cumulative P drift is smaller than SGD's
# (Muon moves less in spectrum space)
test5 = muon_cumP_final < sgd_cumP_final
print(f"\n  T5: Muon ||P-P0|| < SGD ||P-P0|| at step {NUM_STEPS}")
print(f"      Muon={muon_cumP_final:.6f} vs SGD={sgd_cumP_final:.6f}")
print(f"      -> {'YES' if test5 else 'NO'}")

tests_passed = sum([test1, test2, test3, test4, test5])

print(f"\n  ---------------------------------------------------------------")
print(f"  Tests passed: {tests_passed}/5")

# Determine overall verdict
if tests_passed >= 4:
    overall = "STRONG SUPPORT"
    detail = (
        "The data strongly supports the hypothesis that Muon's updates are\n"
        "  predominantly orientation (Q) changes while SGD changes both Q and P.\n"
        "  Muon moves on the Stiefel manifold; SGD wanders in full weight space."
    )
elif tests_passed >= 3:
    overall = "MODERATE SUPPORT"
    detail = (
        "The data moderately supports the polar decomposition hypothesis.\n"
        "  There is a measurable difference in how Muon vs SGD distribute\n"
        "  their updates between orientation (Q) and spectrum (P)."
    )
elif tests_passed >= 2:
    overall = "WEAK SIGNAL"
    detail = (
        "The data shows a weak signal. The difference between Muon and SGD\n"
        "  in polar coordinates is present but not dramatic.\n"
        "  Consistent with 1.2b-i: the distinction may be subtler than expected."
    )
else:
    overall = "NO SUPPORT / SURPRISING"
    detail = (
        "The hypothesis is not supported. Muon does NOT preferentially change\n"
        "  orientation over spectrum, or does so LESS than SGD.\n"
        "  This is a genuine surprise that requires reinterpretation."
    )

# Check for the SURPRISE case: if both optimizers have very similar profiles
if abs(muon_Q_frac_final - sgd_Q_frac_final) < 0.05:
    overall += " (SIMILAR PROFILES)"
    detail += (
        "\n\n  NOTE: Both optimizers have very similar Q/(Q+P) fractions.\n"
        "  The polar decomposition may not be the right lens to distinguish them.\n"
        "  This is consistent with 1.2b-i's finding that the Lyapunov exponents\n"
        "  were similar for gauge and physical directions."
    )

print(f"""
  ======================================================================
  VERDICT: {overall}
  ======================================================================
  {detail}
  ======================================================================
""")

print("=" * 100)
print(f"  Tests passed: {tests_passed}/5")
print(f"  Overall: {overall}")
print("=" * 100)
