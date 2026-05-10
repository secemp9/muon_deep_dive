#!/usr/bin/env python3
"""
Experiment 3.16: Eigenvalue Lifting -- per-step vs cumulative
==============================================================

CONTEXT:
  2.10 showed 330x Hessian condition number (kappa) reduction at convergence
  comparing Muon to SGD. But is that a per-step effect or purely cumulative?

QUESTION:
  Does a SINGLE Muon step lift lambda_min of the Hessian by more than a
  single SGD step, from the SAME starting point?

SETUP:
  - 2-layer 4x4 deep linear net (32 params, full Hessian computable)
  - At a training point (reached by 'step' SGD warmup steps):
    1. Compute full Hessian H_before
    2. Take ONE step with SGD from that point -> compute H_after_SGD
    3. Take ONE step with Muon from the SAME point -> compute H_after_Muon
    4. Measure: lambda_min(H_after) / lambda_min(H_before) for both
  - Repeat at 10 different training points spread over 1500 steps
  - Key test: does Muon lift lambda_min by >3x what SGD does per step?

METRICS:
  - lambda_min lifting ratio: lambda_min(H_after) / lambda_min(H_before)
  - kappa change: kappa(H_after) / kappa(H_before)
  - trace change: tr(H_after) / tr(H_before)
  - Hessian eigenvalue spectrum shift
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 4
NUM_LAYERS = 2
N_PARAMS = NUM_LAYERS * DIM * DIM  # 32
HESSIAN_EPS = 1e-5
DATA_POINTS = 32
MOMENTUM = 0.9
LR_SGD = 0.01
LR_MUON = 0.02
NS_ITERS = 5
WARMUP_MAX_STEPS = 1500
NUM_SEEDS = 5

# Measurement points: step indices during warmup at which to snapshot
MEASUREMENT_STEPS = [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1300]


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
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y_target):
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(diff ** 2)


def compute_gradients(weights, X, Y_target):
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    delta = (activations[-1] - Y_target) / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


# =============================================================================
# NEWTON-SCHULZ ITERATION
# =============================================================================

def newton_schulz_orthogonalize(G, num_iters=5):
    """Quintic Newton-Schulz: a=3.4445, b=-4.7750, c=2.0315."""
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        XtX = X.T @ X
        X_XtX = X @ XtX
        XtX2 = XtX @ XtX
        X_XtX2 = X @ XtX2
        X = a * X + b * X_XtX + c * X_XtX2

    return X


# =============================================================================
# FULL HESSIAN COMPUTATION
# =============================================================================

def weights_to_vector(weights):
    return np.concatenate([W.flatten() for W in weights])


def vector_to_weights(vec, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        W = vec[idx:idx + size].reshape(shape)
        weights.append(W)
        idx += size
    return weights


def compute_gradient_vector(weights, X, Y_target):
    grads = compute_gradients(weights, X, Y_target)
    return np.concatenate([g.flatten() for g in grads])


def compute_full_hessian(weights, X, Y_target, eps=HESSIAN_EPS):
    """Full Hessian via central finite differences on the gradient."""
    shapes = [W.shape for W in weights]
    theta = weights_to_vector(weights)
    n = len(theta)

    H = np.zeros((n, n))
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += eps
        theta_m[i] -= eps

        g_p = compute_gradient_vector(vector_to_weights(theta_p, shapes), X, Y_target)
        g_m = compute_gradient_vector(vector_to_weights(theta_m, shapes), X, Y_target)

        H[:, i] = (g_p - g_m) / (2 * eps)

    H = 0.5 * (H + H.T)
    return H


def analyze_hessian(H):
    """Compute key Hessian statistics."""
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    lambda_max = eigenvalues_sorted[0]
    lambda_min_pos = None
    trace_H = np.sum(eigenvalues)

    # lambda_min among positive eigenvalues
    pos_eigs = eigenvalues[eigenvalues > 1e-12]
    if len(pos_eigs) > 0:
        lambda_min_pos = np.min(pos_eigs)

    # Condition number
    if lambda_min_pos is not None and lambda_min_pos > 1e-15:
        kappa = lambda_max / lambda_min_pos
    else:
        kappa = np.inf

    return {
        'eigenvalues': eigenvalues_sorted,
        'lambda_max': lambda_max,
        'lambda_min_pos': lambda_min_pos,
        'trace': trace_H,
        'kappa': kappa,
    }


# =============================================================================
# SINGLE-STEP COMPARISON
# =============================================================================

def one_step_sgd(weights, X, Y_target, lr, momentum_state=None):
    """Take one SGD+momentum step. Returns new weights."""
    grads = compute_gradients(weights, X, Y_target)
    new_weights = []
    for i in range(len(weights)):
        if momentum_state is not None:
            vel = MOMENTUM * momentum_state[i] + grads[i]
        else:
            vel = grads[i]
        W_new = weights[i] - lr * vel
        new_weights.append(W_new)
    return new_weights


def one_step_muon(weights, X, Y_target, lr, momentum_state=None):
    """Take one Muon step. Returns new weights."""
    grads = compute_gradients(weights, X, Y_target)
    new_weights = []
    for i in range(len(weights)):
        G_orth = newton_schulz_orthogonalize(grads[i], num_iters=NS_ITERS)
        if momentum_state is not None:
            vel = MOMENTUM * momentum_state[i] + G_orth
        else:
            vel = G_orth
        W_new = weights[i] - lr * vel
        new_weights.append(W_new)
    return new_weights


# =============================================================================
# WARMUP TRAINING (SGD to reach measurement points)
# =============================================================================

def warmup_sgd(weights, X, Y_target, max_steps, measurement_steps):
    """
    Train with SGD, collecting weight snapshots and momentum states
    at specified measurement steps.
    """
    velocities = [np.zeros_like(W) for W in weights]
    snapshots = {}

    for step in range(max_steps):
        if step in measurement_steps:
            # Save deep copies of weights and momentum
            snapshots[step] = {
                'weights': [W.copy() for W in weights],
                'velocities': [v.copy() for v in velocities],
                'loss': compute_loss(weights, X, Y_target),
            }

        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - LR_SGD * velocities[i]

    return snapshots


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_single_seed(seed):
    """Run the full per-step eigenvalue lifting test for one seed."""
    rng = np.random.RandomState(seed)

    # Generate data
    W_target = [rng.randn(DIM, DIM) * 0.3 for _ in range(NUM_LAYERS)]
    X = rng.randn(DIM, DATA_POINTS) * 0.5
    Y_target = X.copy()
    for W in W_target:
        Y_target = W @ Y_target

    # Initialize and warmup
    weights_init = init_weights(DIM, NUM_LAYERS, seed + 1000)
    snapshots = warmup_sgd(
        [W.copy() for W in weights_init], X, Y_target,
        max_steps=WARMUP_MAX_STEPS,
        measurement_steps=MEASUREMENT_STEPS
    )

    results = []
    for mstep in MEASUREMENT_STEPS:
        if mstep not in snapshots:
            continue

        snap = snapshots[mstep]
        w_snap = snap['weights']
        v_snap = snap['velocities']
        loss_at = snap['loss']

        # Hessian BEFORE the step
        H_before = compute_full_hessian(w_snap, X, Y_target)
        stats_before = analyze_hessian(H_before)

        # One SGD step from this point
        w_after_sgd = one_step_sgd(w_snap, X, Y_target, LR_SGD, v_snap)
        H_after_sgd = compute_full_hessian(w_after_sgd, X, Y_target)
        stats_after_sgd = analyze_hessian(H_after_sgd)

        # One Muon step from the SAME point (same momentum state)
        w_after_muon = one_step_muon(w_snap, X, Y_target, LR_MUON, v_snap)
        H_after_muon = compute_full_hessian(w_after_muon, X, Y_target)
        stats_after_muon = analyze_hessian(H_after_muon)

        # Compute lifting ratios
        lmin_before = stats_before['lambda_min_pos']
        lmin_sgd = stats_after_sgd['lambda_min_pos']
        lmin_muon = stats_after_muon['lambda_min_pos']

        if lmin_before is not None and lmin_before > 1e-15:
            lift_sgd = lmin_sgd / lmin_before if lmin_sgd is not None else 0.0
            lift_muon = lmin_muon / lmin_before if lmin_muon is not None else 0.0
        else:
            lift_sgd = float('nan')
            lift_muon = float('nan')

        # Kappa change
        kappa_before = stats_before['kappa']
        kappa_sgd = stats_after_sgd['kappa']
        kappa_muon = stats_after_muon['kappa']

        if kappa_before > 0 and not np.isinf(kappa_before):
            kappa_ratio_sgd = kappa_sgd / kappa_before
            kappa_ratio_muon = kappa_muon / kappa_before
        else:
            kappa_ratio_sgd = float('nan')
            kappa_ratio_muon = float('nan')

        # Trace change
        tr_before = stats_before['trace']
        tr_sgd = stats_after_sgd['trace']
        tr_muon = stats_after_muon['trace']

        if abs(tr_before) > 1e-15:
            tr_ratio_sgd = tr_sgd / tr_before
            tr_ratio_muon = tr_muon / tr_before
        else:
            tr_ratio_sgd = float('nan')
            tr_ratio_muon = float('nan')

        results.append({
            'step': mstep,
            'loss': loss_at,
            'lmin_before': lmin_before,
            'lmin_sgd': lmin_sgd,
            'lmin_muon': lmin_muon,
            'lift_sgd': lift_sgd,
            'lift_muon': lift_muon,
            'kappa_before': kappa_before,
            'kappa_ratio_sgd': kappa_ratio_sgd,
            'kappa_ratio_muon': kappa_ratio_muon,
            'tr_ratio_sgd': tr_ratio_sgd,
            'tr_ratio_muon': tr_ratio_muon,
            'lmax_before': stats_before['lambda_max'],
            'lmax_sgd': stats_after_sgd['lambda_max'],
            'lmax_muon': stats_after_muon['lambda_max'],
        })

    return results


def main():
    print()
    print("=" * 110)
    print("  Experiment 3.16: Eigenvalue Lifting -- Per-Step vs Cumulative")
    print("=" * 110)
    print()
    print("  QUESTION: Does a single Muon step lift lambda_min more than a single SGD step?")
    print()
    print(f"  Config: {NUM_LAYERS}-layer {DIM}x{DIM} deep linear net ({N_PARAMS} params)")
    print(f"  LR_SGD={LR_SGD}, LR_MUON={LR_MUON}, momentum={MOMENTUM}, NS_iters={NS_ITERS}")
    print(f"  Measurement points: {MEASUREMENT_STEPS}")
    print(f"  Averaging over {NUM_SEEDS} seeds")
    print()

    # =========================================================================
    # Run all seeds
    # =========================================================================
    all_results = []  # list of lists (per seed)

    for s in range(NUM_SEEDS):
        seed = 42 + s * 137
        print(f"  Running seed {s+1}/{NUM_SEEDS} (seed={seed})...", flush=True)
        seed_results = run_single_seed(seed)
        all_results.append(seed_results)

    # =========================================================================
    # Per-measurement-point aggregation
    # =========================================================================
    print()
    print("=" * 110)
    print("  PER-STEP EIGENVALUE LIFTING RESULTS (averaged over seeds)")
    print("=" * 110)
    print()

    print(f"  {'Step':>6} {'Loss':>10} {'lmin_bef':>10} "
          f"{'lift_SGD':>10} {'lift_Muon':>10} {'Muon/SGD':>10} "
          f"{'kR_SGD':>10} {'kR_Muon':>10} "
          f"{'trR_SGD':>10} {'trR_Muon':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*10}")

    # Organize results by measurement step
    step_to_data = {}
    for mstep in MEASUREMENT_STEPS:
        step_to_data[mstep] = {
            'losses': [], 'lmin_before': [],
            'lift_sgd': [], 'lift_muon': [],
            'kappa_ratio_sgd': [], 'kappa_ratio_muon': [],
            'tr_ratio_sgd': [], 'tr_ratio_muon': [],
        }

    for seed_results in all_results:
        for r in seed_results:
            mstep = r['step']
            if mstep in step_to_data:
                d = step_to_data[mstep]
                d['losses'].append(r['loss'])
                d['lmin_before'].append(r['lmin_before'] if r['lmin_before'] is not None else 0.0)
                if not np.isnan(r['lift_sgd']):
                    d['lift_sgd'].append(r['lift_sgd'])
                if not np.isnan(r['lift_muon']):
                    d['lift_muon'].append(r['lift_muon'])
                if not np.isnan(r['kappa_ratio_sgd']):
                    d['kappa_ratio_sgd'].append(r['kappa_ratio_sgd'])
                if not np.isnan(r['kappa_ratio_muon']):
                    d['kappa_ratio_muon'].append(r['kappa_ratio_muon'])
                if not np.isnan(r['tr_ratio_sgd']):
                    d['tr_ratio_sgd'].append(r['tr_ratio_sgd'])
                if not np.isnan(r['tr_ratio_muon']):
                    d['tr_ratio_muon'].append(r['tr_ratio_muon'])

    muon_vs_sgd_ratios = []  # lift_muon / lift_sgd per step

    for mstep in MEASUREMENT_STEPS:
        d = step_to_data[mstep]
        if len(d['lift_sgd']) == 0 or len(d['lift_muon']) == 0:
            continue

        loss_mean = np.mean(d['losses'])
        lmin_mean = np.mean(d['lmin_before'])
        lift_sgd_mean = np.mean(d['lift_sgd'])
        lift_muon_mean = np.mean(d['lift_muon'])

        if lift_sgd_mean > 1e-15:
            ratio = lift_muon_mean / lift_sgd_mean
        else:
            ratio = float('nan')
        muon_vs_sgd_ratios.append(ratio)

        kr_sgd = np.mean(d['kappa_ratio_sgd']) if d['kappa_ratio_sgd'] else float('nan')
        kr_muon = np.mean(d['kappa_ratio_muon']) if d['kappa_ratio_muon'] else float('nan')
        tr_sgd = np.mean(d['tr_ratio_sgd']) if d['tr_ratio_sgd'] else float('nan')
        tr_muon = np.mean(d['tr_ratio_muon']) if d['tr_ratio_muon'] else float('nan')

        print(f"  {mstep:>6} {loss_mean:10.4e} {lmin_mean:10.4e} "
              f"{lift_sgd_mean:10.4f} {lift_muon_mean:10.4f} {ratio:10.4f} "
              f"{kr_sgd:10.4f} {kr_muon:10.4f} "
              f"{tr_sgd:10.4f} {tr_muon:10.4f}")

    print()
    print("  Legend:")
    print("    lift_SGD/Muon = lambda_min(H_after) / lambda_min(H_before)  [>1 = lifting]")
    print("    Muon/SGD      = lift_Muon / lift_SGD  [>1 = Muon lifts more]")
    print("    kR             = kappa(H_after) / kappa(H_before)  [<1 = conditioning improves]")
    print("    trR            = tr(H_after) / tr(H_before)")

    # =========================================================================
    # Detailed per-seed table at a representative step
    # =========================================================================
    rep_step = 100  # representative step

    print()
    print("=" * 110)
    print(f"  DETAILED PER-SEED RESULTS AT STEP {rep_step}")
    print("=" * 110)
    print()

    print(f"  {'Seed':>6} {'Loss':>10} {'lmin_bef':>10} "
          f"{'lmin_SGD':>10} {'lmin_Muon':>10} "
          f"{'lift_SGD':>10} {'lift_Muon':>10} {'Muon>SGD?':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for s, seed_results in enumerate(all_results):
        for r in seed_results:
            if r['step'] == rep_step:
                muon_wins = "YES" if r['lift_muon'] > r['lift_sgd'] else "NO"
                print(f"  {s+1:>6} {r['loss']:10.4e} {r['lmin_before']:10.4e} "
                      f"{r['lmin_sgd']:10.4e} {r['lmin_muon']:10.4e} "
                      f"{r['lift_sgd']:10.4f} {r['lift_muon']:10.4f} {muon_wins:>10}")

    # =========================================================================
    # Lambda_max analysis (does Muon also reduce sharpness per step?)
    # =========================================================================
    print()
    print("=" * 110)
    print("  LAMBDA_MAX (SHARPNESS) PER-STEP CHANGE")
    print("=" * 110)
    print()

    print(f"  {'Step':>6} {'lmax_bef':>12} {'lmax_SGD':>12} {'lmax_Muon':>12} "
          f"{'SGD ratio':>12} {'Muon ratio':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for mstep in MEASUREMENT_STEPS:
        lmax_bef_list = []
        lmax_sgd_list = []
        lmax_muon_list = []
        for seed_results in all_results:
            for r in seed_results:
                if r['step'] == mstep:
                    lmax_bef_list.append(r['lmax_before'])
                    lmax_sgd_list.append(r['lmax_sgd'])
                    lmax_muon_list.append(r['lmax_muon'])

        if not lmax_bef_list:
            continue

        lmax_bef_mean = np.mean(lmax_bef_list)
        lmax_sgd_mean = np.mean(lmax_sgd_list)
        lmax_muon_mean = np.mean(lmax_muon_list)

        r_sgd = lmax_sgd_mean / lmax_bef_mean if lmax_bef_mean > 1e-15 else float('nan')
        r_muon = lmax_muon_mean / lmax_bef_mean if lmax_bef_mean > 1e-15 else float('nan')

        print(f"  {mstep:>6} {lmax_bef_mean:12.4e} {lmax_sgd_mean:12.4e} {lmax_muon_mean:12.4e} "
              f"{r_sgd:12.4f} {r_muon:12.4f}")

    # =========================================================================
    # HYPOTHESIS TESTS
    # =========================================================================
    print()
    print("=" * 110)
    print("  HYPOTHESIS TESTS")
    print("=" * 110)
    print()

    # Test 1: Muon lifts lambda_min more than SGD at majority of measurement points
    valid_ratios = [r for r in muon_vs_sgd_ratios if not np.isnan(r)]
    muon_wins_count = sum(1 for r in valid_ratios if r > 1.0)
    total_points = len(valid_ratios)

    t1_pass = muon_wins_count > total_points * 0.5
    print(f"  T1: Muon lifts lambda_min more than SGD at majority of points?")
    print(f"      Points where Muon/SGD > 1: {muon_wins_count}/{total_points}")
    print(f"      {'PASS' if t1_pass else 'FAIL'}")
    print()

    # Test 2: KEY TEST -- Muon lifts lambda_min by >3x what SGD does (on average)
    if valid_ratios:
        mean_ratio = np.mean(valid_ratios)
        median_ratio = np.median(valid_ratios)
    else:
        mean_ratio = float('nan')
        median_ratio = float('nan')

    t2_pass = mean_ratio > 3.0
    print(f"  T2: KEY TEST -- Muon lifts lambda_min by >3x vs SGD on average?")
    print(f"      Mean lift_Muon/lift_SGD = {mean_ratio:.4f}")
    print(f"      Median = {median_ratio:.4f}")
    print(f"      {'PASS' if t2_pass else 'FAIL'}: {'>' if t2_pass else '<='} 3x threshold")
    print()

    # Test 3: Muon reduces kappa per step more than SGD
    kappa_muon_better = 0
    kappa_total = 0
    for mstep in MEASUREMENT_STEPS:
        d = step_to_data[mstep]
        if d['kappa_ratio_sgd'] and d['kappa_ratio_muon']:
            kr_sgd = np.mean(d['kappa_ratio_sgd'])
            kr_muon = np.mean(d['kappa_ratio_muon'])
            if kr_muon < kr_sgd:
                kappa_muon_better += 1
            kappa_total += 1

    t3_pass = kappa_muon_better > kappa_total * 0.5
    print(f"  T3: Muon reduces kappa per step more than SGD at majority of points?")
    print(f"      Points where Muon kappa ratio < SGD kappa ratio: {kappa_muon_better}/{kappa_total}")
    print(f"      {'PASS' if t3_pass else 'FAIL'}")
    print()

    # Test 4: Effect is consistent across seeds (low variance)
    if valid_ratios:
        cv = np.std(valid_ratios) / np.mean(valid_ratios) if np.mean(valid_ratios) > 0 else float('inf')
        t4_pass = cv < 1.0  # coefficient of variation < 1 means reasonably consistent
    else:
        cv = float('nan')
        t4_pass = False
    print(f"  T4: Effect is consistent (CV of Muon/SGD ratio < 1.0)?")
    print(f"      Coefficient of variation = {cv:.4f}")
    print(f"      {'PASS' if t4_pass else 'FAIL'}")
    print()

    # Test 5: The per-step effect persists late in training (not just early)
    late_ratios = []
    for mstep in MEASUREMENT_STEPS:
        if mstep >= 600:
            d = step_to_data[mstep]
            if d['lift_sgd'] and d['lift_muon']:
                ls = np.mean(d['lift_sgd'])
                lm = np.mean(d['lift_muon'])
                if ls > 1e-15:
                    late_ratios.append(lm / ls)

    if late_ratios:
        mean_late = np.mean(late_ratios)
        t5_pass = mean_late > 1.0
    else:
        mean_late = float('nan')
        t5_pass = False
    print(f"  T5: Per-step lifting persists late in training (steps >= 600)?")
    print(f"      Mean Muon/SGD ratio for late steps: {mean_late:.4f}")
    print(f"      {'PASS' if t5_pass else 'FAIL'}")
    print()

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    print("=" * 110)
    print("  OVERALL VERDICT")
    print("=" * 110)
    print()

    if t2_pass:
        print("  [CONFIRMED] Muon lifts lambda_min by >3x per step compared to SGD.")
        print(f"  Mean lifting ratio = {mean_ratio:.2f}x.")
        print("  The 330x kappa reduction at convergence is (at least partly) a PER-STEP effect.")
    elif t1_pass:
        print(f"  [PARTIAL] Muon lifts lambda_min more than SGD ({mean_ratio:.2f}x) but not >3x.")
        print("  The kappa reduction may be partly per-step, partly cumulative.")
    else:
        print(f"  [REJECTED] Muon does NOT lift lambda_min more than SGD per step ({mean_ratio:.2f}x).")
        print("  The 330x kappa reduction is a CUMULATIVE effect, not per-step.")

    if t3_pass:
        print("  [BONUS] Muon also reduces condition number more per step.")

    if t5_pass:
        print("  [BONUS] The per-step effect persists late in training.")
    elif t1_pass:
        print("  [NOTE] The per-step effect may weaken late in training.")

    print()
    print("=" * 110)
    print("  EXPERIMENT COMPLETE")
    print("=" * 110)


if __name__ == '__main__':
    main()
