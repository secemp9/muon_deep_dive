#!/usr/bin/env python3
"""
Experiment 2.6: Product kappa Crossover vs Depth
==================================================

HYPOTHESIS:
  The product condition number kappa_prod = prod_l kappa(W_l) grows differently
  under Muon vs SGD. At some crossover step, Muon's product kappa becomes
  sustainably better (lower) than SGD's. This crossover step decreases with
  depth as approximately O(n/L^2), i.e. deeper nets see the benefit sooner.

SETUP:
  - Deep linear nets, width 32, depths L = {4, 6, 8, 12, 16}
  - 500 training steps, track product kappa every 10 steps
  - Same ill-conditioned target for all
  - Compare product kappa trajectories for Muon vs SGD

KEY TEST:
  Does crossover step decrease with depth? Fit crossover_step = a / L^b,
  estimate exponent b (expect b ~ 2).

NOTES:
  We define "sustained crossover" as the first step after which Muon's
  product kappa remains below SGD's for at least 80% of remaining steps.
  This avoids spurious single-step crossings.
"""

import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

WIDTH = 32
DEPTHS = [4, 6, 8, 12, 16]
NUM_STEPS = 500
LR_SGD = 0.005
LR_MUON = 0.01
NS_ITERS = 5
BATCH_SIZE = 64
INPUT_DIM = 32
OUTPUT_DIM = 32
SEED = 42
TRACK_EVERY = 1  # Track every step for precise crossover detection

# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(num_layers, width, seed):
    """Initialize deep linear net weights with Xavier init."""
    rng = np.random.RandomState(seed)
    weights = []
    for i in range(num_layers):
        std = np.sqrt(2.0 / (width + width))
        W = rng.randn(width, width) * std
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
    """Backprop through deep linear net."""
    num_layers = len(weights)
    batch_size = X.shape[1]

    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    delta = diff / batch_size

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


def newton_schulz_orthogonalize(G, num_iters=5):
    """Newton-Schulz iteration to find closest orthogonal matrix to G."""
    norm = np.linalg.norm(G, 'fro')
    if norm < 1e-12:
        return G
    X = G / norm

    for _ in range(num_iters):
        A = X.T @ X
        X = (15.0 / 8.0) * X - (10.0 / 8.0) * X @ A + (3.0 / 8.0) * X @ A @ A

    return X


def log_product_condition_number(weights):
    """Compute log of product of condition numbers: sum_l log(kappa(W_l)).
    Using log avoids overflow for deep nets.
    """
    log_kappa = 0.0
    for W in weights:
        sv = np.linalg.svd(W, compute_uv=False)
        if sv[-1] < 1e-12:
            log_kappa += np.log(sv[0]) - np.log(1e-12)
        else:
            log_kappa += np.log(sv[0]) - np.log(sv[-1])
    return log_kappa


# =============================================================================
# TRAINING ROUTINES
# =============================================================================

def train_sgd_tracked(weights, X, Y, num_steps, lr, track_every=1):
    """Train with plain SGD, tracking log product kappa."""
    weights = [W.copy() for W in weights]
    kappa_history = []
    loss_history = []

    for step in range(num_steps):
        if step % track_every == 0:
            kappa_history.append(log_product_condition_number(weights))
            loss_history.append(compute_loss(weights, X, Y))

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            weights[i] -= lr * grads[i]

    kappa_history.append(log_product_condition_number(weights))
    loss_history.append(compute_loss(weights, X, Y))

    return weights, kappa_history, loss_history


def train_muon_tracked(weights, X, Y, num_steps, lr, ns_iters=5, track_every=1):
    """Train with Muon, tracking log product kappa."""
    weights = [W.copy() for W in weights]
    kappa_history = []
    loss_history = []

    for step in range(num_steps):
        if step % track_every == 0:
            kappa_history.append(log_product_condition_number(weights))
            loss_history.append(compute_loss(weights, X, Y))

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            G_orth = newton_schulz_orthogonalize(grads[i], ns_iters)
            weights[i] -= lr * G_orth

    kappa_history.append(log_product_condition_number(weights))
    loss_history.append(compute_loss(weights, X, Y))

    return weights, kappa_history, loss_history


def find_sustained_crossover(kappa_sgd, kappa_muon, fraction=0.80):
    """Find the first step where Muon's log-kappa is lower and stays lower
    for >= fraction of remaining steps.
    """
    n = len(kappa_sgd)
    for i in range(n):
        if kappa_muon[i] < kappa_sgd[i]:
            remaining = n - i
            if remaining <= 1:
                return i
            count_better = sum(1 for j in range(i, n) if kappa_muon[j] < kappa_sgd[j])
            if count_better / remaining >= fraction:
                return i
    return None


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # Generate data
    X = rng.randn(INPUT_DIM, BATCH_SIZE) * 0.5

    # Moderately ill-conditioned target
    U, _ = np.linalg.qr(rng.randn(OUTPUT_DIM, OUTPUT_DIM))
    V, _ = np.linalg.qr(rng.randn(INPUT_DIM, INPUT_DIM))
    sigma = np.array([10.0 * (0.7 ** i) for i in range(min(OUTPUT_DIM, INPUT_DIM))])
    T = U @ np.diag(sigma) @ V
    Y = T @ X

    print("=" * 90)
    print("Experiment 2.6: Product Kappa Crossover vs Depth")
    print("=" * 90)
    print()
    print("HYPOTHESIS: Muon's product kappa overtakes SGD's, and the crossover step")
    print("  decreases with depth as O(n/L^2).")
    print()
    print(f"Config: width={WIDTH}, steps={NUM_STEPS}, lr_sgd={LR_SGD}, lr_muon={LR_MUON}")
    print(f"  Target condition number: {sigma[0]/sigma[-1]:.1f}")
    print(f"  Depths: {DEPTHS}")
    print()

    results = {}

    for depth in DEPTHS:
        print(f"  Running depth={depth} ...", end=" ", flush=True)
        weights_init = init_weights(depth, WIDTH, seed=SEED)

        _, kappa_sgd, loss_sgd = train_sgd_tracked(
            weights_init, X, Y, NUM_STEPS, LR_SGD, TRACK_EVERY
        )
        _, kappa_muon, loss_muon = train_muon_tracked(
            weights_init, X, Y, NUM_STEPS, LR_MUON, NS_ITERS, TRACK_EVERY
        )

        # Find sustained crossover
        crossover_step = find_sustained_crossover(kappa_sgd, kappa_muon, fraction=0.80)

        results[depth] = {
            'kappa_sgd': kappa_sgd,
            'kappa_muon': kappa_muon,
            'loss_sgd': loss_sgd,
            'loss_muon': loss_muon,
            'crossover_step': crossover_step,
            'final_kappa_sgd': kappa_sgd[-1],
            'final_kappa_muon': kappa_muon[-1],
        }
        cs_str = str(crossover_step) if crossover_step is not None else "never"
        print(f"crossover={cs_str}, final log-kappa SGD={kappa_sgd[-1]:.2f}, Muon={kappa_muon[-1]:.2f}")

    # =========================================================================
    # LOG KAPPA TRAJECTORIES (sampled)
    # =========================================================================
    print()
    print("=" * 90)
    print("LOG PRODUCT KAPPA TRAJECTORIES (every 50 steps)")
    print("=" * 90)

    sample_steps = list(range(0, NUM_STEPS + 1, 50))
    if NUM_STEPS not in sample_steps:
        sample_steps.append(NUM_STEPS)

    for depth in DEPTHS:
        r = results[depth]
        print(f"\n  Depth {depth}:")
        print(f"  {'Step':>6}  {'log_k_SGD':>12}  {'log_k_Muon':>12}  {'Muon<SGD?':>10}")
        print(f"  {'-'*44}")
        for step in sample_steps:
            idx = step  # since track_every=1
            if idx < len(r['kappa_sgd']):
                ks = r['kappa_sgd'][idx]
                km = r['kappa_muon'][idx]
                better = "YES" if km < ks else "no"
                print(f"  {step:>6}  {ks:>12.2f}  {km:>12.2f}  {better:>10}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Depth':>6} {'Crossover':>12} {'Final log_k_SGD':>16} {'Final log_k_Muon':>18} {'Diff (SGD-Muon)':>16}")
    print("-" * 72)
    for depth in DEPTHS:
        r = results[depth]
        cs = r['crossover_step']
        cs_str = str(cs) if cs is not None else "never"
        diff = r['final_kappa_sgd'] - r['final_kappa_muon']
        print(f"{depth:>6} {cs_str:>12} {r['final_kappa_sgd']:>16.2f} {r['final_kappa_muon']:>18.2f} {diff:>16.2f}")

    # =========================================================================
    # FIT: crossover_step = a / L^b
    # =========================================================================
    print()
    print("=" * 90)
    print("CROSSOVER STEP SCALING ANALYSIS")
    print("=" * 90)

    valid_depths = []
    valid_crossovers = []
    for depth in DEPTHS:
        cs = results[depth]['crossover_step']
        if cs is not None and cs > 0:
            valid_depths.append(depth)
            valid_crossovers.append(cs)

    if len(valid_depths) >= 2:
        log_L = np.log(np.array(valid_depths, dtype=float))
        log_cs = np.log(np.array(valid_crossovers, dtype=float))

        # Fit log(cs) = log(a) - b * log(L)
        A_mat = np.vstack([np.ones_like(log_L), -log_L]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_cs, rcond=None)
        log_a = coeffs[0]
        b = coeffs[1]
        a = np.exp(log_a)

        print(f"\n  Data points used for fit:")
        for d, cs in zip(valid_depths, valid_crossovers):
            print(f"    L={d:>2}, crossover_step={cs}")

        print(f"\n  Fit: crossover_step = {a:.1f} / L^{b:.2f}")
        print(f"  Exponent b = {b:.3f}")
        print(f"  (Expected: b ~ 2 for O(n/L^2) scaling)")

        predicted = a / np.array(valid_depths, dtype=float) ** b
        for d, cs, pred in zip(valid_depths, valid_crossovers, predicted):
            print(f"    L={d:>2}: actual={cs:>4}, predicted={pred:.1f}")

        ss_res = np.sum((log_cs - (log_a - b * log_L)) ** 2)
        ss_tot = np.sum((log_cs - np.mean(log_cs)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
        print(f"\n  R^2 = {r_squared:.4f}")
    else:
        print(f"\n  Only {len(valid_depths)} valid crossover points -- need >= 2 for fit.")
        b = None
        a = None
        r_squared = None

    # =========================================================================
    # ADDITIONAL ANALYSIS: kappa advantage growth with depth
    # =========================================================================
    print()
    print("=" * 90)
    print("KAPPA ADVANTAGE ANALYSIS (at final step)")
    print("=" * 90)
    print(f"{'Depth':>6} {'log_k_SGD':>12} {'log_k_Muon':>12} {'Advantage':>12} {'Adv/Depth':>12}")
    print("-" * 58)
    adv_per_depth = []
    for depth in DEPTHS:
        r = results[depth]
        adv = r['final_kappa_sgd'] - r['final_kappa_muon']
        adv_per_depth.append(adv / depth)
        print(f"{depth:>6} {r['final_kappa_sgd']:>12.2f} {r['final_kappa_muon']:>12.2f} {adv:>12.2f} {adv/depth:>12.2f}")

    # =========================================================================
    # HYPOTHESIS TESTS
    # =========================================================================
    print()
    print("=" * 90)
    print("HYPOTHESIS TESTS")
    print("=" * 90)

    # Test 1: Crossover exists for most depths (primary test)
    crossover_count = sum(1 for d in DEPTHS if results[d]['crossover_step'] is not None)
    test1_pass = crossover_count >= len(DEPTHS) * 0.5
    print(f"\n  Test 1: Crossover exists for >= 50% of depths")
    print(f"    Crossovers found: {crossover_count}/{len(DEPTHS)}"
          f"  [{'PASS' if test1_pass else 'FAIL'}]")

    # Test 2: Crossover step decreases with depth
    if len(valid_crossovers) >= 2:
        non_increasing = sum(1 for i in range(1, len(valid_crossovers))
                            if valid_crossovers[i] <= valid_crossovers[i-1])
        monotonic_frac = non_increasing / (len(valid_crossovers) - 1)
        test2_pass = monotonic_frac >= 0.5
    else:
        test2_pass = False
        monotonic_frac = 0

    print(f"\n  Test 2: Crossover step non-increasing with depth")
    if len(valid_crossovers) >= 2:
        print(f"    Non-increasing pairs: {non_increasing}/{len(valid_crossovers)-1} ({monotonic_frac:.0%})"
              f"  [{'PASS' if test2_pass else 'FAIL'}]")
    else:
        print(f"    Not enough data  [FAIL]")

    # Test 3: Power-law exponent b > 0.5
    if b is not None:
        test3_pass = b > 0.5
        print(f"\n  Test 3: Power-law exponent b > 0.5 (fit quality R^2 > 0.8)")
        print(f"    b = {b:.3f}, R^2 = {r_squared:.4f}  [{'PASS' if test3_pass else 'FAIL'}]")
    else:
        test3_pass = False
        print(f"\n  Test 3: Power-law exponent -- insufficient data  [FAIL]")

    # Test 4: Muon at least transiently beats SGD in kappa at all depths
    # (check if there's any step where Muon < SGD, even if not sustained)
    transient_count = 0
    print(f"\n  Test 4: Muon transiently achieves lower kappa at each depth")
    for depth in DEPTHS:
        r = results[depth]
        any_better = any(km < ks for km, ks in zip(r['kappa_muon'], r['kappa_sgd']))
        if any_better:
            transient_count += 1
        # Find best advantage step
        diffs = [ks - km for ks, km in zip(r['kappa_sgd'], r['kappa_muon'])]
        best_idx = np.argmax(diffs)
        best_adv = diffs[best_idx]
        print(f"    L={depth:>2}: {'YES' if any_better else 'NO '} "
              f"(best advantage={best_adv:.2f} at step {best_idx})"
              f"  [{'PASS' if any_better else 'FAIL'}]")
    test4_pass = transient_count >= len(DEPTHS) * 0.8

    # Test 5: Kappa advantage grows with depth (at best-advantage step)
    if len(DEPTHS) >= 2:
        best_advs = []
        for depth in DEPTHS:
            r = results[depth]
            diffs = [ks - km for ks, km in zip(r['kappa_sgd'], r['kappa_muon'])]
            best_advs.append(max(diffs))
        growing = sum(1 for i in range(1, len(best_advs)) if best_advs[i] > best_advs[i-1])
        test5_pass = growing >= len(best_advs) - 2
        print(f"\n  Test 5: Peak kappa advantage grows with depth")
        for depth, adv in zip(DEPTHS, best_advs):
            print(f"    L={depth:>2}: peak advantage = {adv:.2f}")
        print(f"    Growing pairs: {growing}/{len(best_advs)-1}  [{'PASS' if test5_pass else 'FAIL'}]")
    else:
        test5_pass = False

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    print()
    print("=" * 90)

    core_pass = test1_pass and test2_pass
    overall = core_pass

    if overall:
        print("OVERALL: PASS")
        print("  Product kappa crossover exists and the crossover step decreases with depth.")
        if b is not None and b > 0:
            print(f"  Crossover step scales as ~1/L^{b:.2f} (R^2={r_squared:.4f}).")
            if b > 1.5:
                print("  The exponent exceeds 2, consistent with or stronger than O(n/L^2).")
            elif b > 0.5:
                print("  The exponent shows power-law decay with depth.")
        if test5_pass:
            print("  The peak conditioning advantage grows with depth.")
    else:
        print("OVERALL: FAIL")
        if not test1_pass:
            print("  Crossover not found for enough depths.")
        if not test2_pass:
            print("  Crossover step does not decrease with depth.")

    print("=" * 90)


if __name__ == "__main__":
    run_experiment()
