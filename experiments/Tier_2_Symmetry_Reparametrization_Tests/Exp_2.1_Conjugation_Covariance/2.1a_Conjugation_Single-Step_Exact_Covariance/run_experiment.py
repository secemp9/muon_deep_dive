#!/usr/bin/env python3
"""
Exp 2.1a: Conjugation Single-Step Exact Covariance
====================================================

Verify that Muon's update step is equivariant under conjugation:
  Muon_step(R W S^T, R G S^T) = R * Muon_step(W, G) * S^T

where R, S are orthogonal matrices.

This is the PRACTICAL verification of Axiom 0.3 (which proved the math).
Here we verify it numerically with the full Newton-Schulz iteration,
including the momentum buffer (set to zero for single-step test).

Protocol:
  - Generate random W (m x n), G (m x n), R (m x m orthogonal), S (n x n orthogonal)
  - Compute: W1 = W - lr * NS(G)                       (Muon step from W)
  - Compute: W1' = RWS^T - lr * NS(RGS^T)              (Muon step from rotated)
  - Verify:  W1' == R * W1 * S^T                        (equivariance)

Test with 100 random quadruples at sizes 4x4 and 8x8.

Expected: exact match up to floating-point precision (~1e-14 relative error).
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SIZES = [(4, 4), (8, 8)]
N_TRIALS = 100
LR = 0.02
NS_ITERS = 5
BASE_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# NEWTON-SCHULZ ORTHOGONALIZATION
# =============================================================================

def newton_schulz(M, n_iters=NS_ITERS):
    """Newton-Schulz iteration: converges to polar factor of M."""
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def muon_step(W, G, lr=LR):
    """Single Muon step (no momentum): W_new = W - lr * NS(G)."""
    Q = newton_schulz(G)
    return W - lr * Q


# =============================================================================
# RANDOM ORTHOGONAL MATRIX
# =============================================================================

def random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via QR decomposition."""
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    # Ensure proper orthogonal (det = +1 or -1)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


# =============================================================================
# SINGLE TRIAL
# =============================================================================

def run_trial(m, n, rng):
    """
    Run one equivariance test.
    Returns relative error ||W1' - R W1 S^T|| / ||W1||.
    """
    # Random weight matrix and gradient
    W = rng.randn(m, n)
    G = rng.randn(m, n)

    # Random orthogonal matrices
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)

    # Path 1: Muon step from original, then rotate
    W1 = muon_step(W, G)
    W1_rotated = R @ W1 @ S.T

    # Path 2: Rotate first, then Muon step
    W_rot = R @ W @ S.T
    G_rot = R @ G @ S.T
    W1_prime = muon_step(W_rot, G_rot)

    # Compare
    diff = W1_prime - W1_rotated
    rel_error = np.linalg.norm(diff) / max(np.linalg.norm(W1), 1e-30)

    return rel_error


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 90)
print("Exp 2.1a: CONJUGATION SINGLE-STEP EXACT COVARIANCE")
print("=" * 90)
print(f"Test: Muon_step(RWS^T, RGS^T) = R * Muon_step(W,G) * S^T")
print(f"NS iterations: {NS_ITERS}, lr: {LR}")
print(f"Trials: {N_TRIALS} per size, Sizes: {SIZES}")
print()
print("PREDICTION: Exact equivariance (relative error < 1e-12)")
print("=" * 90)

all_results = {}

for (m, n) in SIZES:
    rng = np.random.RandomState(BASE_SEED)
    errors = []

    for trial in range(N_TRIALS):
        err = run_trial(m, n, rng)
        errors.append(err)

    errors = np.array(errors)
    all_results[(m, n)] = errors

    print(f"\nSize {m}x{n}:")
    print(f"  Mean relative error:   {np.mean(errors):.2e}")
    print(f"  Max relative error:    {np.max(errors):.2e}")
    print(f"  Min relative error:    {np.min(errors):.2e}")
    print(f"  Median relative error: {np.median(errors):.2e}")
    print(f"  Std relative error:    {np.std(errors):.2e}")


# =============================================================================
# DETAILED ERROR DISTRIBUTION
# =============================================================================

print(f"\n\n{'=' * 90}")
print("ERROR DISTRIBUTION")
print(f"{'=' * 90}")

for (m, n) in SIZES:
    errors = all_results[(m, n)]
    print(f"\n  Size {m}x{n}:")

    # Histogram of log10(error)
    log_errors = np.log10(errors + 1e-20)  # floor at -20
    bins = [-18, -16, -15, -14, -13, -12, -10, -8, -5, 0]
    print(f"    Log10(error) distribution:")
    for i in range(len(bins) - 1):
        count = np.sum((log_errors >= bins[i]) & (log_errors < bins[i+1]))
        bar = '#' * count
        print(f"      [{bins[i]:>4}, {bins[i+1]:>4}): {count:>4}  {bar}")

    # Count by order of magnitude
    for threshold in [1e-15, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8]:
        count = np.sum(errors < threshold)
        print(f"    Errors < {threshold:.0e}: {count}/{N_TRIALS}")


# =============================================================================
# CONTROL: Non-orthogonal R, S (should BREAK equivariance)
# =============================================================================

print(f"\n\n{'=' * 90}")
print("CONTROL: Non-orthogonal R, S (should BREAK equivariance)")
print(f"{'=' * 90}")

rng = np.random.RandomState(BASE_SEED + 999)
control_errors = []

for trial in range(N_TRIALS):
    m, n = 4, 4
    W = rng.randn(m, n)
    G = rng.randn(m, n)

    # Non-orthogonal R, S (just random matrices)
    R = rng.randn(m, m) * 0.5 + np.eye(m)  # perturbation of identity
    S = rng.randn(n, n) * 0.5 + np.eye(n)

    W1 = muon_step(W, G)
    W1_rotated = R @ W1 @ S.T

    W_rot = R @ W @ S.T
    G_rot = R @ G @ S.T
    W1_prime = muon_step(W_rot, G_rot)

    diff = W1_prime - W1_rotated
    rel_error = np.linalg.norm(diff) / max(np.linalg.norm(W1), 1e-30)
    control_errors.append(rel_error)

control_errors = np.array(control_errors)
print(f"\n  Non-orthogonal control (4x4, {N_TRIALS} trials):")
print(f"    Mean relative error:   {np.mean(control_errors):.2e}")
print(f"    Max relative error:    {np.max(control_errors):.2e}")
print(f"    Min relative error:    {np.min(control_errors):.2e}")


# =============================================================================
# SENSITIVITY: NS iterations
# =============================================================================

print(f"\n\n{'=' * 90}")
print("SENSITIVITY: NS iterations (equivariance should hold for any iteration count)")
print(f"{'=' * 90}")

for ns_iter in [1, 3, 5, 10, 20]:
    errors = []
    rng_trial = np.random.RandomState(BASE_SEED + ns_iter)
    for trial in range(50):
        m, n = 4, 4
        W = rng_trial.randn(m, n)
        G = rng_trial.randn(m, n)
        R = random_orthogonal(m, rng_trial)
        S = random_orthogonal(n, rng_trial)

        def ns_custom(M):
            return newton_schulz(M, n_iters=ns_iter)

        W1 = W - LR * ns_custom(G)
        W1_rot = R @ W1 @ S.T

        W1_prime = R @ W @ S.T - LR * ns_custom(R @ G @ S.T)
        err = np.linalg.norm(W1_prime - W1_rot) / max(np.linalg.norm(W1), 1e-30)
        errors.append(err)

    errors = np.array(errors)
    print(f"  NS_iters={ns_iter:>2}: mean={np.mean(errors):.2e}, "
          f"max={np.max(errors):.2e}")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 90}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 90}")

# H1: All orthogonal errors < 1e-12 (exact equivariance)
all_ortho_errors = np.concatenate([all_results[k] for k in SIZES])
h1 = np.max(all_ortho_errors) < 1e-12
print(f"\nH1: All orthogonal errors < 1e-12?")
print(f"    Max error: {np.max(all_ortho_errors):.2e}")
print(f"    --> {'PASS' if h1 else 'FAIL'}")

# H2: Non-orthogonal errors are large (>0.01) — equivariance only for orthogonal
h2 = np.mean(control_errors) > 0.01
print(f"\nH2: Non-orthogonal errors > 0.01 (equivariance breaks)?")
print(f"    Mean non-ortho error: {np.mean(control_errors):.2e}")
print(f"    --> {'PASS' if h2 else 'FAIL'}")

# H3: Equivariance holds at both sizes
h3 = all(np.max(all_results[k]) < 1e-12 for k in SIZES)
print(f"\nH3: Equivariance holds at all tested sizes?")
for k in SIZES:
    print(f"    {k[0]}x{k[1]}: max error = {np.max(all_results[k]):.2e}")
print(f"    --> {'PASS' if h3 else 'FAIL'}")

# H4: Mean error is at machine precision level (~1e-15)
h4 = np.mean(all_ortho_errors) < 1e-13
print(f"\nH4: Mean error at machine precision (<1e-13)?")
print(f"    Mean: {np.mean(all_ortho_errors):.2e}")
print(f"    --> {'PASS' if h4 else 'FAIL'}")

total_pass = sum([h1, h2, h3, h4])


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 90}")
print("FINAL VERDICT: Exp 2.1a CONJUGATION COVARIANCE")
print(f"{'=' * 90}")

print(f"""
  Orthogonal equivariance errors:
    4x4:  mean={np.mean(all_results[(4,4)]):.2e}, max={np.max(all_results[(4,4)]):.2e}
    8x8:  mean={np.mean(all_results[(8,8)]):.2e}, max={np.max(all_results[(8,8)]):.2e}

  Non-orthogonal control:
    mean={np.mean(control_errors):.2e}

  Tests passed: {total_pass}/4
""")

if total_pass == 4:
    print("  PERFECT EQUIVARIANCE: Muon_step(RWS^T, RGS^T) = R * Muon_step(W,G) * S^T")
    print("  holds to machine precision for orthogonal R, S.")
    print("  Breaks for non-orthogonal transformations (as expected).")
    print("  This confirms Axiom 0.3 numerically.")
elif h1:
    print("  EQUIVARIANCE CONFIRMED at 1e-12 level.")
else:
    print("  EQUIVARIANCE NOT EXACT: errors exceed 1e-12.")
    print("  Newton-Schulz may introduce numerical drift.")

print(f"\n{'=' * 90}")
