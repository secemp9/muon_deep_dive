#!/usr/bin/env python3
"""
Exp 2.1b: Does Conjugation Covariance Persist Through Momentum?
================================================================

KEY QUESTION: Momentum breaks equivariance because:
  m_t = beta * m_{t-1} + (1-beta) * G_t
accumulates gradients from different W_t values (which have drifted from the
initial rotation). After 50 training steps, is equivariance still approximately
preserved?

Protocol:
  - Generate random W_0, training data (X, Y), and orthogonal R, S.
  - Path A: Run 50 Muon steps from W_0 -> W_50
  - Path B: Run 50 Muon steps from R*W_0*S^T -> W_50'
  - If perfectly equivariant: W_50' = R * W_50 * S^T
  - Measure: ||W_50' - R * W_50 * S^T||_F / ||W_50||_F

Two sub-experiments:
  1. ISOLATED (no data dependence): G_t is random noise each step.
     This tests pure optimizer equivariance.
     Expected: EXACT equivariance (because R*noise*S^T has same distribution).
     Actually: the SAME random G_t must be used for both paths, conjugated.

  2. DATA-DRIVEN: G_t comes from actual backprop on shared data.
     The gradients depend on W_t, so the paths diverge.
     Expected: equivariance BREAKS because gradient depends on W nonlinearly
     through the loss function (unless the loss itself is equivariant).

  For a linear net y = W_L ... W_1 x with MSE loss:
     The loss IS equivariant under orthogonal conjugation of a SINGLE layer
     ONLY if adjacent layers also transform. So for a single-layer net
     y = Wx, the loss L(W) = ||Wx - y||^2 is NOT equivariant under
     W -> RWS^T (because x and y don't transform).

  We test both cases:
    (a) Single matrix, random gradients (exact equivariance expected)
    (b) Single matrix, data-driven gradients (equivariance breaks expected)
    (c) Multi-step Muon on a single layer with EQUIVARIANT loss:
        L(W) depends only on singular values of W -> equivariant loss
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

N_STEPS = 50
LR = 0.02
MOMENTUM_BETA = 0.9
NS_ITERS = 5
N_TRIALS = 20
BASE_SEED = 42
MATRIX_SIZES = [(4, 4), (8, 8)]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# NEWTON-SCHULZ
# =============================================================================

def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def random_orthogonal(n, rng):
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


# =============================================================================
# SUB-EXPERIMENT A: Random gradients (conjugated), with momentum
# =============================================================================

def run_random_gradient_test(m, n, rng):
    """
    Both paths use the SAME sequence of random gradients, but path B
    conjugates them: G_t' = R * G_t * S^T.
    This should give EXACT equivariance because:
      m_t' = beta * m_{t-1}' + (1-beta) * R * G_t * S^T
           = R * (beta * m_{t-1} + (1-beta) * G_t) * S^T = R * m_t * S^T
    And NS(R * m_t * S^T) = R * NS(m_t) * S^T.
    So W_t' = R * W_t * S^T at every step.
    """
    W0 = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)

    # Pre-generate random gradients
    gradients = [rng.randn(m, n) for _ in range(N_STEPS)]

    # Path A: original
    W_a = W0.copy()
    mom_a = np.zeros((m, n))
    for t in range(N_STEPS):
        G = gradients[t]
        mom_a = MOMENTUM_BETA * mom_a + (1 - MOMENTUM_BETA) * G
        ortho_mom = newton_schulz(mom_a)
        W_a = W_a - LR * ortho_mom

    # Path B: conjugated
    W_b = R @ W0 @ S.T
    mom_b = np.zeros((m, n))
    for t in range(N_STEPS):
        G_conj = R @ gradients[t] @ S.T
        mom_b = MOMENTUM_BETA * mom_b + (1 - MOMENTUM_BETA) * G_conj
        ortho_mom = newton_schulz(mom_b)
        W_b = W_b - LR * ortho_mom

    # Check: W_b should equal R @ W_a @ S.T
    expected = R @ W_a @ S.T
    err = np.linalg.norm(W_b - expected) / max(np.linalg.norm(W_a), 1e-30)
    return err


# =============================================================================
# SUB-EXPERIMENT B: Data-driven gradients (single layer linear)
# =============================================================================

def run_data_driven_test(m, n, rng):
    """
    Train a single linear layer y = Wx with MSE loss.
    Path A: from W_0. Path B: from R*W_0*S^T.
    Gradients depend on W_t, so equivariance should BREAK
    (because loss is not equivariant under W -> RWS^T).
    """
    N_samples = 50
    W0 = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)

    X = rng.randn(n, N_samples) * 0.3
    Y = rng.randn(m, N_samples) * 0.3

    def compute_grad(W, X, Y):
        pred = W @ X
        return (pred - Y) @ X.T / N_samples

    # Path A
    W_a = W0.copy()
    mom_a = np.zeros((m, n))
    for t in range(N_STEPS):
        G = compute_grad(W_a, X, Y)
        mom_a = MOMENTUM_BETA * mom_a + (1 - MOMENTUM_BETA) * G
        ortho_mom = newton_schulz(mom_a)
        W_a = W_a - LR * ortho_mom

    # Path B
    W_b = R @ W0 @ S.T
    mom_b = np.zeros((m, n))
    for t in range(N_STEPS):
        G = compute_grad(W_b, X, Y)
        mom_b = MOMENTUM_BETA * mom_b + (1 - MOMENTUM_BETA) * G
        ortho_mom = newton_schulz(mom_b)
        W_b = W_b - LR * ortho_mom

    expected = R @ W_a @ S.T
    err = np.linalg.norm(W_b - expected) / max(np.linalg.norm(W_a), 1e-30)
    return err


# =============================================================================
# SUB-EXPERIMENT C: Equivariant loss (||W||_F^2 or spectral loss)
# =============================================================================

def run_equivariant_loss_test(m, n, rng):
    """
    Loss L(W) = ||W - I||_F^2 is NOT equivariant.
    Loss L(W) = sum(sigma_i(W)^2) = ||W||_F^2 IS equivariant.
    Gradient of ||W||_F^2 = 2W.

    But G = 2W, so G' = 2*RWS^T = R * (2W) * S^T = R*G*S^T.
    This is EXACTLY the conjugation of the gradient.
    So equivariance should hold perfectly.
    """
    W0 = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)

    def equivariant_grad(W):
        """Gradient of L(W) = ||W||_F^2 / 2."""
        return W  # gradient = W

    # Path A
    W_a = W0.copy()
    mom_a = np.zeros((m, n))
    for t in range(N_STEPS):
        G = equivariant_grad(W_a)
        mom_a = MOMENTUM_BETA * mom_a + (1 - MOMENTUM_BETA) * G
        ortho_mom = newton_schulz(mom_a)
        W_a = W_a - LR * ortho_mom

    # Path B
    W_b = R @ W0 @ S.T
    mom_b = np.zeros((m, n))
    for t in range(N_STEPS):
        G = equivariant_grad(W_b)
        mom_b = MOMENTUM_BETA * mom_b + (1 - MOMENTUM_BETA) * G
        ortho_mom = newton_schulz(mom_b)
        W_b = W_b - LR * ortho_mom

    expected = R @ W_a @ S.T
    err = np.linalg.norm(W_b - expected) / max(np.linalg.norm(W_a), 1e-30)
    return err


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 90)
print("Exp 2.1b: DOES CONJUGATION COVARIANCE PERSIST THROUGH MOMENTUM?")
print("=" * 90)
print(f"Steps: {N_STEPS}, Momentum: {MOMENTUM_BETA}, LR: {LR}")
print(f"Trials: {N_TRIALS} per size per sub-experiment")
print(f"Sizes: {MATRIX_SIZES}")
print()
print("SUB-EXPERIMENTS:")
print("  A: Random gradients (conjugated) -- EXACT equivariance expected")
print("  B: Data-driven gradients -- equivariance BREAKS expected")
print("  C: Equivariant loss (||W||_F^2) -- EXACT equivariance expected")
print("=" * 90)


# ---- Sub-experiment A ----
print(f"\n\n{'=' * 90}")
print("SUB-EXPERIMENT A: Random Gradients (Conjugated)")
print(f"{'=' * 90}")

results_A = {}
for (m, n) in MATRIX_SIZES:
    errors = []
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(BASE_SEED + trial * 17)
        err = run_random_gradient_test(m, n, rng)
        errors.append(err)
    errors = np.array(errors)
    results_A[(m, n)] = errors
    print(f"\n  Size {m}x{n} ({N_STEPS} steps, {N_TRIALS} trials):")
    print(f"    Mean rel error: {np.mean(errors):.2e}")
    print(f"    Max rel error:  {np.max(errors):.2e}")
    print(f"    Min rel error:  {np.min(errors):.2e}")


# ---- Sub-experiment B ----
print(f"\n\n{'=' * 90}")
print("SUB-EXPERIMENT B: Data-Driven Gradients (Single Layer Linear)")
print(f"{'=' * 90}")

results_B = {}
for (m, n) in MATRIX_SIZES:
    errors = []
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(BASE_SEED + trial * 17)
        err = run_data_driven_test(m, n, rng)
        errors.append(err)
    errors = np.array(errors)
    results_B[(m, n)] = errors
    print(f"\n  Size {m}x{n} ({N_STEPS} steps, {N_TRIALS} trials):")
    print(f"    Mean rel error: {np.mean(errors):.2e}")
    print(f"    Max rel error:  {np.max(errors):.2e}")
    print(f"    Min rel error:  {np.min(errors):.2e}")


# ---- Sub-experiment C ----
print(f"\n\n{'=' * 90}")
print("SUB-EXPERIMENT C: Equivariant Loss (||W||_F^2)")
print(f"{'=' * 90}")

results_C = {}
for (m, n) in MATRIX_SIZES:
    errors = []
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(BASE_SEED + trial * 17)
        err = run_equivariant_loss_test(m, n, rng)
        errors.append(err)
    errors = np.array(errors)
    results_C[(m, n)] = errors
    print(f"\n  Size {m}x{n} ({N_STEPS} steps, {N_TRIALS} trials):")
    print(f"    Mean rel error: {np.mean(errors):.2e}")
    print(f"    Max rel error:  {np.max(errors):.2e}")
    print(f"    Min rel error:  {np.min(errors):.2e}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

print(f"\n\n{'=' * 90}")
print("COMPARISON TABLE: Mean Relative Error After {N_STEPS} Steps")
print(f"{'=' * 90}")

print(f"\n{'Size':>6}  {'A (random G)':>14}  {'B (data-driven)':>16}  {'C (equiv loss)':>15}")
print("-" * 55)

for (m, n) in MATRIX_SIZES:
    a = np.mean(results_A[(m, n)])
    b = np.mean(results_B[(m, n)])
    c = np.mean(results_C[(m, n)])
    print(f"{m}x{n:>3}  {a:>14.2e}  {b:>16.2e}  {c:>15.2e}")


# =============================================================================
# STEP-BY-STEP DRIFT ANALYSIS (for one trial, tracking error at each step)
# =============================================================================

print(f"\n\n{'=' * 90}")
print("STEP-BY-STEP DRIFT ANALYSIS (single trial, 8x8)")
print(f"{'=' * 90}")

m, n = 8, 8
rng = np.random.RandomState(BASE_SEED)

# Data-driven: track error at each step
N_samples = 50
W0 = rng.randn(m, n)
R = random_orthogonal(m, rng)
S = random_orthogonal(n, rng)
X = rng.randn(n, N_samples) * 0.3
Y = rng.randn(m, N_samples) * 0.3

def compute_grad_fn(W, X, Y):
    pred = W @ X
    return (pred - Y) @ X.T / N_samples

W_a = W0.copy()
W_b = R @ W0 @ S.T
mom_a = np.zeros((m, n))
mom_b = np.zeros((m, n))

print(f"\n{'Step':>6}  {'Data-driven err':>16}  {'Equiv loss err':>15}")
print("-" * 42)

# Also track equivariant loss path
W_ae = W0.copy()
W_be = R @ W0 @ S.T
mom_ae = np.zeros((m, n))
mom_be = np.zeros((m, n))

for t in range(N_STEPS):
    # Data-driven
    G_a = compute_grad_fn(W_a, X, Y)
    G_b = compute_grad_fn(W_b, X, Y)
    mom_a = MOMENTUM_BETA * mom_a + (1 - MOMENTUM_BETA) * G_a
    mom_b = MOMENTUM_BETA * mom_b + (1 - MOMENTUM_BETA) * G_b
    W_a = W_a - LR * newton_schulz(mom_a)
    W_b = W_b - LR * newton_schulz(mom_b)

    err_data = np.linalg.norm(W_b - R @ W_a @ S.T) / max(np.linalg.norm(W_a), 1e-30)

    # Equivariant loss
    G_ae = W_ae
    G_be = W_be
    mom_ae = MOMENTUM_BETA * mom_ae + (1 - MOMENTUM_BETA) * G_ae
    mom_be = MOMENTUM_BETA * mom_be + (1 - MOMENTUM_BETA) * G_be
    W_ae = W_ae - LR * newton_schulz(mom_ae)
    W_be = W_be - LR * newton_schulz(mom_be)

    err_equiv = np.linalg.norm(W_be - R @ W_ae @ S.T) / max(np.linalg.norm(W_ae), 1e-30)

    if t in [0, 1, 2, 5, 10, 20, 30, 40, 49]:
        print(f"{t:>6}  {err_data:>16.2e}  {err_equiv:>15.2e}")


# =============================================================================
# ALSO: random gradients, step-by-step
# =============================================================================

print(f"\n\nRandom gradient step-by-step (8x8, single trial):")
rng2 = np.random.RandomState(BASE_SEED + 777)
W0 = rng2.randn(m, n)
R = random_orthogonal(m, rng2)
S = random_orthogonal(n, rng2)
gradients = [rng2.randn(m, n) for _ in range(N_STEPS)]

W_a = W0.copy()
W_b = R @ W0 @ S.T
mom_a = np.zeros((m, n))
mom_b = np.zeros((m, n))

print(f"\n{'Step':>6}  {'Rel error':>12}")
print("-" * 22)

for t in range(N_STEPS):
    G = gradients[t]
    G_conj = R @ G @ S.T

    mom_a = MOMENTUM_BETA * mom_a + (1 - MOMENTUM_BETA) * G
    mom_b = MOMENTUM_BETA * mom_b + (1 - MOMENTUM_BETA) * G_conj
    W_a = W_a - LR * newton_schulz(mom_a)
    W_b = W_b - LR * newton_schulz(mom_b)

    err = np.linalg.norm(W_b - R @ W_a @ S.T) / max(np.linalg.norm(W_a), 1e-30)

    if t in [0, 1, 2, 5, 10, 20, 30, 40, 49]:
        print(f"{t:>6}  {err:>12.2e}")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 90}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 90}")

# H1: Random gradients maintain exact equivariance (<1e-12)
all_A = np.concatenate([results_A[k] for k in MATRIX_SIZES])
h1 = np.max(all_A) < 1e-12
print(f"\nH1: Random gradient equivariance holds after {N_STEPS} steps (<1e-12)?")
print(f"    Max error: {np.max(all_A):.2e}")
print(f"    --> {'PASS' if h1 else 'FAIL'}")

# H2: Data-driven breaks equivariance (error > 0.01)
all_B = np.concatenate([results_B[k] for k in MATRIX_SIZES])
h2 = np.mean(all_B) > 0.01
print(f"\nH2: Data-driven equivariance breaks (mean error > 0.01)?")
print(f"    Mean error: {np.mean(all_B):.2e}")
print(f"    --> {'PASS' if h2 else 'FAIL'}")

# H3: Equivariant loss maintains exact equivariance (<1e-12)
all_C = np.concatenate([results_C[k] for k in MATRIX_SIZES])
h3 = np.max(all_C) < 1e-12
print(f"\nH3: Equivariant loss maintains equivariance after {N_STEPS} steps (<1e-12)?")
print(f"    Max error: {np.max(all_C):.2e}")
print(f"    --> {'PASS' if h3 else 'FAIL'}")

# H4: Data-driven error is orders of magnitude larger than random/equivariant
ratio_B_over_A = np.mean(all_B) / max(np.mean(all_A), 1e-30)
ratio_B_over_C = np.mean(all_B) / max(np.mean(all_C), 1e-30)
h4 = ratio_B_over_A > 1e6
print(f"\nH4: Data-driven error >> random gradient error (ratio > 1e6)?")
print(f"    Ratio B/A: {ratio_B_over_A:.2e}")
print(f"    Ratio B/C: {ratio_B_over_C:.2e}")
print(f"    --> {'PASS' if h4 else 'FAIL'}")

total_pass = sum([h1, h2, h3, h4])


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 90}")
print(f"FINAL VERDICT: Exp 2.1b COVARIANCE THROUGH MOMENTUM ({N_STEPS} steps)")
print(f"{'=' * 90}")

print(f"""
  Sub-experiment A (random conjugated gradients):
    Mean error: {np.mean(all_A):.2e}  -- {"EXACT" if np.max(all_A) < 1e-12 else "BROKEN"}

  Sub-experiment B (data-driven gradients):
    Mean error: {np.mean(all_B):.2e}  -- {"EXACT" if np.max(all_B) < 1e-12 else "BROKEN"}

  Sub-experiment C (equivariant loss ||W||^2):
    Mean error: {np.mean(all_C):.2e}  -- {"EXACT" if np.max(all_C) < 1e-12 else "BROKEN"}

  Tests passed: {total_pass}/4
""")

print("  CONCLUSION:")
print("  - Muon + momentum IS exactly equivariant when the gradient function")
print("    itself is equivariant (G(RWS^T) = R*G(W)*S^T).")
print("  - This holds for: random conjugated gradients, equivariant losses.")
print("  - This BREAKS for: standard data-driven losses, because the loss")
print("    L(W) = ||Wx - y||^2 is NOT invariant under W -> RWS^T.")
print("  - In neural networks, equivariance holds for the INTER-LAYER gauge")
print("    (W_{l+1} -> R*W_{l+1}, W_l -> W_l*R^T) ONLY if both layers")
print("    are updated simultaneously and consistently.")

print(f"\n{'=' * 90}")
