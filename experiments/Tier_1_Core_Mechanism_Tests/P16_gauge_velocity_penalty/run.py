#!/usr/bin/env python3
"""
P16: Gauge Velocity Penalty — Testing the Fluid Dynamics Prediction
====================================================================
PREDICTION (from fluid dynamics / RG gauge-fixing perspective):
  Adding a penalty lambda * ||Q_t - Q_{t-1}||^2_F to SGD's loss (where Q_t is the
  orthogonal factor from polar decomposition W_t = Q_t P_t) should recover MOST of
  Muon's advantage WITHOUT ever computing ortho(gradient).

  The penalty directly suppresses gauge rotation velocity — it doesn't fix the gauge,
  it just prevents it from CHANGING FAST.

  If this works, Muon's benefit isn't from "projecting onto the correct direction" —
  it's from "preventing gauge accumulation." A cheaper alternative to Muon might exist.

Key comparison:
  (a) SGD baseline
  (b) Muon (full orthogonal projection, reference)
  (c) SGD + gauge velocity penalty (lambda sweep: 0.1, 1.0, 10.0, 100.0)
  (d) SGD + spectral norm clipping (pure norm control, no direction info)
  (e) SGD + weight orthogonalization penalty (Cayley regularizer: lambda*||W^TW - I||^2_F)

Pass criteria:
  - Method (c) with optimal lambda closes >= 60% of the SGD-Muon gap  -> STRONG PASS at >=80%
  - Method (d) closes ~30% (norm control only)
  - Method (e) helps but less than (c)

Setup: 6-layer deep linear net, 32x32, quadratic loss, 300 steps.
"""

import numpy as np

np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

DIM = 32
NUM_LAYERS = 6
NUM_STEPS = 300
BATCH_SIZE = 64
LR_MUON = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
LAMBDA_VALUES = [0.1, 1.0, 10.0, 100.0]

# Random target matrix
W_target = np.random.randn(DIM, DIM) * 0.5

# Random input data (fixed batch)
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3


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
    return 0.5 * np.mean(np.sum(diff**2, axis=0))


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


def polar_decomposition(W):
    """Compute polar decomposition W = Q P where Q is orthogonal, P is symmetric PSD."""
    U, S, Vt = np.linalg.svd(W, full_matrices=True)
    Q = U @ Vt
    P = Vt.T @ np.diag(S) @ Vt
    return Q, P


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


def condition_number(W):
    """Compute condition number of matrix W."""
    sv = np.linalg.svd(W, compute_uv=False)
    if sv[-1] < 1e-12:
        return 1e12
    return sv[0] / sv[-1]


def product_matrix(weights):
    """Compute product of all weight matrices."""
    W_prod = np.eye(DIM)
    for W in weights:
        W_prod = W @ W_prod
    return W_prod


# =============================================================================
# GAUGE VELOCITY PENALTY GRADIENT
# =============================================================================

def gauge_penalty_gradient(W_current, Q_prev, lam):
    """
    Compute gradient of lambda * ||Q(W) - Q_prev||^2_F w.r.t. W.

    For the polar decomposition W = Q*P (Q orthogonal, P symmetric PSD),
    we use the chain rule through the polar factor.

    The key identity: for W = QP and a perturbation dW:
      Q^T dW = Omega*P + dP
    where Omega is antisymmetric (dQ = Q*Omega) and dP is symmetric.
    Decomposing Q^T*dW = M into symmetric and antisymmetric parts:
      dP = sym(M) + sym(Omega*P - P*Omega)/... actually:
      Omega*P + dP = M
      Omega antisym, dP sym => separate:
        dP = sym(M) (symmetric part of Q^T*dW)
        Omega*P + P*Omega^T = antisym(M)  ... no.

    CORRECT: from Q^T dW = Omega P + dP, taking antisymmetric part:
      Omega*P - P*Omega = antisym(Q^T*dW) = (Q^T dW - dW^T Q)/2 * 2
      Actually: (Omega*P + dP) = Q^T*dW
      sym(Q^T*dW) = sym(Omega*P) + dP = (Omega*P + P*Omega^T)/2 + dP
                  = (Omega*P - P*Omega)/2 + dP  [since Omega^T = -Omega]
      antisym(Q^T*dW) = antisym(Omega*P) = (Omega*P + P*Omega)/2

    So: antisym(Q^T*dW) = (Omega*P + P*Omega)/2
    => Omega*P + P*Omega = 2*antisym(Q^T*dW)

    This is a Sylvester equation for Omega given dW.

    Now df/dW where f = ||Q - Q_prev||^2:
      df = 2*tr((Q-Q_prev)^T * dQ) = 2*tr((Q-Q_prev)^T * Q * Omega)
         = 2*tr(Omega * (Q-Q_prev)^T * Q)  [cyclic]
         = 2*tr(Omega * B) where B = (Q-Q_prev)^T * Q

    Since Omega is antisymmetric: tr(Omega * B) = tr(Omega * antisym(B))
    Let A_B = antisym(B) = (B - B^T)/2

    So df = 2*tr(Omega * 2*A_B) = 4*tr(Omega * A_B)  ... wait let me be careful.
    df = 2*tr((Q-Q_prev)^T * Q * Omega)

    Now Omega solves: Omega*P + P*Omega = 2*antisym(Q^T*dW) = Q^T*dW - dW^T*Q

    For df = <grad_W f, dW>, we need to express this in terms of dW.
    Using the Sylvester operator L(X) = X*P + P*X:
      Omega = L^{-1}(Q^T*dW - dW^T*Q)

    df = 2*tr((Q-Q_prev)^T * Q * L^{-1}(Q^T*dW - dW^T*Q))

    The adjoint of L^{-1} w.r.t. Frobenius is L^{-1} itself (self-adjoint),
    since L is self-adjoint on symmetric AND antisymmetric subspaces separately.

    Let C = (Q-Q_prev)^T * Q, and A_C = antisym(C) = (C - C^T)/2.
    Then: df = 2*tr(C * L^{-1}(Q^T*dW - dW^T*Q))
             = 2*tr(A_C * L^{-1}(Q^T*dW - dW^T*Q))  [only antisym part contributes]

    Hmm, actually since Omega is antisymmetric and L^{-1} maps antisym to antisym:
    df = 2*tr(C * Omega) = 2*tr(antisym(C) * Omega) * 2 ... no.
    tr(C * Omega) = tr(sym(C)*Omega) + tr(antisym(C)*Omega) = 0 + tr(antisym(C)*Omega)
    since tr(sym * antisym) = 0.

    So df = 2*tr(antisym(C) * Omega) where Omega = L^{-1}(Q^T*dW - dW^T*Q)

    By self-adjointness: = 2*tr(L^{-1}(antisym(C)) * (Q^T*dW - dW^T*Q))
    Let S = L^{-1}(2*antisym(C)) [solve P*S + S*P = 2*antisym(C) = C - C^T]

    df = tr(S * (Q^T*dW - dW^T*Q)) = tr(S*Q^T*dW) - tr(S*dW^T*Q)
       = tr(S*Q^T*dW) - tr(Q*S*dW^T) [cyclic]
       = tr(S*Q^T*dW) - tr((Q*S)^T*dW^T) ... hmm
    tr(S * dW^T * Q) = tr(Q * S * dW^T) [cyclic] = sum_{ij} (QS)_{ij} (dW^T)_{ji}
                     = sum_{ij} (QS)_{ij} dW_{ij} = tr((QS)^T * dW) ... no.
    tr(A * B^T) = sum_{ij} A_{ij} B_{ij}. So tr(QS * dW^T) = <QS, dW>_F.

    Actually: tr(S * dW^T * Q) = tr(Q * S * dW^T) [cyclic property of trace]
    And tr(M * dW^T) = <M^T, dW^T>... let me just use index notation.

    tr(S * Q^T * dW) = sum_{ij} (S * Q^T)_{ij} dW_{ji} = sum_{ij} (Q*S^T)_{ji} dW_{ji}
    Wait: tr(A*B) = sum_i (AB)_{ii} = sum_{ij} A_{ij} B_{ji}
    So tr(S*Q^T*dW) = sum_{ij} (S*Q^T)_{ij} * dW_{ji}
                    = sum_{ji} (S*Q^T)_{ij} * dW_{ji}
    This equals <(S*Q^T)^T, dW>_F = <Q*S^T, dW>_F

    And tr(S*dW^T*Q) = sum_{ij} (S)_{ik} (dW^T)_{kl} Q_{lj} delta_{ij}
    ... this is getting complicated. Let me just use:
    tr(S * dW^T * Q) = tr(Q * S * dW^T) = <(QS)^T, dW^T> ... no.
    tr(M * N^T) = <M, N>_F. So tr(QS * dW^T) = <QS, dW>_F.

    Therefore: df = <Q*S^T, dW> - <Q*S, dW> = <Q*(S^T - S), dW>
    Since S is antisymmetric (S^T = -S): = <Q*(-2S), dW> = <-2*Q*S, dW>

    So: grad_W f = -2*Q*S where S solves P*S + S*P = C - C^T,
    and C = (Q - Q_prev)^T * Q.

    Let's verify: C = (Q-A)^T * Q where A = Q_prev.
    C - C^T = (Q-A)^T*Q - Q^T*(Q-A) = Q^T*Q - A^T*Q - Q^T*Q + Q^T*A
            = Q^T*A - A^T*Q

    So the RHS of the Sylvester equation is Q^T*Q_prev - Q_prev^T*Q.
    And grad_W ||Q - Q_prev||^2 = -2*Q*S.
    """
    Q_current, P_current = polar_decomposition(W_current)
    penalty_value = np.sum((Q_current - Q_prev)**2)

    # RHS of Sylvester equation: Q^T*Q_prev - Q_prev^T*Q
    RHS = Q_current.T @ Q_prev - Q_prev.T @ Q_current  # antisymmetric

    # Solve P*S + S*P = RHS in eigenspace of P
    eigvals_P, V_P = np.linalg.eigh(P_current)
    eigvals_P = np.maximum(eigvals_P, 1e-10)

    RHS_rot = V_P.T @ RHS @ V_P
    D_sum = eigvals_P[:, None] + eigvals_P[None, :]
    S_rot = RHS_rot / D_sum
    S = V_P @ S_rot @ V_P.T

    # Gradient: grad_W ||Q - Q_prev||^2 = -2*Q*S
    grad = -2.0 * Q_current @ S

    return lam * grad, penalty_value


def gauge_penalty_gradient_fd(W_current, Q_prev, lam, epsilon=1e-5):
    """
    Compute gradient via finite differences for verification.
    """
    Q_current, _ = polar_decomposition(W_current)
    penalty_value = np.sum((Q_current - Q_prev)**2)

    grad = np.zeros_like(W_current)
    for i in range(DIM):
        for j in range(DIM):
            W_plus = W_current.copy()
            W_plus[i, j] += epsilon
            Q_plus, _ = polar_decomposition(W_plus)
            f_plus = np.sum((Q_plus - Q_prev)**2)

            W_minus = W_current.copy()
            W_minus[i, j] -= epsilon
            Q_minus, _ = polar_decomposition(W_minus)
            f_minus = np.sum((Q_minus - Q_prev)**2)

            grad[i, j] = (f_plus - f_minus) / (2 * epsilon)

    return lam * grad, penalty_value


# =============================================================================
# VERIFY ANALYTIC GRADIENT
# =============================================================================

print("Verifying analytic gradient against finite differences...")
rng_test = np.random.RandomState(123)
W_test = np.eye(DIM) + rng_test.randn(DIM, DIM) * 0.1
Q_prev_test, _ = polar_decomposition(W_test)
# Perturbation to create a nontrivial Q difference
W_test = W_test + rng_test.randn(DIM, DIM) * 0.05

grad_an, pval_an = gauge_penalty_gradient(W_test, Q_prev_test, 1.0)
grad_fd, pval_fd = gauge_penalty_gradient_fd(W_test, Q_prev_test, 1.0)

cosine_sim = np.sum(grad_fd * grad_an) / (np.linalg.norm(grad_fd) * np.linalg.norm(grad_an) + 1e-12)
rel_error = np.linalg.norm(grad_fd - grad_an) / (np.linalg.norm(grad_fd) + 1e-12)
norm_ratio = np.linalg.norm(grad_an) / (np.linalg.norm(grad_fd) + 1e-12)

print(f"  Cosine similarity: {cosine_sim:.6f}")
print(f"  Relative error:    {rel_error:.6f}")
print(f"  Norm ratio (an/fd): {norm_ratio:.6f}")
print(f"  ||grad_FD||: {np.linalg.norm(grad_fd):.6f}, ||grad_AN||: {np.linalg.norm(grad_an):.6f}")

if cosine_sim > 0.95 and 0.8 < norm_ratio < 1.2:
    print("  -> EXCELLENT: Analytic gradient matches FD well")
    GRAD_CORRECTION = 1.0
elif cosine_sim > 0.9:
    GRAD_CORRECTION = 1.0 / norm_ratio  # correct the magnitude
    print(f"  -> GOOD direction, correcting magnitude by {GRAD_CORRECTION:.4f}")
else:
    GRAD_CORRECTION = 1.0 / norm_ratio
    print(f"  -> WARNING: Direction imperfect. Correcting magnitude by {GRAD_CORRECTION:.4f}")


# =============================================================================
# OPTIMIZER IMPLEMENTATIONS
# =============================================================================

def find_stable_lr():
    """Find a stable SGD learning rate by testing candidates."""
    candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
    for lr in candidates:
        weights = init_weights(NUM_LAYERS)
        velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
        initial_loss = compute_loss(weights, X_data, W_target)
        stable = True
        for step in range(80):
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


def run_sgd(lr):
    """Standard SGD with momentum."""
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    losses = []

    for step in range(NUM_STEPS + 1):
        loss = compute_loss(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            losses.extend([float('nan')] * (NUM_STEPS - step))
            break
        if step < NUM_STEPS:
            grads = compute_gradients(weights, X_data, W_target)
            for i in range(NUM_LAYERS):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]

    return losses, weights


def run_muon():
    """Muon optimizer (orthogonal projection of gradient)."""
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    losses = []

    for step in range(NUM_STEPS + 1):
        loss = compute_loss(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            losses.extend([float('nan')] * (NUM_STEPS - step))
            break
        if step < NUM_STEPS:
            grads = compute_gradients(weights, X_data, W_target)
            for i in range(NUM_LAYERS):
                ortho_grad = newton_schulz_orthogonalize(grads[i])
                velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                weights[i] -= LR_MUON * velocities[i]

    return losses, weights


def run_sgd_gauge_penalty(lam, lr):
    """
    SGD with gauge velocity penalty.
    At each step, add gradient of lambda * ||Q_t - Q_{t-1}||^2_F to the loss gradient.
    The penalty gradient is normalized so that lambda=1 means the penalty gradient
    has the same magnitude as the loss gradient (lambda controls relative weighting).
    """
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    losses = []
    gauge_penalties = []

    # Initialize Q_prev for each layer
    Q_prevs = []
    for i in range(NUM_LAYERS):
        Q, _ = polar_decomposition(weights[i])
        Q_prevs.append(Q.copy())

    for step in range(NUM_STEPS + 1):
        loss = compute_loss(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            losses.extend([float('nan')] * (NUM_STEPS - step))
            break
        if step < NUM_STEPS:
            grads = compute_gradients(weights, X_data, W_target)
            total_penalty = 0.0

            for i in range(NUM_LAYERS):
                # Compute gauge velocity penalty gradient (with lam=1 for direction)
                penalty_grad, pval = gauge_penalty_gradient(
                    weights[i], Q_prevs[i], 1.0 * GRAD_CORRECTION)
                total_penalty += pval

                # Normalize: scale penalty grad so unit lambda = equal weight to loss grad
                loss_grad_norm = np.linalg.norm(grads[i])
                pen_grad_norm = np.linalg.norm(penalty_grad)
                if pen_grad_norm > 1e-12 and loss_grad_norm > 1e-12:
                    # At lambda=1, penalty has same magnitude as loss gradient
                    scaled_penalty_grad = lam * penalty_grad * (loss_grad_norm / pen_grad_norm)
                else:
                    scaled_penalty_grad = lam * penalty_grad

                # Combined gradient
                combined_grad = grads[i] + scaled_penalty_grad
                velocities[i] = MOMENTUM * velocities[i] + combined_grad
                weights[i] -= lr * velocities[i]

            gauge_penalties.append(total_penalty)

            # Update Q_prev for next step
            for i in range(NUM_LAYERS):
                Q, _ = polar_decomposition(weights[i])
                Q_prevs[i] = Q.copy()

    return losses, weights, gauge_penalties


def run_sgd_gauge_penalty_raw(lam, lr):
    """
    SGD with gauge velocity penalty — RAW version without normalization.
    Lambda directly scales the penalty gradient magnitude.
    """
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    losses = []
    gauge_penalties = []

    # Initialize Q_prev for each layer
    Q_prevs = []
    for i in range(NUM_LAYERS):
        Q, _ = polar_decomposition(weights[i])
        Q_prevs.append(Q.copy())

    for step in range(NUM_STEPS + 1):
        loss = compute_loss(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            losses.extend([float('nan')] * (NUM_STEPS - step))
            break
        if step < NUM_STEPS:
            grads = compute_gradients(weights, X_data, W_target)
            total_penalty = 0.0

            for i in range(NUM_LAYERS):
                penalty_grad, pval = gauge_penalty_gradient(
                    weights[i], Q_prevs[i], lam * GRAD_CORRECTION)
                total_penalty += pval
                combined_grad = grads[i] + penalty_grad
                velocities[i] = MOMENTUM * velocities[i] + combined_grad
                weights[i] -= lr * velocities[i]

            gauge_penalties.append(total_penalty)

            # Update Q_prev
            for i in range(NUM_LAYERS):
                Q, _ = polar_decomposition(weights[i])
                Q_prevs[i] = Q.copy()

    return losses, weights, gauge_penalties


def run_sgd_spectral_clip(lr):
    """SGD with spectral norm clipping: clip gradient so ||G||_op <= 1."""
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    losses = []

    for step in range(NUM_STEPS + 1):
        loss = compute_loss(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            losses.extend([float('nan')] * (NUM_STEPS - step))
            break
        if step < NUM_STEPS:
            grads = compute_gradients(weights, X_data, W_target)
            for i in range(NUM_LAYERS):
                sv_max = np.linalg.svd(grads[i], compute_uv=False)[0]
                if sv_max > 1.0:
                    grads[i] = grads[i] / sv_max
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]

    return losses, weights


def run_sgd_ortho_penalty(lam, lr):
    """
    SGD with weight orthogonalization penalty (Cayley regularizer).
    Adds lambda * ||W^T W - I||^2_F penalty gradient to the loss gradient.
    """
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
    losses = []

    for step in range(NUM_STEPS + 1):
        loss = compute_loss(weights, X_data, W_target)
        losses.append(loss)
        if np.isnan(loss) or loss > 1e10:
            losses.extend([float('nan')] * (NUM_STEPS - step))
            break
        if step < NUM_STEPS:
            grads = compute_gradients(weights, X_data, W_target)
            for i in range(NUM_LAYERS):
                WtW = weights[i].T @ weights[i]
                diff_orth = WtW - np.eye(DIM)
                # grad of ||W^TW - I||^2_F = 4*W*(W^TW - I)
                penalty_grad = 4.0 * weights[i] @ diff_orth
                combined_grad = grads[i] + lam * penalty_grad
                velocities[i] = MOMENTUM * velocities[i] + combined_grad
                weights[i] -= lr * velocities[i]

    return losses, weights


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("\n" + "=" * 100)
print("P16: GAUGE VELOCITY PENALTY — TESTING THE FLUID DYNAMICS PREDICTION")
print("=" * 100)
print(f"Setup: {NUM_LAYERS}-layer deep linear net (dim={DIM}), quadratic loss, {NUM_STEPS} steps")
print(f"LR_Muon={LR_MUON}, Momentum={MOMENTUM}")
print(f"Lambda values for gauge penalty: {LAMBDA_VALUES}")
print("=" * 100)

# Find stable LR for SGD
lr_sgd = find_stable_lr()
print(f"\nSGD learning rate (max stable): {lr_sgd}")

# --- (a) SGD baseline ---
print(f"\n{'─' * 100}")
print("  (a) Running SGD baseline...")
losses_sgd, weights_sgd_final = run_sgd(lr_sgd)
print(f"      Final loss: {losses_sgd[-1]:.6e}")

# --- (b) Muon reference ---
print(f"\n{'─' * 100}")
print("  (b) Running Muon (orthogonal projection)...")
losses_muon, weights_muon_final = run_muon()
print(f"      Final loss: {losses_muon[-1]:.6e}")

# --- (c) SGD + gauge velocity penalty (NORMALIZED version) ---
print(f"\n{'─' * 100}")
print("  (c) Running SGD + gauge velocity penalty (normalized)...")
results_gauge = {}
for lam in LAMBDA_VALUES:
    print(f"      lambda={lam:.1f}...", end=" ", flush=True)
    losses_gp, weights_gp_final, penalties_gp = run_sgd_gauge_penalty(lam, lr_sgd)
    results_gauge[('norm', lam)] = {
        'losses': losses_gp,
        'weights': weights_gp_final,
        'penalties': penalties_gp,
        'label': f'gauge_norm_lam={lam}'
    }
    final = losses_gp[-1]
    if np.isnan(final):
        print("DIVERGED")
    else:
        print(f"Final loss: {final:.6e}")

# Also try reduced LR for stability
for lam in LAMBDA_VALUES:
    for lr_factor in [0.5, 0.3, 0.1]:
        lr_try = lr_sgd * lr_factor
        losses_gp2, weights_gp2, pen2 = run_sgd_gauge_penalty(lam, lr_try)
        key = ('norm', lam)
        current_final = results_gauge[key]['losses'][-1]
        new_final = losses_gp2[-1]
        if not np.isnan(new_final) and (np.isnan(current_final) or new_final < current_final):
            results_gauge[key] = {
                'losses': losses_gp2,
                'weights': weights_gp2,
                'penalties': pen2,
                'label': f'gauge_norm_lam={lam}_lr={lr_try:.4f}'
            }

# --- (c') SGD + gauge velocity penalty (RAW, large lambda) ---
print(f"\n{'─' * 100}")
print("  (c') Running SGD + gauge velocity penalty (raw, larger lambda range)...")
lambda_raw = [1.0, 10.0, 100.0, 1000.0, 10000.0]
for lam in lambda_raw:
    print(f"      lambda={lam:.0f}...", end=" ", flush=True)
    losses_gp, weights_gp_final, penalties_gp = run_sgd_gauge_penalty_raw(lam, lr_sgd)
    results_gauge[('raw', lam)] = {
        'losses': losses_gp,
        'weights': weights_gp_final,
        'penalties': penalties_gp,
        'label': f'gauge_raw_lam={lam}'
    }
    final = losses_gp[-1]
    if np.isnan(final):
        print("DIVERGED")
    else:
        print(f"Final loss: {final:.6e}")
    # Try reduced LR
    for lr_factor in [0.5, 0.3, 0.1, 0.05]:
        lr_try = lr_sgd * lr_factor
        losses_gp2, weights_gp2, pen2 = run_sgd_gauge_penalty_raw(lam, lr_try)
        key = ('raw', lam)
        current_final = results_gauge[key]['losses'][-1]
        new_final = losses_gp2[-1]
        if not np.isnan(new_final) and (np.isnan(current_final) or new_final < current_final):
            results_gauge[key] = {
                'losses': losses_gp2,
                'weights': weights_gp2,
                'penalties': pen2,
                'label': f'gauge_raw_lam={lam}_lr={lr_try:.4f}'
            }

# --- (d) SGD + spectral norm clipping ---
print(f"\n{'─' * 100}")
print("  (d) Running SGD + spectral norm clipping...")
losses_spec, weights_spec_final = run_sgd_spectral_clip(lr_sgd)
print(f"      Final loss: {losses_spec[-1]:.6e}")

# --- (e) SGD + weight orthogonalization penalty ---
print(f"\n{'─' * 100}")
print("  (e) Running SGD + orthogonalization penalty...")
lambda_ortho_values = [0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05]
results_ortho = {}
for lam in lambda_ortho_values:
    losses_op, weights_op_final = run_sgd_ortho_penalty(lam, lr_sgd)
    results_ortho[lam] = {'losses': losses_op, 'weights': weights_op_final}
    # Also try reduced LR
    for lr_factor in [0.5, 0.3]:
        lr_try = lr_sgd * lr_factor
        losses_op2, weights_op2 = run_sgd_ortho_penalty(lam, lr_try)
        if not np.isnan(losses_op2[-1]) and (np.isnan(results_ortho[lam]['losses'][-1]) or losses_op2[-1] < results_ortho[lam]['losses'][-1]):
            results_ortho[lam] = {'losses': losses_op2, 'weights': weights_op2}

# Print results
for lam in lambda_ortho_values:
    final = results_ortho[lam]['losses'][-1]
    if np.isnan(final):
        print(f"      lambda={lam}: DIVERGED")
    else:
        print(f"      lambda={lam}: Final loss = {final:.6e}")

# --- CRITICAL CONTROL: SGD at reduced LR (to separate LR effect from penalty effect) ---
print(f"\n{'─' * 100}")
print("  CONTROL: SGD at various reduced learning rates (no penalty)...")
sgd_lr_controls = {}
for lr_factor in [1.0, 0.5, 0.3, 0.2, 0.1]:
    lr_try = lr_sgd * lr_factor
    losses_ctrl, weights_ctrl = run_sgd(lr_try)
    sgd_lr_controls[lr_try] = {'losses': losses_ctrl, 'weights': weights_ctrl}
    print(f"      SGD lr={lr_try:.4f}: Final loss = {losses_ctrl[-1]:.6e}")


# =============================================================================
# ANALYSIS: FRACTION OF MUON ADVANTAGE RECOVERED
# =============================================================================

print("\n\n" + "=" * 100)
print("ANALYSIS: FRACTION OF MUON ADVANTAGE RECOVERED")
print("=" * 100)

loss_sgd_final = losses_sgd[-1]
loss_muon_final = losses_muon[-1]
gap = loss_sgd_final - loss_muon_final

print(f"\n  SGD final loss:  {loss_sgd_final:.6e}")
print(f"  Muon final loss: {loss_muon_final:.6e}")
print(f"  Gap (SGD - Muon): {gap:.6e}")

if gap <= 0:
    print("\n  WARNING: Muon did not outperform SGD! The test premise is invalid.")
    gap = max(abs(gap), 1e-10)

def fraction_recovered(loss_method):
    """Fraction of Muon advantage recovered."""
    if np.isnan(loss_method) or loss_method > loss_sgd_final:
        return 0.0
    frac = (loss_sgd_final - loss_method) / gap
    return min(frac, 2.0)

print(f"\n{'Method':<50} | {'Final Loss':>12} | {'Frac Recovered':>14} | {'Cond(Prod)':>11}")
print("-" * 100)

# SGD
kappa_sgd = condition_number(product_matrix(weights_sgd_final))
print(f"{'(a) SGD':<50} | {loss_sgd_final:12.6e} | {0.0:14.4f} | {kappa_sgd:11.2f}")

# Muon
kappa_muon = condition_number(product_matrix(weights_muon_final))
print(f"{'(b) Muon':<50} | {loss_muon_final:12.6e} | {1.0:14.4f} | {kappa_muon:11.2f}")

# Gauge velocity penalty (all variants)
best_gauge_frac = 0.0
best_gauge_key = None
print(f"{'--- Gauge velocity penalty variants ---':<50} |")
for key in sorted(results_gauge.keys(), key=lambda k: (k[0], k[1])):
    loss_gp = results_gauge[key]['losses'][-1]
    frac = fraction_recovered(loss_gp)
    if not np.isnan(loss_gp):
        kappa = condition_number(product_matrix(results_gauge[key]['weights']))
    else:
        kappa = float('inf')
    label = f"  (c) {results_gauge[key]['label']}"
    if not np.isnan(loss_gp):
        print(f"{label:<50} | {loss_gp:12.6e} | {frac:14.4f} | {kappa:11.2f}")
    else:
        print(f"{label:<50} |     DIVERGED   | {frac:14.4f} |        inf")
    if frac > best_gauge_frac:
        best_gauge_frac = frac
        best_gauge_key = key

# Spectral norm clipping
loss_spec_final = losses_spec[-1]
frac_spec = fraction_recovered(loss_spec_final)
kappa_spec = condition_number(product_matrix(weights_spec_final))
print(f"{'(d) SGD + spectral clip':<50} | {loss_spec_final:12.6e} | {frac_spec:14.4f} | {kappa_spec:11.2f}")

# Orthogonalization penalty
best_ortho_frac = 0.0
best_ortho_lam = None
print(f"{'--- Orthogonalization penalty variants ---':<50} |")
for lam in lambda_ortho_values:
    loss_op = results_ortho[lam]['losses'][-1]
    frac = fraction_recovered(loss_op)
    if not np.isnan(loss_op):
        kappa = condition_number(product_matrix(results_ortho[lam]['weights']))
    else:
        kappa = float('inf')
    label = f"  (e) ortho_lam={lam}"
    if not np.isnan(loss_op):
        print(f"{label:<50} | {loss_op:12.6e} | {frac:14.4f} | {kappa:11.2f}")
    else:
        print(f"{label:<50} |     DIVERGED   | {frac:14.4f} |        inf")
    if frac > best_ortho_frac:
        best_ortho_frac = frac
        best_ortho_lam = lam

# CRITICAL CONTROL: Plain SGD at reduced LRs
print(f"{'--- CONTROL: Plain SGD at reduced LR ---':<50} |")
best_sgd_reduced_frac = 0.0
best_sgd_reduced_lr = lr_sgd
for lr_ctrl, data in sorted(sgd_lr_controls.items()):
    loss_ctrl = data['losses'][-1]
    frac = fraction_recovered(loss_ctrl)
    kappa = condition_number(product_matrix(data['weights']))
    label = f"  CONTROL: SGD lr={lr_ctrl:.4f}"
    print(f"{label:<50} | {loss_ctrl:12.6e} | {frac:14.4f} | {kappa:11.2f}")
    if frac > best_sgd_reduced_frac:
        best_sgd_reduced_frac = frac
        best_sgd_reduced_lr = lr_ctrl

# KEY DIAGNOSTIC: Does gauge penalty do better than plain SGD at same LR?
print(f"\n  CRITICAL CHECK: Does gauge penalty beat plain SGD at the same LR?")
print(f"  Best gauge penalty recovery: {best_gauge_frac:.4f} (at LR used by best config)")
print(f"  Best plain SGD at reduced LR: {best_sgd_reduced_frac:.4f} (lr={best_sgd_reduced_lr:.4f})")
gauge_above_lr_control = best_gauge_frac - best_sgd_reduced_frac
print(f"  Gauge penalty advantage OVER plain LR reduction: {gauge_above_lr_control:.4f}")
if abs(gauge_above_lr_control) < 0.05:
    print("  -> The gauge penalty provides NO benefit beyond simply reducing the LR!")
    print("     This means the penalty gradient is ineffective at changing the optimization path.")
elif gauge_above_lr_control > 0.05:
    print("  -> The gauge penalty provides REAL benefit beyond LR reduction.")
else:
    print("  -> The gauge penalty is WORSE than plain LR reduction (penalty hurts).")


# =============================================================================
# LAMBDA SWEEP FOR NORMALIZED GAUGE PENALTY
# =============================================================================

print("\n\n" + "=" * 100)
print("LAMBDA SWEEP: NORMALIZED GAUGE PENALTY (best LR per lambda)")
print("=" * 100)

lambda_sweep = np.logspace(-2, 2, 20)
fracs_sweep = []
best_frac_sweep = 0.0
best_lam_sweep = None

print(f"\n{'Lambda':>10} | {'Best Loss':>12} | {'Frac Recovered':>14} | {'Best LR':>8}")
print("-" * 55)

for lam in lambda_sweep:
    best_loss_this = float('inf')
    best_lr_this = lr_sgd
    for lr_factor in [1.0, 0.5, 0.3, 0.2, 0.1]:
        lr_try = lr_sgd * lr_factor
        losses_try, _, _ = run_sgd_gauge_penalty(lam, lr_try)
        final = losses_try[-1]
        if not np.isnan(final) and final < best_loss_this:
            best_loss_this = final
            best_lr_this = lr_try

    if np.isinf(best_loss_this):
        best_loss_this = float('nan')

    frac = fraction_recovered(best_loss_this)
    fracs_sweep.append(frac)
    if frac > best_frac_sweep:
        best_frac_sweep = frac
        best_lam_sweep = lam
    print(f"{lam:10.4f} | {best_loss_this:12.6e} | {frac:14.4f} | {best_lr_this:8.5f}")

print(f"\n  Best lambda (normalized): {best_lam_sweep:.4f}")
print(f"  Best fraction recovered:  {best_frac_sweep:.4f} ({best_frac_sweep*100:.1f}%)")

# Update best_gauge_frac if sweep found something better
if best_frac_sweep > best_gauge_frac:
    best_gauge_frac = best_frac_sweep


# =============================================================================
# TRAJECTORY ANALYSIS: GAUGE ROTATION VELOCITY OVER TIME
# =============================================================================

print("\n\n" + "=" * 100)
print("TRAJECTORY ANALYSIS: GAUGE ROTATION VELOCITY OVER TIME")
print("=" * 100)
print("Measuring ||Q_t - Q_{t-1}||_F for each method to verify mechanism")

def measure_gauge_velocity(optimizer_type, lr=None, lam=None):
    """Run an optimizer and track the gauge rotation velocity at each step."""
    weights = init_weights(NUM_LAYERS)
    velocities = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]

    # Get initial Q
    Q_prevs = []
    for i in range(NUM_LAYERS):
        Q, _ = polar_decomposition(weights[i])
        Q_prevs.append(Q.copy())

    gauge_velocities = []

    for step in range(NUM_STEPS):
        grads = compute_gradients(weights, X_data, W_target)

        if optimizer_type == 'sgd':
            for i in range(NUM_LAYERS):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]
        elif optimizer_type == 'muon':
            for i in range(NUM_LAYERS):
                ortho_grad = newton_schulz_orthogonalize(grads[i])
                velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                weights[i] -= LR_MUON * velocities[i]
        elif optimizer_type == 'spectral_clip':
            for i in range(NUM_LAYERS):
                sv_max = np.linalg.svd(grads[i], compute_uv=False)[0]
                if sv_max > 1.0:
                    grads[i] = grads[i] / sv_max
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] -= lr * velocities[i]
        elif optimizer_type == 'gauge_penalty':
            for i in range(NUM_LAYERS):
                penalty_grad, _ = gauge_penalty_gradient(
                    weights[i], Q_prevs[i], 1.0 * GRAD_CORRECTION)
                loss_grad_norm = np.linalg.norm(grads[i])
                pen_grad_norm = np.linalg.norm(penalty_grad)
                if pen_grad_norm > 1e-12 and loss_grad_norm > 1e-12:
                    penalty_grad = lam * penalty_grad * (loss_grad_norm / pen_grad_norm)
                else:
                    penalty_grad = lam * penalty_grad
                combined = grads[i] + penalty_grad
                velocities[i] = MOMENTUM * velocities[i] + combined
                weights[i] -= lr * velocities[i]

        # Measure gauge velocity
        total_gv = 0.0
        for i in range(NUM_LAYERS):
            Q, _ = polar_decomposition(weights[i])
            total_gv += np.linalg.norm(Q - Q_prevs[i], 'fro')
            Q_prevs[i] = Q.copy()
        gauge_velocities.append(total_gv / NUM_LAYERS)

    return gauge_velocities

print("\n  Computing gauge velocities...")
gv_sgd = measure_gauge_velocity('sgd', lr=lr_sgd)
gv_muon = measure_gauge_velocity('muon')
gv_spec = measure_gauge_velocity('spectral_clip', lr=lr_sgd)
gv_gauge = measure_gauge_velocity('gauge_penalty', lr=lr_sgd * 0.3, lam=best_lam_sweep if best_lam_sweep else 1.0)

print(f"\n  {'Method':<35} | {'Mean ||dQ||_F':>12} | {'Max ||dQ||_F':>12} | {'Final ||dQ||_F':>14}")
print("  " + "-" * 85)
print(f"  {'SGD':<35} | {np.mean(gv_sgd):12.6f} | {np.max(gv_sgd):12.6f} | {gv_sgd[-1]:14.6f}")
print(f"  {'Muon':<35} | {np.mean(gv_muon):12.6f} | {np.max(gv_muon):12.6f} | {gv_muon[-1]:14.6f}")
print(f"  {'Spectral clip':<35} | {np.mean(gv_spec):12.6f} | {np.max(gv_spec):12.6f} | {gv_spec[-1]:14.6f}")
lbl = f"Gauge penalty (lam={best_lam_sweep:.2f})" if best_lam_sweep else "Gauge penalty"
print(f"  {lbl:<35} | {np.mean(gv_gauge):12.6f} | {np.max(gv_gauge):12.6f} | {gv_gauge[-1]:14.6f}")

muon_gauge_reduction = np.mean(gv_sgd) / (np.mean(gv_muon) + 1e-12)
gauge_pen_reduction = np.mean(gv_sgd) / (np.mean(gv_gauge) + 1e-12)
print(f"\n  Muon reduces mean gauge velocity by:          {muon_gauge_reduction:.2f}x vs SGD")
print(f"  Gauge penalty reduces mean gauge velocity by:  {gauge_pen_reduction:.2f}x vs SGD")


# =============================================================================
# CONVERGENCE CURVES
# =============================================================================

print("\n\n" + "=" * 100)
print("CONVERGENCE CURVES (loss at steps 0, 50, 100, 150, 200, 250, 300)")
print("=" * 100)

sample_steps = [0, 50, 100, 150, 200, 250, 300]

print(f"\n{'Method':<42}", end="")
for s in sample_steps:
    print(f" | {'Step '+str(s):>10}", end="")
print()
print("-" * 122)

def print_curve(label, losses):
    print(f"{label:<42}", end="")
    for s in sample_steps:
        val = losses[s] if s < len(losses) and not np.isnan(losses[s]) else float('nan')
        if np.isnan(val):
            print(f" | {'NaN':>10}", end="")
        else:
            print(f" | {val:10.4e}", end="")
    print()

print_curve("(a) SGD", losses_sgd)
print_curve("(b) Muon", losses_muon)

# Best gauge penalty
if best_gauge_key:
    print_curve(f"(c) Best gauge penalty", results_gauge[best_gauge_key]['losses'])

print_curve("(d) Spectral clip", losses_spec)

# Best ortho penalty
if best_ortho_lam is not None:
    print_curve(f"(e) Best ortho (lam={best_ortho_lam})", results_ortho[best_ortho_lam]['losses'])


# =============================================================================
# DIAGNOSTIC: GRADIENT SCALE ANALYSIS
# =============================================================================

print("\n\n" + "=" * 100)
print("DIAGNOSTIC: GRADIENT SCALE ANALYSIS AT STEP 0")
print("=" * 100)
print("Why does the gauge penalty not help more? Let's examine gradient magnitudes.")

weights_diag = init_weights(NUM_LAYERS)
grads_diag = compute_gradients(weights_diag, X_data, W_target)

Q_prevs_diag = []
for i in range(NUM_LAYERS):
    Q, _ = polar_decomposition(weights_diag[i])
    Q_prevs_diag.append(Q.copy())

# Take one SGD step, then compute penalty gradient
velocities_diag = [np.zeros((DIM, DIM)) for _ in range(NUM_LAYERS)]
for i in range(NUM_LAYERS):
    velocities_diag[i] = MOMENTUM * velocities_diag[i] + grads_diag[i]
    weights_diag[i] -= lr_sgd * velocities_diag[i]

# Now compute penalty gradient
print(f"\n  After 1 SGD step (lr={lr_sgd}):")
for i in range(NUM_LAYERS):
    Q_new, P_new = polar_decomposition(weights_diag[i])
    gauge_change = np.linalg.norm(Q_new - Q_prevs_diag[i], 'fro')
    grads_new = compute_gradients(weights_diag, X_data, W_target)
    pen_grad, pval = gauge_penalty_gradient(weights_diag[i], Q_prevs_diag[i], 1.0)

    loss_grad_norm = np.linalg.norm(grads_new[i])
    pen_grad_norm = np.linalg.norm(pen_grad)
    ratio = pen_grad_norm / (loss_grad_norm + 1e-12)

    if i == 0:
        print(f"  {'Layer':>6} | {'||loss_grad||':>12} | {'||pen_grad||':>12} | {'ratio':>8} | {'||dQ||_F':>10} | {'penalty':>10}")
        print("  " + "-" * 75)
    print(f"  {i:6d} | {loss_grad_norm:12.6f} | {pen_grad_norm:12.6f} | {ratio:8.4f} | {gauge_change:10.6f} | {pval:10.6f}")

print(f"\n  KEY INSIGHT: The penalty gradient magnitude relative to loss gradient.")
print(f"  If ratio << 1, the penalty has no effect unless lambda >> 1.")
print(f"  If ratio ~ 1, lambda=1 provides equal weighting.")


# =============================================================================
# FINAL VERDICT
# =============================================================================

print("\n\n" + "=" * 100)
print("FINAL VERDICT: P16 GAUGE VELOCITY PENALTY TEST")
print("=" * 100)

print(f"""
  Key Results:
  ─────────────────────────────────────────────────────────────────────────────
  SGD final loss:                    {loss_sgd_final:.6e}
  Muon final loss:                   {loss_muon_final:.6e}
  Gap (SGD - Muon):                  {gap:.6e}

  Best gauge velocity penalty (normalized):
    Fraction of Muon advantage recovered: {best_gauge_frac:.4f} ({best_gauge_frac*100:.1f}%)

  Spectral norm clipping:
    Fraction recovered: {frac_spec:.4f} ({frac_spec*100:.1f}%)

  Best orthogonalization penalty:
    lambda = {best_ortho_lam}
    Fraction recovered: {best_ortho_frac:.4f} ({best_ortho_frac*100:.1f}%)

  Gauge velocity (mean ||dQ||_F/step):
    SGD:           {np.mean(gv_sgd):.6f}
    Muon:          {np.mean(gv_muon):.6f}  ({muon_gauge_reduction:.1f}x reduction)
    Gauge penalty: {np.mean(gv_gauge):.6f}  ({gauge_pen_reduction:.1f}x reduction)
  ─────────────────────────────────────────────────────────────────────────────
""")

# Determine pass/fail
if best_gauge_frac >= 0.80:
    verdict = "STRONG PASS"
    verdict_detail = (
        "The gauge velocity penalty recovers >=80% of Muon's advantage.\n"
        "  This confirms the fluid dynamics prediction: Muon's benefit comes from\n"
        "  suppressing gauge rotation velocity, NOT from projecting onto a specific direction.\n"
        "  A cheaper alternative to Muon likely exists: penalize d(Q)/dt instead of\n"
        "  computing ortho(gradient) at each step."
    )
elif best_gauge_frac >= 0.60:
    verdict = "PASS"
    verdict_detail = (
        "The gauge velocity penalty recovers 60-80% of Muon's advantage.\n"
        "  This supports the fluid dynamics prediction, though other mechanisms\n"
        "  (beyond pure gauge velocity control) also contribute to Muon's benefit."
    )
elif best_gauge_frac >= 0.40:
    verdict = "PARTIAL PASS"
    verdict_detail = (
        "The gauge velocity penalty recovers 40-60% of Muon's advantage.\n"
        "  Gauge velocity is A significant factor but not THE dominant one.\n"
        "  Muon likely benefits from both gauge control AND directional information."
    )
elif best_gauge_frac >= 0.20:
    verdict = "WEAK SIGNAL"
    verdict_detail = (
        "The gauge velocity penalty recovers only 20-40% of Muon's advantage.\n"
        "  Gauge velocity suppression helps but is not the primary mechanism.\n"
        "  Muon's orthogonal projection provides value beyond gauge control."
    )
else:
    verdict = "FAIL"
    verdict_detail = (
        "The gauge velocity penalty recovers <20% of Muon's advantage.\n"
        "  The fluid dynamics prediction is NOT confirmed.\n"
        "  Muon's benefit comes primarily from directional information in ortho(gradient),\n"
        "  not from suppressing gauge rotation."
    )

# Diagnostic comparisons
print(f"  Diagnostic comparisons:")
print(f"    (c) gauge penalty ({best_gauge_frac:.2f}) vs (d) spectral clip ({frac_spec:.2f}): ", end="")
if best_gauge_frac > frac_spec + 0.1:
    print("Gauge penalty BETTER (direction matters, not just norm)")
elif abs(best_gauge_frac - frac_spec) <= 0.1:
    print("SIMILAR (gauge penalty ~ spectral clip)")
else:
    print("Spectral clip BETTER (norm control sufficient, gauge penalty not key)")

print(f"    (c) gauge penalty ({best_gauge_frac:.2f}) vs (e) ortho penalty ({best_ortho_frac:.2f}): ", end="")
if best_gauge_frac > best_ortho_frac + 0.1:
    print("Gauge VELOCITY better than gauge POSITION")
    print("      -> Confirms: it's about rate of change, not being on Stiefel manifold")
elif abs(best_gauge_frac - best_ortho_frac) <= 0.1:
    print("SIMILAR (both gauge controls work equally)")
else:
    print("Ortho penalty BETTER (staying on Stiefel > controlling velocity)")
    print("      -> The 'gauge position' matters more than 'gauge velocity'")

print(f"""
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║  VERDICT: {verdict:<67}║
  ╠══════════════════════════════════════════════════════════════════════════════╣
  ║                                                                            ║""")
for line in verdict_detail.split('\n'):
    print(f"  ║  {line:<74}║")
print(f"""  ║                                                                            ║
  ╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Extended interpretation
print("  INTERPRETATION:")
print("  ───────────────")
if best_gauge_frac >= 0.60:
    print("  The mechanism IS about gauge VELOCITY, not gauge POSITION.")
    print("  You don't need to be ON the Stiefel manifold — you just need to not MOVE")
    print("  in gauge directions too fast. This means a cheaper alternative to Muon exists:")
    print("  just penalize the rate of gauge change (no Newton-Schulz iteration needed).")
    print(f"\n  The optimal penalty strength tells us how strongly Muon implicitly")
    print("  suppresses gauge rotation per step.")
elif best_ortho_frac > best_gauge_frac + 0.2:
    print("  SURPRISING RESULT: The ORTHOGONALIZATION penalty (constraining W to Stiefel)")
    print(f"  recovers {best_ortho_frac*100:.1f}% of Muon's advantage vs {best_gauge_frac*100:.1f}% for gauge velocity.")
    print("")
    print("  This suggests Muon's benefit is about GAUGE POSITION, not velocity:")
    print("  - Keeping W^TW close to I (orthogonal weights) matters more than")
    print("    preventing gauge rotation speed.")
    print("  - Muon's ortho(gradient) implicitly keeps the weight matrices well-conditioned.")
    print("  - The correct interpretation: Muon prevents ACCUMULATION of non-orthogonality,")
    print("    not accumulation of gauge rotation speed.")
    print("")
    print("  This still supports a 'gauge fixing' interpretation of Muon, but the")
    print("  relevant gauge is POSITION (how far from Stiefel) not VELOCITY (how fast rotating).")
else:
    print("  The gauge velocity penalty does NOT fully explain Muon's advantage.")
    print("  Muon's orthogonal projection provides directional information that cannot")
    print("  be replicated by simply penalizing gauge rotation speed.")
    print("")
    print("  However, note that:")
    print(f"  - Muon reduces gauge velocity by only {muon_gauge_reduction:.1f}x vs SGD")
    if muon_gauge_reduction < 2.0:
        print("    (This is SMALL — gauge velocity is NOT what Muon primarily controls)")
        print("    Muon's orthogonal projection works by EQUALIZING singular values of")
        print("    the effective gradient, which improves conditioning of the optimization")
        print("    landscape without needing to suppress gauge rotation.")
    else:
        print("    (This is significant — gauge control IS part of the mechanism)")
        print("    But the penalty approach fails because adding a competing gradient")
        print("    conflicts with the loss gradient, while Muon achieves gauge control")
        print("    as a SIDE EFFECT of its projection.")

print("\n" + "=" * 100)
