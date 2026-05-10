#!/usr/bin/env python3
"""
Experiment 3.11: Product-Space Muon
=====================================

FROM 2.6: Per-layer ortho fails at depth 16 (product kappa overtakes at step 4).
FIX: every K steps, compute the product W_prod = W_L...W_1, then SVD it:
W_prod = U Sigma V^T. Redistribute: set each W_i = Sigma^{1/L} rotated.

This rebalances the product spectrum so no single layer dominates.

PROTOCOL:
  8-layer deep linear, 32x32, 500 steps.
  Compare:
    (a) Plain Muon (per-layer ortho only)
    (b) Muon + product rebalance every K=50 steps
    (c) SGD (baseline)

  Track product condition number kappa_prod at each step.

PREDICTION:
  (b) prevents the kappa crossover -- product kappa stays controlled
  even at depth 8 where plain Muon struggles.

SETUP:
  - Ill-conditioned target (kappa=100)
  - Xavier init
  - Product kappa = cond(W_L @ ... @ W_1) = sigma_max(prod) / sigma_min(prod)
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
np.random.seed(SEED)

WIDTH = 32
NUM_LAYERS = 8
NUM_STEPS = 500
BATCH_SIZE = 64
REBALANCE_EVERY = 50  # K
NS_ITERS = 5
MOMENTUM = 0.9
NUM_SEEDS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# LR search candidates
LR_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]


# =============================================================================
# NETWORK UTILITIES
# =============================================================================

def init_weights(num_layers, width, rng):
    """Xavier init."""
    weights = []
    for _ in range(num_layers):
        std = np.sqrt(2.0 / (width + width))
        W = rng.randn(width, width) * std
        weights.append(W.copy())
    return weights


def copy_weights(weights):
    return [W.copy() for W in weights]


def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_product_matrix(weights):
    """Compute W_L @ ... @ W_1."""
    prod = weights[0].copy()
    for W in weights[1:]:
        prod = W @ prod
    return prod


def product_condition_number(weights):
    """Condition number of the product matrix."""
    prod = compute_product_matrix(weights)
    svs = np.linalg.svd(prod, compute_uv=False)
    if svs[-1] < 1e-15:
        return 1e15
    return svs[0] / svs[-1]


def compute_loss(weights, X, Y_target):
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))


def compute_gradients(weights, X, Y_target):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])
    delta = (activations[-1] - Y_target) / N
    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


# =============================================================================
# NEWTON-SCHULZ ORTHOGONALIZATION
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
# PRODUCT REBALANCE
# =============================================================================

def rebalance_product(weights):
    """
    Compute product W_prod = W_L ... W_1, SVD it, and redistribute
    singular values equally across layers.

    W_prod = U @ diag(sigma) @ V^T
    Set W_1 = diag(sigma^{1/L}) @ V^T
    Set W_i = diag(sigma^{1/L}) for i in 2..L-1  (with random rotations absorbed)
    Set W_L = U @ diag(sigma^{1/L})

    Actually, a cleaner approach: use the balanced factorization.
    W_prod = U @ diag(sigma) @ V^T
    sigma_balanced = sigma^{1/L}
    W_1 = diag(sigma_balanced) @ V^T
    W_L = U @ diag(sigma_balanced)
    W_i = diag(sigma_balanced) @ Q_i for middle layers (using identity rotations)

    Simplest correct approach that preserves the product:
    W_prod = U @ diag(sigma) @ V^T
    Let S = diag(sigma^{1/L})
    W_1 = S @ V^T
    W_i = S    for i = 2, ..., L-1
    W_L = U @ S
    Product = U @ S @ S @ ... @ S @ V^T = U @ S^L @ V^T = U @ diag(sigma) @ V^T

    But this ignores that S^L = diag(sigma) only if sigma has ALL positive entries.
    Since SVD guarantees sigma >= 0, we're fine.
    """
    L = len(weights)
    prod = compute_product_matrix(weights)
    U, sigma, Vt = np.linalg.svd(prod, full_matrices=True)

    # sigma^{1/L} -- handle zeros
    sigma_balanced = np.power(np.maximum(sigma, 1e-30), 1.0 / L)
    S = np.diag(sigma_balanced)

    new_weights = []
    # Layer 0: S @ V^T
    new_weights.append(S @ Vt)
    # Middle layers: S (diagonal in standard basis)
    for _ in range(1, L - 1):
        new_weights.append(S.copy())
    # Last layer: U @ S
    new_weights.append(U @ S)

    return new_weights


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train(weights_init, Y_target, X, n_steps, method='sgd', lr=0.01,
          rebalance_every=None):
    """
    Train and return losses, product kappas.

    method: 'sgd', 'muon', 'muon_rebalance'
    """
    weights = copy_weights(weights_init)
    velocities = [np.zeros_like(w) for w in weights]

    losses = []
    kappas = []

    for step in range(n_steps):
        loss = compute_loss(weights, X, Y_target)
        kappa = product_condition_number(weights)
        losses.append(loss)
        kappas.append(kappa)

        if np.isnan(loss) or loss > 1e10:
            losses.extend([1e10] * (n_steps - step - 1))
            kappas.extend([1e15] * (n_steps - step - 1))
            break

        grads = compute_gradients(weights, X, Y_target)

        if method == 'sgd':
            for i in range(len(weights)):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] = weights[i] - lr * velocities[i]
        elif method in ('muon', 'muon_rebalance'):
            for i in range(len(weights)):
                ortho_grad = newton_schulz_ortho(grads[i])
                velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                weights[i] = weights[i] - lr * velocities[i]

            # Product rebalance
            if method == 'muon_rebalance' and rebalance_every is not None:
                if (step + 1) % rebalance_every == 0:
                    weights = rebalance_product(weights)
                    # Reset velocities after rebalance (weight space changed)
                    velocities = [np.zeros_like(w) for w in weights]

    # Final measurements
    losses.append(compute_loss(weights, X, Y_target))
    kappas.append(product_condition_number(weights))

    return np.array(losses), np.array(kappas)


def find_best_lr(weights_init, Y_target, X, method, rebalance_every=None):
    """Grid search for best LR."""
    best_loss = 1e20
    best_lr = 0.005
    for lr in LR_CANDIDATES:
        w = copy_weights(weights_init)
        losses, _ = train(w, Y_target, X, min(200, NUM_STEPS), method=method,
                          lr=lr, rebalance_every=rebalance_every)
        final = losses[-1]
        if not np.isnan(final) and final < best_loss:
            best_loss = final
            best_lr = lr
    return best_lr


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 85)
print("Experiment 3.11: Product-Space Muon")
print("=" * 85)
print(f"Setup: {NUM_LAYERS}-layer deep linear ({WIDTH}x{WIDTH})")
print(f"Steps: {NUM_STEPS}, Rebalance every: {REBALANCE_EVERY} steps")
print(f"Seeds: {NUM_SEEDS}")
print("=" * 85)

# Results accumulators
all_results = {
    'sgd': {'final_losses': [], 'final_kappas': [], 'kappa_curves': [], 'loss_curves': []},
    'muon': {'final_losses': [], 'final_kappas': [], 'kappa_curves': [], 'loss_curves': []},
    'muon_rebalance': {'final_losses': [], 'final_kappas': [], 'kappa_curves': [], 'loss_curves': []},
}

for seed_idx in range(NUM_SEEDS):
    run_seed = SEED + seed_idx * 17
    rng = np.random.RandomState(run_seed)

    print(f"\n--- Seed {run_seed} ---")

    # Ill-conditioned target
    U_t, _ = np.linalg.qr(rng.randn(WIDTH, WIDTH))
    V_t, _ = np.linalg.qr(rng.randn(WIDTH, WIDTH))
    sigma_t = np.logspace(2, 0, WIDTH)  # 100 down to 1, kappa=100
    W_target = U_t @ np.diag(sigma_t) @ V_t
    Y_target = W_target @ np.random.randn(WIDTH, BATCH_SIZE) * 0.3
    X = np.random.randn(WIDTH, BATCH_SIZE) * 0.3

    # Same init for all methods
    weights_init = init_weights(NUM_LAYERS, WIDTH, rng)
    init_kappa = product_condition_number(weights_init)
    print(f"  Initial product kappa: {init_kappa:.2f}")

    # Find LRs
    lr_sgd = find_best_lr(weights_init, Y_target, X, 'sgd')
    lr_muon = find_best_lr(weights_init, Y_target, X, 'muon')
    lr_muon_rb = find_best_lr(weights_init, Y_target, X, 'muon_rebalance',
                               rebalance_every=REBALANCE_EVERY)
    print(f"  LRs: SGD={lr_sgd}, Muon={lr_muon}, Muon+Rebal={lr_muon_rb}")

    # Full runs
    for method, lr, rb_every in [('sgd', lr_sgd, None),
                                   ('muon', lr_muon, None),
                                   ('muon_rebalance', lr_muon_rb, REBALANCE_EVERY)]:
        losses, kappas = train(copy_weights(weights_init), Y_target, X,
                               NUM_STEPS, method=method, lr=lr,
                               rebalance_every=rb_every)
        all_results[method]['final_losses'].append(losses[-1])
        all_results[method]['final_kappas'].append(kappas[-1])
        all_results[method]['loss_curves'].append(losses)
        all_results[method]['kappa_curves'].append(kappas)

    print(f"  Finals: SGD loss={all_results['sgd']['final_losses'][-1]:.4e} kappa={all_results['sgd']['final_kappas'][-1]:.1f}")
    print(f"          Muon loss={all_results['muon']['final_losses'][-1]:.4e} kappa={all_results['muon']['final_kappas'][-1]:.1f}")
    print(f"          Muon+RB loss={all_results['muon_rebalance']['final_losses'][-1]:.4e} kappa={all_results['muon_rebalance']['final_kappas'][-1]:.1f}")


# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("AGGREGATE RESULTS")
print(f"{'=' * 85}")

print(f"\n{'Method':<25} {'Mean Loss':>14} {'Std Loss':>14} {'Mean kappa':>14} {'Std kappa':>14}")
print("-" * 85)

for method in ['sgd', 'muon', 'muon_rebalance']:
    mean_loss = np.mean(all_results[method]['final_losses'])
    std_loss = np.std(all_results[method]['final_losses'])
    mean_kappa = np.mean(all_results[method]['final_kappas'])
    std_kappa = np.std(all_results[method]['final_kappas'])

    label = {'sgd': 'SGD', 'muon': 'Muon (per-layer)',
             'muon_rebalance': f'Muon + Rebalance(K={REBALANCE_EVERY})'}[method]
    print(f"{label:<25} {mean_loss:>14.6e} {std_loss:>14.6e} {mean_kappa:>14.1f} {std_kappa:>14.1f}")


# =============================================================================
# KAPPA TRAJECTORY (averaged)
# =============================================================================

print(f"\n\n{'=' * 85}")
print("PRODUCT KAPPA TRAJECTORY (averaged over seeds)")
print(f"{'=' * 85}")

# Pad curves to same length
max_len = NUM_STEPS + 1
print(f"\n{'Step':>6} {'SGD kappa':>14} {'Muon kappa':>14} {'Muon+RB kappa':>14} {'SGD loss':>14} {'Muon loss':>14} {'Muon+RB loss':>14}")
print("-" * 95)

snapshot_steps = [0, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
for step_idx in snapshot_steps:
    parts = [f"{step_idx:>6}"]
    for method in ['sgd', 'muon', 'muon_rebalance']:
        kappas = []
        for curve in all_results[method]['kappa_curves']:
            if step_idx < len(curve):
                kappas.append(curve[step_idx])
        mean_k = np.mean(kappas) if kappas else float('nan')
        parts.append(f"{mean_k:>14.1f}")

    for method in ['sgd', 'muon', 'muon_rebalance']:
        losses = []
        for curve in all_results[method]['loss_curves']:
            if step_idx < len(curve):
                losses.append(curve[step_idx])
        mean_l = np.mean(losses) if losses else float('nan')
        parts.append(f"{mean_l:>14.6e}")

    print("".join(parts))


# =============================================================================
# CROSSOVER ANALYSIS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("KAPPA CROSSOVER ANALYSIS")
print(f"{'=' * 85}")

# For each seed, find when Muon's kappa exceeds SGD's kappa
# and when Muon+RB's kappa exceeds SGD's kappa
muon_crossover_steps = []
muon_rb_crossover_steps = []

for seed_idx in range(NUM_SEEDS):
    sgd_k = all_results['sgd']['kappa_curves'][seed_idx]
    muon_k = all_results['muon']['kappa_curves'][seed_idx]
    muon_rb_k = all_results['muon_rebalance']['kappa_curves'][seed_idx]

    min_len = min(len(sgd_k), len(muon_k))

    # Find first step where Muon kappa > SGD kappa (sustained)
    muon_cross = None
    for s in range(min_len):
        if muon_k[s] > sgd_k[s]:
            # Check if sustained for 80% of remaining steps
            remaining = min_len - s
            if remaining > 5:
                frac_above = sum(1 for t in range(s, min_len) if muon_k[t] > sgd_k[t]) / remaining
                if frac_above > 0.8:
                    muon_cross = s
                    break
    muon_crossover_steps.append(muon_cross)

    # Same for Muon+RB
    min_len_rb = min(len(sgd_k), len(muon_rb_k))
    muon_rb_cross = None
    for s in range(min_len_rb):
        if muon_rb_k[s] > sgd_k[s]:
            remaining = min_len_rb - s
            if remaining > 5:
                frac_above = sum(1 for t in range(s, min_len_rb) if muon_rb_k[t] > sgd_k[t]) / remaining
                if frac_above > 0.8:
                    muon_rb_cross = s
                    break
    muon_rb_crossover_steps.append(muon_rb_cross)

print(f"\nPer-seed kappa crossover (Muon overtakes SGD):")
print(f"  {'Seed':>6} {'Muon cross':>12} {'Muon+RB cross':>15}")
print("-" * 40)
for seed_idx in range(NUM_SEEDS):
    mc = muon_crossover_steps[seed_idx]
    mrc = muon_rb_crossover_steps[seed_idx]
    print(f"  {seed_idx:>6} {str(mc):>12} {str(mrc):>15}")

# Count how many seeds show no crossover for Muon+RB
n_no_cross_muon = sum(1 for x in muon_crossover_steps if x is None)
n_no_cross_rb = sum(1 for x in muon_rb_crossover_steps if x is None)
print(f"\n  Muon:    {n_no_cross_muon}/{NUM_SEEDS} seeds with NO crossover (kappa stays below SGD)")
print(f"  Muon+RB: {n_no_cross_rb}/{NUM_SEEDS} seeds with NO crossover")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 85}")

mean_final_kappa_sgd = np.mean(all_results['sgd']['final_kappas'])
mean_final_kappa_muon = np.mean(all_results['muon']['final_kappas'])
mean_final_kappa_rb = np.mean(all_results['muon_rebalance']['final_kappas'])

mean_final_loss_sgd = np.mean(all_results['sgd']['final_losses'])
mean_final_loss_muon = np.mean(all_results['muon']['final_losses'])
mean_final_loss_rb = np.mean(all_results['muon_rebalance']['final_losses'])

# T1: Muon+RB has lower final kappa than plain Muon
t1 = mean_final_kappa_rb < mean_final_kappa_muon
print(f"\nT1: Muon+Rebalance final kappa < plain Muon final kappa?")
print(f"    Muon+RB: {mean_final_kappa_rb:.1f}, Muon: {mean_final_kappa_muon:.1f}")
print(f"    --> {'PASS' if t1 else 'FAIL'}")

# T2: Muon+RB prevents kappa crossover (in more seeds than plain Muon)
t2 = n_no_cross_rb >= n_no_cross_muon
print(f"\nT2: Muon+RB prevents kappa crossover in >= as many seeds as Muon?")
print(f"    Muon+RB no-cross: {n_no_cross_rb}/{NUM_SEEDS}, Muon: {n_no_cross_muon}/{NUM_SEEDS}")
print(f"    --> {'PASS' if t2 else 'FAIL'}")

# T3: Muon+RB has better (lower) final loss than plain Muon
t3 = mean_final_loss_rb < mean_final_loss_muon
print(f"\nT3: Muon+Rebalance final loss < plain Muon final loss?")
print(f"    Muon+RB: {mean_final_loss_rb:.6e}, Muon: {mean_final_loss_muon:.6e}")
print(f"    --> {'PASS' if t3 else 'FAIL'}")

# T4: Muon+RB has lower final loss than SGD
t4 = mean_final_loss_rb < mean_final_loss_sgd
print(f"\nT4: Muon+Rebalance final loss < SGD final loss?")
print(f"    Muon+RB: {mean_final_loss_rb:.6e}, SGD: {mean_final_loss_sgd:.6e}")
print(f"    --> {'PASS' if t4 else 'FAIL'}")


# =============================================================================
# FINAL VERDICT
# =============================================================================

total_pass = sum([t1, t2, t3, t4])

print(f"\n\n{'=' * 85}")
print("FINAL VERDICT: PRODUCT-SPACE MUON")
print(f"{'=' * 85}")

print(f"""
  Tests passed: {total_pass}/4

  Product rebalance every {REBALANCE_EVERY} steps:
    - Final kappa: {mean_final_kappa_rb:.1f} (vs Muon {mean_final_kappa_muon:.1f}, SGD {mean_final_kappa_sgd:.1f})
    - Final loss:  {mean_final_loss_rb:.6e} (vs Muon {mean_final_loss_muon:.6e}, SGD {mean_final_loss_sgd:.6e})
    - Crossover prevention: {n_no_cross_rb}/{NUM_SEEDS} seeds (vs Muon {n_no_cross_muon}/{NUM_SEEDS})
""")

if total_pass >= 3:
    print("  VERDICT: CONFIRMED -- Product rebalancing controls kappa and improves convergence.")
    print("  The product-space SVD redistribution prevents catastrophic conditioning")
    print(f"  at depth {NUM_LAYERS}.")
elif total_pass >= 2:
    print("  VERDICT: PARTIALLY CONFIRMED -- Some benefit from rebalancing.")
else:
    print("  VERDICT: INCONCLUSIVE -- Product rebalancing did not clearly help.")
    if mean_final_kappa_muon < mean_final_kappa_sgd:
        print("  Note: Plain Muon already controls kappa well at this depth.")

print("=" * 85)
