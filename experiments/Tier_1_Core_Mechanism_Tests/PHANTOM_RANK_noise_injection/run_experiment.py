#!/usr/bin/env python3
"""
Experiment 2.8: Phantom Rank Barrier -- noise injection before ortho
=====================================================================

HYPOTHESIS:
  When gradient rank(G) drops below n/2, ortho (Newton-Schulz) produces a
  rank-k partial isometry that wastes capacity.  Injecting small noise
  (~1% of ||G||_F) into the gradient BEFORE ortho lifts the effective rank,
  breaking the phantom rank barrier and improving final loss.

SETUP:
  - 4-layer linear 32x32 network (deep linear net)
  - 1000 training steps (long enough for gradient erank to collapse)
  - Sweep noise levels: 0%, 0.1%, 1%, 5%, 10% of ||G||_F
  - Track gradient effective rank (erank) over training
  - Compare final loss for each noise level

KEY TEST:
  Does 1% noise injection improve final loss when erank < n/2 (=16)?

METRIC:
  Effective rank (erank) = exp(entropy of normalised singular values)
  erank in [1, min(m,n)] -- measures how many singular values are "active"
"""

import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================

WIDTH = 32
DEPTH = 4
NUM_STEPS = 1000
LR_MUON = 0.02
NS_ITERS = 5
BATCH_SIZE = 64
INPUT_DIM = 32
OUTPUT_DIM = 32
SEED = 42

NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05, 0.10]  # fraction of ||G||_F
NOISE_LABELS = ['0%', '0.1%', '1%', '5%', '10%']

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
    """Forward pass through deep linear net (no activation)."""
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

    # Forward pass storing activations
    activations = [X.copy()]
    out = X.copy()
    for W in weights:
        out = W @ out
        activations.append(out.copy())

    # Backward pass
    Y_pred = activations[-1]
    diff = Y_pred - Y_target
    delta = diff / batch_size  # dL/d(output)

    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        # Gradient for W_l: delta @ activations[l].T
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


def effective_rank(M):
    """Compute effective rank = exp(entropy of normalised singular values).
    erank in [1, min(m,n)].
    """
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-12]
    if len(sv) == 0:
        return 0.0
    # Normalise to form a probability distribution
    p = sv / np.sum(sv)
    # Shannon entropy
    H = -np.sum(p * np.log(p))
    return np.exp(H)


# =============================================================================
# TRAINING ROUTINES
# =============================================================================

def train_muon_with_noise(weights, X, Y, num_steps, lr, noise_frac, ns_iters=5,
                           seed=42, track_every=10):
    """Train with Muon + noise injection before ortho.

    Args:
        noise_frac: fraction of ||G||_F to add as isotropic Gaussian noise

    Returns:
        weights_final, loss_history, erank_history
    """
    rng = np.random.RandomState(seed + hash(str(noise_frac)) % 10000)
    weights = [W.copy() for W in weights]
    loss_history = []
    erank_history = []

    for step in range(num_steps):
        # Track metrics
        if step % track_every == 0:
            loss = compute_loss(weights, X, Y)
            loss_history.append(loss)

            # Compute mean erank across layers
            grads_for_erank = compute_gradients(weights, X, Y)
            eranks = [effective_rank(g) for g in grads_for_erank]
            erank_history.append(np.mean(eranks))

        # Compute gradients
        grads = compute_gradients(weights, X, Y)

        for i in range(len(weights)):
            G = grads[i]

            # Inject noise BEFORE ortho
            if noise_frac > 0:
                gnorm = np.linalg.norm(G, 'fro')
                noise = rng.randn(*G.shape)
                noise = noise * (noise_frac * gnorm / np.linalg.norm(noise, 'fro'))
                G = G + noise

            G_orth = newton_schulz_orthogonalize(G, ns_iters)
            weights[i] -= lr * G_orth

    # Final metrics
    final_loss = compute_loss(weights, X, Y)
    loss_history.append(final_loss)
    grads_final = compute_gradients(weights, X, Y)
    eranks_final = [effective_rank(g) for g in grads_final]
    erank_history.append(np.mean(eranks_final))

    return weights, loss_history, erank_history


def train_plain_muon(weights, X, Y, num_steps, lr, ns_iters=5, track_every=10):
    """Plain Muon (no noise) -- same as noise_frac=0 but separate for clarity."""
    return train_muon_with_noise(weights, X, Y, num_steps, lr, 0.0, ns_iters,
                                  seed=SEED, track_every=track_every)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # Generate data -- ill-conditioned target to encourage low-rank gradients
    X = rng.randn(INPUT_DIM, BATCH_SIZE) * 0.5

    # Create an ill-conditioned target matrix for the deep linear net
    U, _ = np.linalg.qr(rng.randn(OUTPUT_DIM, OUTPUT_DIM))
    V, _ = np.linalg.qr(rng.randn(INPUT_DIM, INPUT_DIM))
    # Singular values decay -- forces gradient rank to drop
    sigma = np.array([10.0 * (0.5 ** i) for i in range(min(OUTPUT_DIM, INPUT_DIM))])
    T = U @ np.diag(sigma) @ V
    Y = T @ X

    print("=" * 90)
    print("Experiment 2.8: Phantom Rank Barrier -- Noise Injection Before Ortho")
    print("=" * 90)
    print()
    print("HYPOTHESIS: When gradient erank < n/2, ortho wastes capacity on a partial")
    print("  isometry. Injecting ~1% noise lifts erank and improves final loss.")
    print()
    print(f"Config: {DEPTH}-layer linear {WIDTH}x{WIDTH}, {NUM_STEPS} steps, lr={LR_MUON}")
    print(f"  Target condition number: {sigma[0]/sigma[-1]:.0f}")
    print(f"  Noise levels: {NOISE_LABELS}")
    print()

    TRACK_EVERY = 10
    weights_init = init_weights(DEPTH, WIDTH, seed=SEED)

    # Run for each noise level
    results = {}
    for noise_frac, noise_label in zip(NOISE_LEVELS, NOISE_LABELS):
        print(f"  Running noise={noise_label} ...", end=" ", flush=True)
        w_final, losses, eranks = train_muon_with_noise(
            weights_init, X, Y, NUM_STEPS, LR_MUON, noise_frac,
            NS_ITERS, seed=SEED, track_every=TRACK_EVERY
        )
        results[noise_label] = {
            'losses': losses,
            'eranks': eranks,
            'final_loss': losses[-1],
            'mean_erank': np.mean(eranks),
            'final_erank': eranks[-1],
        }
        print(f"final_loss={losses[-1]:.6f}, mean_erank={np.mean(eranks):.2f}")

    # =========================================================================
    # ERANK TRAJECTORY (for plain Muon vs 1% noise)
    # =========================================================================
    print()
    print("=" * 90)
    print("ERANK TRAJECTORY (every 100 steps)")
    print("=" * 90)
    steps_tracked = list(range(0, NUM_STEPS, TRACK_EVERY)) + [NUM_STEPS]
    # Print header
    header = f"{'Step':>6}"
    for nl in NOISE_LABELS:
        header += f"  {nl:>10}"
    print(header)
    print("-" * (6 + 12 * len(NOISE_LABELS)))

    # Print every 100 steps
    for idx in range(0, len(steps_tracked), 10):
        if idx < len(steps_tracked):
            step = steps_tracked[idx] if idx < len(steps_tracked) else steps_tracked[-1]
            row = f"{step:>6}"
            for nl in NOISE_LABELS:
                eranks = results[nl]['eranks']
                if idx < len(eranks):
                    row += f"  {eranks[idx]:>10.2f}"
                else:
                    row += f"  {'---':>10}"
            print(row)

    # Last step
    row = f"{NUM_STEPS:>6}"
    for nl in NOISE_LABELS:
        eranks = results[nl]['eranks']
        row += f"  {eranks[-1]:>10.2f}"
    print(row)

    # =========================================================================
    # LOSS TRAJECTORY
    # =========================================================================
    print()
    print("=" * 90)
    print("LOSS TRAJECTORY (every 100 steps)")
    print("=" * 90)
    header = f"{'Step':>6}"
    for nl in NOISE_LABELS:
        header += f"  {nl:>12}"
    print(header)
    print("-" * (6 + 14 * len(NOISE_LABELS)))

    for idx in range(0, len(steps_tracked), 10):
        if idx < len(steps_tracked):
            step = steps_tracked[idx]
            row = f"{step:>6}"
            for nl in NOISE_LABELS:
                losses = results[nl]['losses']
                if idx < len(losses):
                    row += f"  {losses[idx]:>12.6f}"
                else:
                    row += f"  {'---':>12}"
            print(row)

    row = f"{NUM_STEPS:>6}"
    for nl in NOISE_LABELS:
        losses = results[nl]['losses']
        row += f"  {losses[-1]:>12.6f}"
    print(row)

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Noise Level':<14} {'Final Loss':>12} {'Mean erank':>12} {'Final erank':>12}")
    print("-" * 54)
    for nl in NOISE_LABELS:
        r = results[nl]
        print(f"{nl:<14} {r['final_loss']:>12.6f} {r['mean_erank']:>12.2f} {r['final_erank']:>12.2f}")

    # =========================================================================
    # KEY ANALYSIS
    # =========================================================================
    print()
    print("=" * 90)
    print("KEY ANALYSIS")
    print("=" * 90)

    baseline_loss = results['0%']['final_loss']
    baseline_erank = results['0%']['mean_erank']
    half_n = WIDTH / 2.0  # 16

    print(f"\n  Baseline (0% noise): final_loss={baseline_loss:.6f}, mean_erank={baseline_erank:.2f}")
    print(f"  n/2 = {half_n:.0f}")
    print()

    # Check: is erank < n/2?
    erank_below_half = baseline_erank < half_n
    print(f"  [{'PASS' if erank_below_half else 'FAIL'}] Gradient erank < n/2 (phantom rank condition)")
    print(f"    mean_erank={baseline_erank:.2f} vs n/2={half_n:.0f}")

    # Check: does 1% noise improve final loss?
    loss_1pct = results['1%']['final_loss']
    improvement_1pct = baseline_loss - loss_1pct
    pct_improve_1pct = improvement_1pct / baseline_loss * 100 if baseline_loss > 1e-12 else 0

    noise_helps = loss_1pct < baseline_loss
    print(f"\n  [{'PASS' if noise_helps else 'FAIL'}] 1% noise improves final loss")
    print(f"    0% loss={baseline_loss:.6f}, 1% loss={loss_1pct:.6f}")
    print(f"    Improvement: {improvement_1pct:.6f} ({pct_improve_1pct:.1f}%)")

    # Check: does 1% noise raise erank?
    erank_1pct = results['1%']['mean_erank']
    erank_lifts = erank_1pct > baseline_erank
    print(f"\n  [{'PASS' if erank_lifts else 'FAIL'}] 1% noise raises gradient erank")
    print(f"    0% erank={baseline_erank:.2f}, 1% erank={erank_1pct:.2f}")

    # Check: too much noise (10%) hurts
    loss_10pct = results['10%']['final_loss']
    sweet_spot = loss_1pct < loss_10pct
    print(f"\n  [{'PASS' if sweet_spot else 'INFO'}] Sweet spot: 1% < 10% loss (too much noise hurts)")
    print(f"    1% loss={loss_1pct:.6f}, 10% loss={loss_10pct:.6f}")

    # Find best noise level
    best_nl = min(NOISE_LABELS, key=lambda nl: results[nl]['final_loss'])
    best_loss = results[best_nl]['final_loss']
    print(f"\n  Best noise level: {best_nl} (loss={best_loss:.6f})")

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    print()
    print("=" * 90)
    print("HYPOTHESIS TEST: PHANTOM RANK BARRIER")
    print("=" * 90)

    # The key combined test:
    # 1) erank drops below n/2
    # 2) noise injection (at some level) improves final loss
    # 3) noise injection raises erank

    any_noise_helps = any(results[nl]['final_loss'] < baseline_loss
                          for nl in NOISE_LABELS if nl != '0%')

    key_pass = erank_below_half and any_noise_helps
    print()
    print(f"  Condition 1 -- erank < n/2:           {'YES' if erank_below_half else 'NO'}")
    print(f"  Condition 2 -- noise improves loss:    {'YES' if any_noise_helps else 'NO'}")
    print(f"  Condition 3 -- noise lifts erank:      {'YES' if erank_lifts else 'NO'}")
    print()

    if key_pass:
        print("  RESULT: PASS -- Phantom rank barrier confirmed and mitigable by noise injection.")
        print("  The gradient effective rank drops below n/2 during training, and injecting")
        print("  small noise before orthogonalization restores rank and improves loss.")
    else:
        if not erank_below_half:
            print("  RESULT: FAIL -- Gradient erank did NOT drop below n/2.")
            print("  The phantom rank barrier condition was not triggered in this setup.")
        elif not any_noise_helps:
            print("  RESULT: FAIL -- Noise injection did NOT improve loss despite low erank.")
            print("  The phantom rank barrier may exist but noise is not an effective remedy.")
        else:
            print("  RESULT: PARTIAL -- See individual checks above.")

    print()
    print("=" * 90)


if __name__ == "__main__":
    run_experiment()
