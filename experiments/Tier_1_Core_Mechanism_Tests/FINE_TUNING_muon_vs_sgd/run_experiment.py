#!/usr/bin/env python3
"""
Exp 2.5: Muon worse for fine-tuning -- chaos prevents settling near pre-trained minimum
========================================================================================

HYPOTHESIS:
  Muon's higher Lyapunov exponent means it is better for exploration (training
  from scratch) but worse for exploitation (fine-tuning from a checkpoint).
  The Newton-Schulz orthogonalization injects gauge-direction chaos that
  destabilises the basin around a pre-trained minimum, whereas SGD can gently
  slide into the nearest good minimum.

PROTOCOL:
  Phase 1 -- Pre-training (shared):
      Train a 4-layer deep linear net (32x32) from random init with SGD
      for 500 steps on target W_target.  Save checkpoint.

  Phase 2 -- Fine-tuning (from checkpoint):
      Modify 20% of the target matrix -> W_target_modified.
      From the checkpoint, fine-tune 200 steps with:
        (a) SGD   lr=0.01
        (b) Muon  lr=0.005

  Phase 3 -- From-scratch comparison:
      Train from random init for 700 steps on W_target_modified with both
      optimizers.

MEASUREMENTS:
  - Fine-tuning loss curves  (SGD vs Muon)
  - From-scratch loss curves  (SGD vs Muon)
  - Distance from checkpoint  ||W_t - W_checkpoint||_F  for fine-tuning runs

PREDICTIONS:
  - Fine-tuning: SGD stays closer to checkpoint, Muon wanders further
  - Fine-tuning final loss: SGD < Muon (SGD exploits the nearby minimum)
  - From-scratch final loss: Muon < SGD (Muon explores better)
"""

import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
np.random.seed(SEED)

DIM = 32
NUM_LAYERS = 4
BATCH_SIZE = 64

# Phase 1 -- pre-training
PRETRAIN_STEPS = 500
PRETRAIN_LR = 0.01

# Phase 2 -- fine-tuning
FINETUNE_STEPS = 200
SGD_FT_LR = 0.01
MUON_FT_LR = 0.005

# Phase 3 -- from scratch
SCRATCH_STEPS = 700
SGD_SCRATCH_LR = 0.01
MUON_SCRATCH_LR = 0.005

# Muon parameters
MOMENTUM = 0.9
NS_ITERS = 5

# Target modification fraction
MODIFY_FRAC = 0.20

# Number of runs to average over for robustness
NUM_RUNS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed data
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3

# =============================================================================
# TARGET MATRICES
# =============================================================================

W_target_original = np.random.randn(DIM, DIM) * 0.5

def make_modified_target(W_original, frac, rng):
    """Change `frac` of entries to new random values."""
    W_mod = W_original.copy()
    n_entries = W_mod.size
    n_change = int(frac * n_entries)
    indices = rng.choice(n_entries, size=n_change, replace=False)
    flat = W_mod.ravel()
    flat[indices] = rng.randn(n_change) * 0.5
    return W_mod

# =============================================================================
# NETWORK HELPERS
# =============================================================================

def init_weights(num_layers, rng):
    """Initialize layers near identity for stability."""
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


def forward_linear(weights, X):
    """Forward pass through deep linear net."""
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y_target):
    """Quadratic loss: 0.5 * ||f(X) - Y||^2 / N."""
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(np.sum(diff**2, axis=0))


def compute_gradients(weights, X, Y_target):
    """Backprop through deep linear net for quadratic loss."""
    num_layers = len(weights)
    N = X.shape[1]

    # Forward pass -- store activations
    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])

    # Output error
    delta = (activations[-1] - Y_target) / N

    # Backward pass
    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta

    return grads


def flatten_weights(weights):
    return np.concatenate([W.ravel() for W in weights])


def checkpoint_distance(weights, checkpoint_weights):
    """Frobenius distance between current weights and checkpoint."""
    flat_w = flatten_weights(weights)
    flat_c = flatten_weights(checkpoint_weights)
    return np.linalg.norm(flat_w - flat_c)


def copy_weights(weights):
    return [W.copy() for W in weights]

# =============================================================================
# OPTIMIZERS
# =============================================================================

def newton_schulz_ortho(M, n_iters=NS_ITERS):
    """Newton-Schulz iteration to approximate the orthogonal polar factor."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (np.linalg.norm(M, ord='fro') + 1e-7)
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    Id = np.eye(X.shape[0])
    for _ in range(n_iters):
        A = X @ X.T
        X = (a * Id + b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X


class SGDOptimizer:
    def __init__(self, weights, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(W) for W in weights]

    def step(self, weights, grads):
        for i in range(len(weights)):
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i]
            weights[i] -= self.lr * self.velocity[i]
        return weights


class MuonOptimizer:
    def __init__(self, weights, lr, momentum=0.9, ns_iters=NS_ITERS):
        self.lr = lr
        self.momentum = momentum
        self.ns_iters = ns_iters
        self.velocity = [np.zeros_like(W) for W in weights]

    def step(self, weights, grads):
        for i in range(len(weights)):
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i]
            ortho_update = newton_schulz_ortho(self.velocity[i], self.ns_iters)
            weights[i] -= self.lr * ortho_update
        return weights

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(weights, optimizer, W_target, X, n_steps, checkpoint_weights=None):
    """Train and record loss + distance from checkpoint at each step."""
    losses = []
    distances = []
    Y_target = W_target @ X

    for step in range(n_steps):
        loss = compute_loss(weights, X, Y_target)
        losses.append(loss)

        if checkpoint_weights is not None:
            dist = checkpoint_distance(weights, checkpoint_weights)
            distances.append(dist)

        grads = compute_gradients(weights, X, Y_target)
        weights = optimizer.step(weights, grads)

    # Final loss
    loss = compute_loss(weights, X, Y_target)
    losses.append(loss)
    if checkpoint_weights is not None:
        distances.append(checkpoint_distance(weights, checkpoint_weights))

    return weights, np.array(losses), np.array(distances) if distances else None

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_single_experiment(run_seed):
    """Run one full experiment with a given seed."""
    rng = np.random.RandomState(run_seed)

    # ---- Targets ----
    W_target_mod = make_modified_target(W_target_original, MODIFY_FRAC, rng)

    # ---- Phase 1: Pre-train with SGD on original target ----
    weights_init = init_weights(NUM_LAYERS, rng)
    pretrain_opt = SGDOptimizer(copy_weights(weights_init), lr=PRETRAIN_LR, momentum=MOMENTUM)
    weights_pretrained, pretrain_losses, _ = train(
        copy_weights(weights_init), pretrain_opt, W_target_original, X_data, PRETRAIN_STEPS
    )
    checkpoint = copy_weights(weights_pretrained)

    # ---- Phase 2a: Fine-tune from checkpoint with SGD ----
    ft_sgd_opt = SGDOptimizer(copy_weights(checkpoint), lr=SGD_FT_LR, momentum=MOMENTUM)
    ft_sgd_weights, ft_sgd_losses, ft_sgd_dists = train(
        copy_weights(checkpoint), ft_sgd_opt, W_target_mod, X_data, FINETUNE_STEPS,
        checkpoint_weights=checkpoint
    )

    # ---- Phase 2b: Fine-tune from checkpoint with Muon ----
    ft_muon_opt = MuonOptimizer(copy_weights(checkpoint), lr=MUON_FT_LR, momentum=MOMENTUM)
    ft_muon_weights, ft_muon_losses, ft_muon_dists = train(
        copy_weights(checkpoint), ft_muon_opt, W_target_mod, X_data, FINETUNE_STEPS,
        checkpoint_weights=checkpoint
    )

    # ---- Phase 3a: From scratch with SGD ----
    scratch_init = init_weights(NUM_LAYERS, rng)
    scratch_sgd_opt = SGDOptimizer(copy_weights(scratch_init), lr=SGD_SCRATCH_LR, momentum=MOMENTUM)
    _, scratch_sgd_losses, _ = train(
        copy_weights(scratch_init), scratch_sgd_opt, W_target_mod, X_data, SCRATCH_STEPS
    )

    # ---- Phase 3b: From scratch with Muon ----
    scratch_muon_opt = MuonOptimizer(copy_weights(scratch_init), lr=MUON_SCRATCH_LR, momentum=MOMENTUM)
    _, scratch_muon_losses, _ = train(
        copy_weights(scratch_init), scratch_muon_opt, W_target_mod, X_data, SCRATCH_STEPS
    )

    return {
        'pretrain_final_loss': pretrain_losses[-1],
        'ft_sgd_losses': ft_sgd_losses,
        'ft_muon_losses': ft_muon_losses,
        'ft_sgd_dists': ft_sgd_dists,
        'ft_muon_dists': ft_muon_dists,
        'scratch_sgd_losses': scratch_sgd_losses,
        'scratch_muon_losses': scratch_muon_losses,
    }


def main():
    print("=" * 80)
    print("Exp 2.5: Muon worse for fine-tuning?")
    print("       Chaos prevents settling near pre-trained minimum")
    print("=" * 80)
    print()

    # Collect results across runs
    all_results = []
    for run_idx in range(NUM_RUNS):
        seed = SEED + run_idx * 137
        print(f"  Run {run_idx + 1}/{NUM_RUNS} (seed={seed})...")
        result = run_single_experiment(seed)
        all_results.append(result)

    # Aggregate
    ft_sgd_final_losses   = [r['ft_sgd_losses'][-1] for r in all_results]
    ft_muon_final_losses  = [r['ft_muon_losses'][-1] for r in all_results]
    ft_sgd_final_dists    = [r['ft_sgd_dists'][-1] for r in all_results]
    ft_muon_final_dists   = [r['ft_muon_dists'][-1] for r in all_results]
    sc_sgd_final_losses   = [r['scratch_sgd_losses'][-1] for r in all_results]
    sc_muon_final_losses  = [r['scratch_muon_losses'][-1] for r in all_results]
    pretrain_final_losses = [r['pretrain_final_loss'] for r in all_results]

    # Means and stds
    def ms(arr):
        return np.mean(arr), np.std(arr)

    pt_m, pt_s     = ms(pretrain_final_losses)
    ft_sgd_m, ft_sgd_s   = ms(ft_sgd_final_losses)
    ft_muon_m, ft_muon_s = ms(ft_muon_final_losses)
    ft_sgd_d_m, ft_sgd_d_s   = ms(ft_sgd_final_dists)
    ft_muon_d_m, ft_muon_d_s = ms(ft_muon_final_dists)
    sc_sgd_m, sc_sgd_s   = ms(sc_sgd_final_losses)
    sc_muon_m, sc_muon_s = ms(sc_muon_final_losses)

    # ---- Loss curves (averaged) ----
    print()
    print("-" * 70)
    print("LOSS CURVE SNAPSHOTS (averaged over runs)")
    print("-" * 70)

    # Fine-tuning curves
    ft_steps = len(all_results[0]['ft_sgd_losses'])
    ft_sgd_curve = np.mean([r['ft_sgd_losses'] for r in all_results], axis=0)
    ft_muon_curve = np.mean([r['ft_muon_losses'] for r in all_results], axis=0)

    print("\nFine-tuning from checkpoint (200 steps):")
    print(f"  {'Step':>6}  {'SGD loss':>12}  {'Muon loss':>12}  {'SGD dist':>12}  {'Muon dist':>12}")
    ft_sgd_dist_curve = np.mean([r['ft_sgd_dists'] for r in all_results], axis=0)
    ft_muon_dist_curve = np.mean([r['ft_muon_dists'] for r in all_results], axis=0)
    for step_idx in [0, 10, 25, 50, 100, 150, 200]:
        if step_idx < ft_steps:
            print(f"  {step_idx:>6}  {ft_sgd_curve[step_idx]:>12.6f}  "
                  f"{ft_muon_curve[step_idx]:>12.6f}  "
                  f"{ft_sgd_dist_curve[step_idx]:>12.4f}  "
                  f"{ft_muon_dist_curve[step_idx]:>12.4f}")

    # From-scratch curves
    sc_steps = len(all_results[0]['scratch_sgd_losses'])
    sc_sgd_curve = np.mean([r['scratch_sgd_losses'] for r in all_results], axis=0)
    sc_muon_curve = np.mean([r['scratch_muon_losses'] for r in all_results], axis=0)

    print("\nFrom scratch (700 steps):")
    print(f"  {'Step':>6}  {'SGD loss':>12}  {'Muon loss':>12}")
    for step_idx in [0, 50, 100, 200, 350, 500, 700]:
        if step_idx < sc_steps:
            print(f"  {step_idx:>6}  {sc_sgd_curve[step_idx]:>12.6f}  "
                  f"{sc_muon_curve[step_idx]:>12.6f}")

    # ---- Main results table ----
    print()
    print("=" * 80)
    print("MAIN RESULTS TABLE")
    print("=" * 80)
    print()
    print(f"Pre-training final loss (SGD, 500 steps): {pt_m:.6f} +/- {pt_s:.6f}")
    print()

    header = (f"  {'Scenario':<30}  {'SGD final loss':>16}  {'Muon final loss':>17}  "
              f"{'SGD dist':>14}  {'Muon dist':>14}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    print(f"  {'Fine-tune (from ckpt)':<30}  "
          f"{ft_sgd_m:>10.6f}+/-{ft_sgd_s:<5.4f}  "
          f"{ft_muon_m:>10.6f}+/-{ft_muon_s:<6.4f}  "
          f"{ft_sgd_d_m:>8.4f}+/-{ft_sgd_d_s:<4.3f}  "
          f"{ft_muon_d_m:>8.4f}+/-{ft_muon_d_s:<4.3f}")

    print(f"  {'From scratch (700 steps)':<30}  "
          f"{sc_sgd_m:>10.6f}+/-{sc_sgd_s:<5.4f}  "
          f"{sc_muon_m:>10.6f}+/-{sc_muon_s:<6.4f}  "
          f"{'n/a':>14}  {'n/a':>14}")

    # ---- Verdict ----
    print()
    print("=" * 80)
    print("HYPOTHESIS TESTS")
    print("=" * 80)
    print()

    # Test 1: Fine-tuning distance
    dist_test = ft_muon_d_m > ft_sgd_d_m
    print(f"[H1] Muon wanders further from checkpoint during fine-tuning?")
    print(f"     SGD dist = {ft_sgd_d_m:.4f},  Muon dist = {ft_muon_d_m:.4f}")
    print(f"     Ratio Muon/SGD = {ft_muon_d_m / (ft_sgd_d_m + 1e-12):.2f}x")
    print(f"     --> {'CONFIRMED' if dist_test else 'REJECTED'}")
    print()

    # Test 2: Fine-tuning loss -- SGD < Muon
    ft_loss_test = ft_sgd_m < ft_muon_m
    print(f"[H2] Fine-tuning: SGD reaches lower loss than Muon?")
    print(f"     SGD loss = {ft_sgd_m:.6f},  Muon loss = {ft_muon_m:.6f}")
    if ft_sgd_m > 0:
        print(f"     Muon/SGD ratio = {ft_muon_m / ft_sgd_m:.2f}")
    print(f"     --> {'CONFIRMED' if ft_loss_test else 'REJECTED'}")
    print()

    # Test 3: From-scratch loss -- Muon < SGD
    scratch_loss_test = sc_muon_m < sc_sgd_m
    print(f"[H3] From scratch: Muon reaches lower loss than SGD?")
    print(f"     SGD loss = {sc_sgd_m:.6f},  Muon loss = {sc_muon_m:.6f}")
    if sc_sgd_m > 0:
        print(f"     SGD/Muon ratio = {sc_sgd_m / (sc_muon_m + 1e-12):.2f}")
    print(f"     --> {'CONFIRMED' if scratch_loss_test else 'REJECTED'}")
    print()

    # Overall
    all_confirmed = dist_test and ft_loss_test and scratch_loss_test
    partial = sum([dist_test, ft_loss_test, scratch_loss_test])
    print("=" * 80)
    if all_confirmed:
        print("OVERALL VERDICT: HYPOTHESIS FULLY CONFIRMED (3/3)")
        print("  Muon is worse for fine-tuning (chaos prevents settling)")
        print("  but better from scratch (exploration advantage).")
    elif partial >= 2:
        print(f"OVERALL VERDICT: HYPOTHESIS PARTIALLY CONFIRMED ({partial}/3)")
        if not dist_test:
            print("  Surprise: Muon did NOT wander further from checkpoint.")
        if not ft_loss_test:
            print("  Surprise: Muon actually fine-tuned to LOWER loss than SGD.")
        if not scratch_loss_test:
            print("  Surprise: SGD actually trained from scratch to LOWER loss.")
    else:
        print(f"OVERALL VERDICT: HYPOTHESIS REJECTED ({partial}/3)")
        if not ft_loss_test:
            print("  Key finding: Muon fine-tunes as well or better than SGD.")
        if not scratch_loss_test:
            print("  Key finding: SGD trains from scratch as well or better.")
    print("=" * 80)

    # ---- Additional analysis: convergence speed ----
    print()
    print("-" * 70)
    print("ADDITIONAL ANALYSIS: Convergence Speed")
    print("-" * 70)

    # For fine-tuning: steps to reach 50% of initial loss
    def steps_to_threshold(losses, frac=0.5):
        threshold = losses[0] * frac
        for i, l in enumerate(losses):
            if l <= threshold:
                return i
        return len(losses)

    ft_sgd_half = np.mean([steps_to_threshold(r['ft_sgd_losses']) for r in all_results])
    ft_muon_half = np.mean([steps_to_threshold(r['ft_muon_losses']) for r in all_results])
    sc_sgd_half = np.mean([steps_to_threshold(r['scratch_sgd_losses']) for r in all_results])
    sc_muon_half = np.mean([steps_to_threshold(r['scratch_muon_losses']) for r in all_results])

    print(f"\nSteps to reach 50% of initial loss:")
    print(f"  Fine-tune SGD:      {ft_sgd_half:.1f}")
    print(f"  Fine-tune Muon:     {ft_muon_half:.1f}")
    print(f"  From-scratch SGD:   {sc_sgd_half:.1f}")
    print(f"  From-scratch Muon:  {sc_muon_half:.1f}")

    # ---- Early vs Late fine-tuning comparison ----
    print()
    print("-" * 70)
    print("FINE-TUNING DYNAMICS: Early vs Late")
    print("-" * 70)
    # Compare loss reduction in first 50 steps vs last 50 steps
    for name, curve in [("SGD", ft_sgd_curve), ("Muon", ft_muon_curve)]:
        early_drop = curve[0] - curve[50]
        late_drop = curve[150] - curve[200] if len(curve) > 200 else curve[-51] - curve[-1]
        print(f"  {name}: early drop (0-50) = {early_drop:.6f},  "
              f"late drop (150-200) = {late_drop:.6f},  "
              f"ratio = {early_drop / (abs(late_drop) + 1e-12):.1f}x")

    print()
    print("Experiment complete.")


if __name__ == "__main__":
    main()
