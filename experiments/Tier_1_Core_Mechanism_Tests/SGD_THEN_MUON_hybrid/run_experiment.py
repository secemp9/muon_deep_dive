#!/usr/bin/env python3
"""
Experiment 3.13: SGD->Muon Hybrid Fine-tuning
===============================================

FROM 2.5: SGD is faster to half-loss (6.8 steps) but Muon gets better final
quality. Hybrid: SGD for first S steps, then Muon for the rest.

PROTOCOL:
  - 4-layer deep linear, 32x32
  - Fine-tuning from pre-trained checkpoint (same setup as 2.5)
  - Pre-train with SGD for 500 steps on W_target_original
  - Modify 20% of target -> W_target_modified
  - Fine-tune 200 total steps with hybrid: SGD for S steps, then Muon
  - Sweep S in {0, 5, 10, 20, 50, 100} plus pure-SGD (S=200) and pure-Muon (S=0)
  - 5 seeds for robustness

MEASUREMENTS:
  - Final loss after 200 steps
  - Steps to reach 50% of initial loss (convergence speed)
  - Find S that gets both fast early convergence AND best final loss

PREDICTIONS:
  - S=0 (pure Muon) has best final loss but slower early
  - S=200 (pure SGD) has fastest early but worse final
  - Optimal S is somewhere in between (maybe S=5-20)
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

# Pre-training
PRETRAIN_STEPS = 500
PRETRAIN_LR = 0.01

# Fine-tuning
FINETUNE_STEPS = 200
SGD_FT_LR = 0.01
MUON_FT_LR = 0.005

# Muon parameters
MOMENTUM = 0.9
NS_ITERS = 5

# Target modification
MODIFY_FRAC = 0.20

# Sweep
SWITCH_POINTS = [0, 5, 10, 20, 50, 100, 200]  # S=0 means pure Muon, S=200 means pure SGD

NUM_SEEDS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed data
X_data = np.random.randn(DIM, BATCH_SIZE) * 0.3
W_target_original = np.random.randn(DIM, DIM) * 0.5


# =============================================================================
# NETWORK HELPERS
# =============================================================================

def init_weights(num_layers, rng):
    weights = []
    for _ in range(num_layers):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W.copy())
    return weights


def copy_weights(weights):
    return [W.copy() for W in weights]


def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


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


def make_modified_target(W_original, frac, rng):
    W_mod = W_original.copy()
    n_entries = W_mod.size
    n_change = int(frac * n_entries)
    indices = rng.choice(n_entries, size=n_change, replace=False)
    flat = W_mod.ravel()
    flat[indices] = rng.randn(n_change) * 0.5
    return W_mod


# =============================================================================
# NEWTON-SCHULZ ORTHOGONALIZATION
# =============================================================================

def newton_schulz_ortho(M, n_iters=NS_ITERS):
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


# =============================================================================
# HYBRID TRAINING
# =============================================================================

def train_hybrid(weights_init, W_target, X, n_steps, switch_step):
    """
    Train for n_steps total:
      - Steps 0..switch_step-1: SGD with momentum
      - Steps switch_step..n_steps-1: Muon with momentum

    Returns loss trajectory.
    """
    weights = copy_weights(weights_init)
    Y_target = W_target @ X
    velocities = [np.zeros_like(w) for w in weights]

    losses = []

    for step in range(n_steps):
        loss = compute_loss(weights, X, Y_target)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            losses.extend([loss] * (n_steps - step - 1))
            break

        grads = compute_gradients(weights, X, Y_target)

        if step < switch_step:
            # SGD phase
            for i in range(len(weights)):
                velocities[i] = MOMENTUM * velocities[i] + grads[i]
                weights[i] = weights[i] - SGD_FT_LR * velocities[i]
        else:
            if step == switch_step and switch_step > 0:
                # Reset momentum when switching to Muon
                velocities = [np.zeros_like(w) for w in weights]

            # Muon phase
            for i in range(len(weights)):
                ortho_grad = newton_schulz_ortho(grads[i])
                velocities[i] = MOMENTUM * velocities[i] + ortho_grad
                weights[i] = weights[i] - MUON_FT_LR * velocities[i]

    # Final loss
    final_loss = compute_loss(weights, X, Y_target)
    losses.append(final_loss)

    return np.array(losses)


def steps_to_threshold(losses, frac=0.5):
    """Steps to reach frac of initial loss."""
    if len(losses) == 0:
        return len(losses)
    threshold = losses[0] * frac
    for i, l in enumerate(losses):
        if l <= threshold:
            return i
    return len(losses)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

print("=" * 85)
print("Experiment 3.13: SGD->Muon Hybrid Fine-tuning")
print("=" * 85)
print(f"Setup: {NUM_LAYERS}-layer deep linear ({DIM}x{DIM})")
print(f"Pre-train: {PRETRAIN_STEPS} steps SGD, then modify {int(MODIFY_FRAC*100)}% of target")
print(f"Fine-tune: {FINETUNE_STEPS} steps, SGD lr={SGD_FT_LR}, Muon lr={MUON_FT_LR}")
print(f"Switch points S: {SWITCH_POINTS}")
print(f"Seeds: {NUM_SEEDS}")
print("=" * 85)

# Collect results
all_results = {S: {'final_losses': [], 'half_steps': [], 'loss_curves': []}
               for S in SWITCH_POINTS}

for seed_idx in range(NUM_SEEDS):
    run_seed = SEED + seed_idx * 137
    rng = np.random.RandomState(run_seed)

    print(f"\n--- Seed {run_seed} ---")

    # Pre-train
    weights_init = init_weights(NUM_LAYERS, rng)
    Y_target_orig = W_target_original @ X_data
    weights = copy_weights(weights_init)
    velocities = [np.zeros_like(w) for w in weights]

    for step in range(PRETRAIN_STEPS):
        grads = compute_gradients(weights, X_data, Y_target_orig)
        for i in range(NUM_LAYERS):
            velocities[i] = MOMENTUM * velocities[i] + grads[i]
            weights[i] = weights[i] - PRETRAIN_LR * velocities[i]

    checkpoint = copy_weights(weights)
    pretrain_loss = compute_loss(checkpoint, X_data, Y_target_orig)
    print(f"  Pre-train final loss: {pretrain_loss:.6f}")

    # Modified target
    W_target_mod = make_modified_target(W_target_original, MODIFY_FRAC, rng)

    # Run each switch point
    for S in SWITCH_POINTS:
        loss_curve = train_hybrid(checkpoint, W_target_mod, X_data, FINETUNE_STEPS, S)
        final_loss = loss_curve[-1]
        half_step = steps_to_threshold(loss_curve)

        all_results[S]['final_losses'].append(final_loss)
        all_results[S]['half_steps'].append(half_step)
        all_results[S]['loss_curves'].append(loss_curve)

    # Print this seed's results
    final_str = "  Finals: " + ", ".join([f"S={S}:{all_results[S]['final_losses'][-1]:.6f}" for S in SWITCH_POINTS])
    print(final_str)


# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("AGGREGATE RESULTS")
print(f"{'=' * 85}")

print(f"\n{'Switch S':<12} {'Description':<25} {'Mean Final Loss':>16} {'Std':>10} {'Mean Half-Steps':>16}")
print("-" * 82)

for S in SWITCH_POINTS:
    if S == 0:
        desc = "Pure Muon"
    elif S == FINETUNE_STEPS:
        desc = "Pure SGD"
    else:
        desc = f"SGD({S})->Muon({FINETUNE_STEPS-S})"

    mean_final = np.mean(all_results[S]['final_losses'])
    std_final = np.std(all_results[S]['final_losses'])
    mean_half = np.mean(all_results[S]['half_steps'])

    print(f"{S:<12} {desc:<25} {mean_final:>16.6f} {std_final:>10.6f} {mean_half:>16.1f}")


# =============================================================================
# LOSS CURVE SNAPSHOTS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("LOSS CURVES (averaged, selected steps)")
print(f"{'=' * 85}")

header_parts = [f"{'Step':>6}"]
for S in SWITCH_POINTS:
    if S == 0:
        label = "Muon"
    elif S == FINETUNE_STEPS:
        label = "SGD"
    else:
        label = f"S={S}"
    header_parts.append(f"{label:>12}")
print("  ".join(header_parts))
print("-" * (8 + 14 * len(SWITCH_POINTS)))

snapshot_steps = [0, 5, 10, 20, 50, 100, 150, 200]
for step_idx in snapshot_steps:
    parts = [f"{step_idx:>6}"]
    for S in SWITCH_POINTS:
        curves = all_results[S]['loss_curves']
        if step_idx < len(curves[0]):
            mean_loss = np.mean([c[step_idx] for c in curves])
            parts.append(f"{mean_loss:>12.6f}")
        else:
            parts.append(f"{'':>12}")
    print("  ".join(parts))


# =============================================================================
# FIND OPTIMAL SWITCH POINT
# =============================================================================

print(f"\n\n{'=' * 85}")
print("OPTIMAL SWITCH POINT ANALYSIS")
print(f"{'=' * 85}")

# Find best S by final loss
mean_finals = {S: np.mean(all_results[S]['final_losses']) for S in SWITCH_POINTS}
best_S = min(mean_finals, key=mean_finals.get)
pure_muon_loss = mean_finals[0]
pure_sgd_loss = mean_finals[FINETUNE_STEPS]

print(f"\n  Pure Muon (S=0) final loss:      {pure_muon_loss:.6f}")
print(f"  Pure SGD (S={FINETUNE_STEPS}) final loss:    {pure_sgd_loss:.6f}")
print(f"  Best hybrid (S={best_S}) final loss: {mean_finals[best_S]:.6f}")

# Is the best hybrid better than both?
better_than_muon = mean_finals[best_S] < pure_muon_loss
better_than_sgd = mean_finals[best_S] < pure_sgd_loss
print(f"\n  Best hybrid beats pure Muon? {better_than_muon}")
print(f"  Best hybrid beats pure SGD?  {better_than_sgd}")

# Convergence speed comparison
mean_half_steps = {S: np.mean(all_results[S]['half_steps']) for S in SWITCH_POINTS}
fastest_S = min(mean_half_steps, key=mean_half_steps.get)
print(f"\n  Fastest to 50% loss: S={fastest_S} ({mean_half_steps[fastest_S]:.1f} steps)")
print(f"  Pure Muon speed:     {mean_half_steps[0]:.1f} steps")
print(f"  Pure SGD speed:      {mean_half_steps[FINETUNE_STEPS]:.1f} steps")

# Combined metric: rank by final loss, break ties by speed
print(f"\n  Ranking by final loss:")
sorted_S = sorted(SWITCH_POINTS, key=lambda s: mean_finals[s])
for rank, S in enumerate(sorted_S, 1):
    marker = " <-- BEST" if S == best_S else ""
    if S == 0:
        desc = "Pure Muon"
    elif S == FINETUNE_STEPS:
        desc = "Pure SGD"
    else:
        desc = f"Hybrid S={S}"
    print(f"    #{rank}: {desc:<25} loss={mean_finals[S]:.6f}  speed={mean_half_steps[S]:.1f} steps{marker}")


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

print(f"\n\n{'=' * 85}")
print("HYPOTHESIS TESTS")
print(f"{'=' * 85}")

# T1: Pure SGD converges faster (fewer steps to half-loss)
t1 = mean_half_steps[FINETUNE_STEPS] < mean_half_steps[0]
print(f"\nT1: Pure SGD faster to half-loss than pure Muon?")
print(f"    SGD: {mean_half_steps[FINETUNE_STEPS]:.1f} steps, Muon: {mean_half_steps[0]:.1f} steps")
print(f"    --> {'PASS' if t1 else 'FAIL'}")

# T2: Pure Muon gets better final loss
t2 = pure_muon_loss < pure_sgd_loss
print(f"\nT2: Pure Muon gets better final loss than pure SGD?")
print(f"    Muon: {pure_muon_loss:.6f}, SGD: {pure_sgd_loss:.6f}")
print(f"    --> {'PASS' if t2 else 'FAIL'}")

# T3: Some hybrid S (not 0 or 200) is competitive with or beats pure Muon
hybrid_only = {S: mean_finals[S] for S in SWITCH_POINTS if S > 0 and S < FINETUNE_STEPS}
best_hybrid_S = min(hybrid_only, key=hybrid_only.get)
best_hybrid_loss = hybrid_only[best_hybrid_S]
t3 = best_hybrid_loss <= pure_muon_loss * 1.05  # within 5% of Muon
print(f"\nT3: Best hybrid within 5% of pure Muon?")
print(f"    Best hybrid S={best_hybrid_S}: {best_hybrid_loss:.6f}")
print(f"    Pure Muon: {pure_muon_loss:.6f}")
print(f"    Ratio: {best_hybrid_loss / (pure_muon_loss + 1e-12):.4f}")
print(f"    --> {'PASS' if t3 else 'FAIL'}")

# T4: Best hybrid is faster than pure Muon
t4 = mean_half_steps[best_hybrid_S] < mean_half_steps[0]
print(f"\nT4: Best hybrid faster than pure Muon?")
print(f"    Hybrid S={best_hybrid_S}: {mean_half_steps[best_hybrid_S]:.1f} steps")
print(f"    Pure Muon: {mean_half_steps[0]:.1f} steps")
print(f"    --> {'PASS' if t4 else 'FAIL'}")


# =============================================================================
# FINAL VERDICT
# =============================================================================

print(f"\n\n{'=' * 85}")
print("FINAL VERDICT: SGD->MUON HYBRID FINE-TUNING")
print(f"{'=' * 85}")

total_pass = sum([t1, t2, t3, t4])
print(f"""
  Tests passed: {total_pass}/4

  Best switch point: S={best_S}
  Best hybrid (non-pure) switch: S={best_hybrid_S}

  The strategy: SGD for {best_hybrid_S} steps (fast descent),
  then switch to Muon for the remaining {FINETUNE_STEPS - best_hybrid_S} steps (quality refinement).
""")

if total_pass >= 3:
    print("  VERDICT: CONFIRMED -- Hybrid strategy is effective.")
    print(f"  SGD handles fast early descent, Muon handles quality refinement.")
    print(f"  Optimal switch point: S={best_hybrid_S}")
elif total_pass >= 2:
    print("  VERDICT: PARTIALLY CONFIRMED")
else:
    print("  VERDICT: INCONCLUSIVE -- pure strategies may be sufficient.")

print("=" * 85)
