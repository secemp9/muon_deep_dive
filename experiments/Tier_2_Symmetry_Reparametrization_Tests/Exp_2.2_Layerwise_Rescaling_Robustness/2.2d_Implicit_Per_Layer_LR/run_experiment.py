#!/usr/bin/env python3
"""
2.2d: Muon as Implicit Per-Layer LR -- Oracle Per-Layer LR SGD vs Muon
========================================================================

MOTIVATION (from 2.2a/b):
  Muon's Newton-Schulz normalizes each layer's momentum by ||M||_F,
  effectively giving per-layer LR adaptation. Is that ALL Muon does?

QUESTION: Does SGD with ORACLE per-layer LR (optimal per layer) match
  Muon at c=100 rescaling? If yes, Muon is just implicit per-layer LR
  adaptation. If no, Muon provides directional value beyond LR scaling.

PROTOCOL:
  4-layer deep linear 32x32 with c=100 rescaling:
    layer 0 *= 100, layer 3 *= 0.01
  300 steps. Compare:
    (a) SGD with per-layer LR: grid of 5 LRs per layer = 5^4 = 625
        combos, pick best.
    (b) Muon with single LR (sweep 7 values).
    (c) SGD with single LR (sweep 7 values).

KEY TESTS:
  T1: Oracle-LR SGD matches Muon within 2x?
      If yes -> Muon is just implicit per-layer LR.
      If no  -> Muon provides directional value beyond LR scaling.

Setup: 4-layer, 32x32, 300 steps, c=100, 5 seeds.
"""

import numpy as np
import os
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM_BETA = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
C = 100

# 7 candidates for single-LR methods
SGD_LR_CANDIDATES = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
MUON_LR_CANDIDATES = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003]

# 5 candidates per layer for oracle grid search (5^4 = 625 combos)
PER_LAYER_LR_CANDIDATES = [0.1, 0.01, 0.001, 0.0001, 0.00001]


# =============================================================================
# NETWORK
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


def init_weights(seed, c):
    rng = np.random.RandomState(seed)
    weights = []
    for l in range(N_LAYERS):
        W = np.eye(DIM) + rng.randn(DIM, DIM) * 0.1
        weights.append(W)
    # Apply c-rescaling: layer 0 *= c, layer -1 *= 1/c
    weights[0] = weights[0] * c
    weights[-1] = weights[-1] / c
    return weights


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def compute_gradients(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads


def make_data(seed):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


# =============================================================================
# TRAINING METHODS
# =============================================================================

def train_sgd(weights_init, X, Y, lr):
    """SGD with momentum, single global LR."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + grads[l]
            weights[l] = weights[l] - lr * mom[l]
    return compute_loss(weights, X, Y)


def train_muon(weights_init, X, Y, lr):
    """Muon: orthogonalize gradient via Newton-Schulz, then momentum."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + newton_schulz(grads[l])
            weights[l] = weights[l] - lr * mom[l]
    return compute_loss(weights, X, Y)


def train_sgd_per_layer(weights_init, X, Y, per_layer_lrs):
    """SGD with momentum, independent LR per layer."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + grads[l]
            weights[l] = weights[l] - per_layer_lrs[l] * mom[l]
    return compute_loss(weights, X, Y)


# =============================================================================
# LR SWEEP HELPERS
# =============================================================================

def sweep_single_lr(train_fn, seeds, candidates):
    """Sweep over LR candidates using first 3 seeds. Return best LR."""
    best_lr, best_loss = candidates[-1], float('inf')
    for lr in candidates:
        losses = []
        for s in seeds[:3]:
            X, Y = make_data(s)
            w = init_weights(s + 5000, C)
            fl = train_fn(w, X, Y, lr)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        ml = np.mean(finite) if finite else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_lr = lr
    return best_lr, best_loss


def sweep_oracle_per_layer(seeds):
    """
    Grid search: 5 LRs per layer, 5^4 = 625 combos.
    Use first 3 seeds for selection. Return best per-layer LR tuple.
    """
    combos = list(itertools.product(PER_LAYER_LR_CANDIDATES, repeat=N_LAYERS))
    print(f"    Oracle grid search: {len(combos)} combos...")

    best_combo = None
    best_loss = float('inf')

    for idx, combo in enumerate(combos):
        lrs = list(combo)
        losses = []
        for s in seeds[:3]:
            X, Y = make_data(s)
            w = init_weights(s + 5000, C)
            fl = train_sgd_per_layer(w, X, Y, lrs)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        ml = np.mean(finite) if finite else float('inf')
        if ml < best_loss:
            best_loss = ml
            best_combo = lrs

        if (idx + 1) % 100 == 0:
            print(f"      {idx+1}/{len(combos)} combos evaluated, best so far: {best_loss:.6e}")

    return best_combo, best_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print(f"2.2d: MUON AS IMPLICIT PER-LAYER LR (c={C})")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear, {DIM}x{DIM}, {NUM_STEPS} steps")
    print(f"Rescaling: layer 0 *= {C}, layer {N_LAYERS-1} *= {1.0/C}")
    print(f"SGD LR candidates:  {SGD_LR_CANDIDATES}")
    print(f"Muon LR candidates: {MUON_LR_CANDIDATES}")
    print(f"Per-layer LR grid:  {PER_LAYER_LR_CANDIDATES} (5^4 = {len(PER_LAYER_LR_CANDIDATES)**N_LAYERS} combos)")
    print()

    # Phase 1: LR sweeps
    print("Phase 1: LR sweeps...")

    print("  SGD single LR sweep...")
    sgd_lr, sgd_sweep_loss = sweep_single_lr(train_sgd, seeds, SGD_LR_CANDIDATES)
    print(f"    Best: lr={sgd_lr}, sweep loss={sgd_sweep_loss:.6e}")

    print("  Muon single LR sweep...")
    muon_lr, muon_sweep_loss = sweep_single_lr(train_muon, seeds, MUON_LR_CANDIDATES)
    print(f"    Best: lr={muon_lr}, sweep loss={muon_sweep_loss:.6e}")

    print("  SGD oracle per-layer LR sweep...")
    oracle_lrs, oracle_sweep_loss = sweep_oracle_per_layer(seeds)
    print(f"    Best per-layer LRs: {oracle_lrs}")
    print(f"    Sweep loss: {oracle_sweep_loss:.6e}")

    # Phase 2: Full training with all seeds
    print(f"\nPhase 2: Full training ({NUM_SEEDS} seeds)...")

    results = {}
    for name, desc in [('sgd', 'SGD (single LR)'),
                       ('muon', 'Muon (single LR)'),
                       ('sgd_oracle', 'SGD (oracle per-layer LR)')]:
        losses = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(s + 5000, C)
            if name == 'sgd':
                fl = train_sgd(w, X, Y, sgd_lr)
            elif name == 'muon':
                fl = train_muon(w, X, Y, muon_lr)
            elif name == 'sgd_oracle':
                fl = train_sgd_per_layer(w, X, Y, oracle_lrs)
            losses.append(fl)
        finite = [l for l in losses if np.isfinite(l)]
        mean_loss = np.mean(finite) if finite else float('inf')
        std_loss = np.std(finite) if len(finite) > 1 else 0.0
        finite_frac = len(finite) / len(losses) * 100
        results[name] = {
            'mean': mean_loss,
            'std': std_loss,
            'finite_frac': finite_frac,
            'losses': losses,
        }
        print(f"  {desc:>30}: loss={mean_loss:.6e} +/- {std_loss:.6e}  (finite={finite_frac:.0f}%)")

    # ==========================================================================
    # RESULTS TABLE
    # ==========================================================================

    muon_loss = results['muon']['mean']
    sgd_loss = results['sgd']['mean']
    oracle_loss = results['sgd_oracle']['mean']

    print(f"\n\n{'=' * 100}")
    print("RESULTS")
    print(f"{'=' * 100}")

    print(f"\n  {'Method':>30}  {'Loss':>14}  {'Std':>14}  {'vs Muon':>10}  {'Finite%':>8}  {'LR(s)':>20}")
    print("  " + "-" * 100)
    print(f"  {'SGD (single LR)':>30}  {sgd_loss:>14.6e}  {results['sgd']['std']:>14.6e}  "
          f"{sgd_loss/max(muon_loss,1e-30):>10.2f}x  {results['sgd']['finite_frac']:>7.0f}%  lr={sgd_lr}")
    print(f"  {'Muon (single LR)':>30}  {muon_loss:>14.6e}  {results['muon']['std']:>14.6e}  "
          f"{'1.00x':>10}  {results['muon']['finite_frac']:>7.0f}%  lr={muon_lr}")
    print(f"  {'SGD (oracle per-layer)':>30}  {oracle_loss:>14.6e}  {results['sgd_oracle']['std']:>14.6e}  "
          f"{oracle_loss/max(muon_loss,1e-30):>10.2f}x  {results['sgd_oracle']['finite_frac']:>7.0f}%  "
          f"lrs={[f'{l:.1e}' for l in oracle_lrs]}")

    # Show gradient norms at init to explain why per-layer LR is needed
    print(f"\n  Gradient norms at init (c={C} rescaling):")
    X, Y = make_data(seeds[0])
    w = init_weights(seeds[0] + 5000, C)
    grads = compute_gradients(w, X, Y)
    for l in range(N_LAYERS):
        g_norm = np.linalg.norm(grads[l], 'fro')
        print(f"    Layer {l}: ||G||_F = {g_norm:.4e}  (oracle LR = {oracle_lrs[l]:.1e})")

    # ==========================================================================
    # HYPOTHESIS TESTS
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    # T1: Oracle per-layer SGD matches Muon within 2x?
    ratio_oracle = oracle_loss / max(muon_loss, 1e-30)
    t1 = ratio_oracle < 2.0
    print(f"\n  T1: SGD with ORACLE per-layer LR matches Muon within 2x?")
    print(f"      Oracle SGD loss:  {oracle_loss:.6e}")
    print(f"      Muon loss:        {muon_loss:.6e}")
    print(f"      Ratio:            {ratio_oracle:.2f}x")
    if t1:
        print(f"      --> PASS: Muon's advantage IS explained by implicit per-layer LR.")
        print(f"         With perfect per-layer tuning, SGD matches Muon.")
    else:
        print(f"      --> FAIL: Muon provides DIRECTIONAL value beyond LR scaling.")
        print(f"         Even with oracle per-layer LR, SGD cannot match Muon.")

    # How much does single-LR SGD suffer?
    ratio_sgd = sgd_loss / max(muon_loss, 1e-30)
    print(f"\n  Context: SGD (single LR) vs Muon = {ratio_sgd:.1f}x")
    print(f"  The per-layer LR reduces the gap from {ratio_sgd:.1f}x to {ratio_oracle:.2f}x")
    if ratio_sgd > 1 and ratio_oracle > 1:
        pct_explained = (1 - (ratio_oracle - 1) / (ratio_sgd - 1)) * 100 if ratio_sgd > 1 else 0
        print(f"  Per-layer LR explains {pct_explained:.0f}% of Muon's advantage")
    elif ratio_oracle <= 1:
        print(f"  Per-layer LR FULLY explains Muon's advantage")

    # ==========================================================================
    # CONCLUSION
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("CONCLUSION")
    print(f"{'=' * 100}")

    if t1:
        print(f"\n  Muon's robustness to c={C} rescaling IS primarily per-layer LR adaptation.")
        print(f"  SGD with oracle per-layer LR achieves {ratio_oracle:.2f}x Muon's loss.")
        print(f"  The Newton-Schulz normalization acts as an automatic per-layer LR tuner.")
    else:
        print(f"\n  Muon provides value BEYOND per-layer LR adaptation.")
        print(f"  Even with oracle per-layer LR (625 combos searched), SGD achieves")
        print(f"  only {ratio_oracle:.2f}x Muon's loss. The directional quality of the")
        print(f"  polar factor (SV equalization) provides additional convergence benefit")
        print(f"  that cannot be replicated by LR tuning alone.")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")

    # ==========================================================================
    # PLOT
    # ==========================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'2.2d: Muon as Implicit Per-Layer LR (c={C})\n'
                     f'{N_LAYERS}-layer deep linear {DIM}x{DIM}, {NUM_STEPS} steps',
                     fontsize=13, fontweight='bold')

        # (a) Loss comparison bar chart
        ax = axes[0]
        methods = ['SGD\n(single LR)', 'SGD\n(oracle per-layer)', 'Muon\n(single LR)']
        losses_plot = [sgd_loss, oracle_loss, muon_loss]
        stds_plot = [results['sgd']['std'], results['sgd_oracle']['std'], results['muon']['std']]
        colors = ['#4477AA', '#FF8800', '#CC3311']
        bars = ax.bar(range(3), losses_plot, yerr=stds_plot, color=colors,
                      edgecolor='black', capsize=5, width=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel('Final Loss')
        ax.set_yscale('log')
        ax.set_title(f'Final Loss (c={C} rescaling)')
        ax.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{losses_plot[i]:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # (b) Gradient norm vs oracle LR per layer
        ax = axes[1]
        g_norms = [np.linalg.norm(grads[l], 'fro') for l in range(N_LAYERS)]
        ax2 = ax.twinx()
        x_pos = range(N_LAYERS)
        ax.bar([x - 0.15 for x in x_pos], g_norms, width=0.3, color='#4477AA',
               edgecolor='black', label='||G||_F', alpha=0.7)
        ax2.bar([x + 0.15 for x in x_pos], oracle_lrs, width=0.3, color='#CC3311',
                edgecolor='black', label='Oracle LR', alpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Gradient Frobenius Norm', color='#4477AA')
        ax2.set_ylabel('Oracle Learning Rate', color='#CC3311')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Layer {l}' for l in range(N_LAYERS)])
        ax.set_yscale('log')
        ax2.set_yscale('log')
        ax.set_title('Gradient Norms vs Oracle LR')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(SCRIPT_DIR, '2_2d_implicit_per_layer_lr.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {plot_path}")
    except ImportError:
        print("\nWARNING: matplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
