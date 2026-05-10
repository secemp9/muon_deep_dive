#!/usr/bin/env python3
"""
H17: GRADIENT ANISOTROPY PREDICTS MUON BENEFIT
================================================

TEST: Across 5 architectures, does gradient anisotropy at init predict
      how much Muon outperforms SGD?

ARCHITECTURES:
  (a) Deep linear 4-layer 32x32
  (b) ReLU MLP 4-layer 32
  (c) Tanh MLP 4-layer 32
  (d) Bottleneck 32->8->32->8->32
  (e) Wide 4-layer 128x128

PROTOCOL:
  For each architecture:
    1. Measure gradient anisotropy at init: sigma_1/sigma_min of first gradient
    2. Train with SGD and Muon at optimal LR (sweep for both), 300 steps
    3. Compute advantage = loss_SGD / loss_Muon

  Plot advantage vs anisotropy. Compute Spearman correlation.

PREDICTION: r > 0.5 (higher anisotropy = more Muon benefit because more
  SVs to equalize).

Setup: 300 steps, 5 seeds per architecture, LR swept independently.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

ARCH_NAMES = ['deep_linear', 'relu_mlp', 'tanh_mlp', 'bottleneck', 'wide']

LR_SGD_CANDIDATES = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]
LR_MUON_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]


# =============================================================================
# ARCHITECTURE DEFINITIONS
# =============================================================================

def get_arch_config(arch):
    """Return (dims_list, activation) for each architecture.
    dims_list gives the weight matrix dimensions: each entry is (rows, cols).
    """
    if arch == 'deep_linear':
        # 4 layers, 32x32 each
        dims = [(32, 32)] * 4
        return dims, 'linear'
    elif arch == 'relu_mlp':
        # 4 layers, 32x32 each, ReLU activations
        dims = [(32, 32)] * 4
        return dims, 'relu'
    elif arch == 'tanh_mlp':
        # 4 layers, 32x32 each, tanh activations
        dims = [(32, 32)] * 4
        return dims, 'tanh'
    elif arch == 'bottleneck':
        # 32->8->32->8->32
        dims = [(8, 32), (32, 8), (8, 32), (32, 8)]
        return dims, 'relu'
    elif arch == 'wide':
        # 4 layers, 128x128 each
        dims = [(128, 128)] * 4
        return dims, 'linear'
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def init_weights(arch, seed):
    rng = np.random.RandomState(seed)
    dims, _ = get_arch_config(arch)
    weights = []
    for (rows, cols) in dims:
        # Xavier-like initialization
        scale = np.sqrt(2.0 / (rows + cols))
        W = rng.randn(rows, cols) * scale
        # Add identity-like component for square matrices
        if rows == cols:
            W += np.eye(rows) * 0.5
        weights.append(W)
    return weights


def apply_activation(x, act, is_last_layer):
    if is_last_layer or act == 'linear':
        return x
    elif act == 'relu':
        return np.maximum(0, x)
    elif act == 'tanh':
        return np.tanh(x)


def act_deriv(pre, act, is_last_layer):
    if is_last_layer or act == 'linear':
        return np.ones_like(pre)
    elif act == 'relu':
        return (pre > 0).astype(float)
    elif act == 'tanh':
        return 1.0 - np.tanh(pre)**2


def forward(weights, X, arch):
    _, act = get_arch_config(arch)
    L = len(weights)
    out = X.copy()
    for idx, W in enumerate(weights):
        out = W @ out
        out = apply_activation(out, act, idx == L - 1)
    return out


def compute_loss(weights, X, Y, arch):
    pred = forward(weights, X, arch)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def compute_gradients(weights, X, Y, arch):
    _, act = get_arch_config(arch)
    L = len(weights)
    N = X.shape[1]

    # Forward pass storing pre and post activations
    post_acts = [X.copy()]
    pre_acts = []
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre)
        out = apply_activation(pre, act, idx == L - 1)
        post_acts.append(out)

    # Backward pass
    delta = (post_acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ post_acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * act_deriv(pre_acts[l-1], act, l-1 == L-1)
    return grads


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# TRAINING
# =============================================================================

def train(weights_init, X, Y, lr, optimizer, arch):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y, arch)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y, arch)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y, arch)


def make_data(arch, seed):
    rng = np.random.RandomState(seed)
    dims, _ = get_arch_config(arch)
    input_dim = dims[0][1]   # cols of first weight = input dim
    output_dim = dims[-1][0]  # rows of last weight = output dim
    X = rng.randn(input_dim, BATCH_SIZE) * 0.3
    Y = rng.randn(output_dim, BATCH_SIZE) * 0.3
    return X, Y


# =============================================================================
# ANISOTROPY MEASUREMENT
# =============================================================================

def measure_anisotropy(arch, seeds):
    """
    Measure gradient anisotropy at init: sigma_1/sigma_min of first gradient.
    Average across seeds.
    """
    anisotropies = []
    for s in seeds:
        X, Y = make_data(arch, s)
        w = init_weights(arch, s + 5000)
        grads = compute_gradients(w, X, Y, arch)
        # Use first gradient (layer 0) for anisotropy
        G = grads[0]
        sv = np.linalg.svd(G, compute_uv=False)
        if sv[-1] > 1e-15:
            aniso = sv[0] / sv[-1]
        else:
            aniso = sv[0] / 1e-15
        anisotropies.append(aniso)
    return np.mean(anisotropies), np.std(anisotropies)


# =============================================================================
# SPEARMAN CORRELATION (no scipy dependency)
# =============================================================================

def spearman_correlation(x, y):
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return float('nan')

    def rank_data(arr):
        order = np.argsort(arr)
        ranks = np.empty(len(arr), dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        # Handle ties by averaging ranks
        for i in range(len(arr)):
            tied = np.where(arr == arr[i])[0]
            if len(tied) > 1:
                avg_rank = np.mean(ranks[tied])
                ranks[tied] = avg_rank
        return ranks

    rx = rank_data(np.array(x, dtype=float))
    ry = rank_data(np.array(y, dtype=float))
    # Pearson on ranks
    return np.corrcoef(rx, ry)[0, 1]


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H17: GRADIENT ANISOTROPY PREDICTS MUON BENEFIT")
    print("=" * 100)
    print(f"Architectures: {ARCH_NAMES}")
    print(f"Steps: {NUM_STEPS}, Seeds: {NUM_SEEDS}")
    print()

    arch_results = {}

    for arch in ARCH_NAMES:
        dims, act = get_arch_config(arch)
        print(f"\n{'=' * 80}")
        print(f"  ARCHITECTURE: {arch}")
        print(f"  Layers: {dims}, Activation: {act}")
        print(f"{'=' * 80}")

        # Step 1: Measure anisotropy at init
        aniso_mean, aniso_std = measure_anisotropy(arch, seeds)
        print(f"  Gradient anisotropy at init: {aniso_mean:.2f} +/- {aniso_std:.2f}")

        # Step 2: LR sweep for both optimizers
        print(f"  LR sweep...")
        best = {}
        for opt, candidates in [('sgd', LR_SGD_CANDIDATES), ('muon', LR_MUON_CANDIDATES)]:
            best_lr, best_loss = candidates[-1], float('inf')
            for lr in candidates:
                losses = []
                for s in seeds[:3]:
                    X, Y = make_data(arch, s)
                    w = init_weights(arch, s + 5000)
                    fl = train(w, X, Y, lr, opt, arch)
                    losses.append(fl)
                finite = [l for l in losses if np.isfinite(l)]
                ml = np.mean(finite) if finite else float('inf')
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr
            best[opt] = best_lr
            print(f"    {opt:>5}: best_lr={best_lr}")

        # Step 3: Full training with all seeds
        final_losses = {}
        for opt in ['sgd', 'muon']:
            losses = []
            for s in seeds:
                X, Y = make_data(arch, s)
                w = init_weights(arch, s + 5000)
                fl = train(w, X, Y, best[opt], opt, arch)
                losses.append(fl)
            finite = [l for l in losses if np.isfinite(l)]
            mean_loss = np.mean(finite) if finite else float('inf')
            std_loss = np.std(finite) if len(finite) > 1 else 0.0
            final_losses[opt] = {'mean': mean_loss, 'std': std_loss, 'lr': best[opt]}

        advantage = final_losses['sgd']['mean'] / max(final_losses['muon']['mean'], 1e-30)
        print(f"  SGD loss:  {final_losses['sgd']['mean']:.6e} (lr={final_losses['sgd']['lr']})")
        print(f"  Muon loss: {final_losses['muon']['mean']:.6e} (lr={final_losses['muon']['lr']})")
        print(f"  Muon advantage: {advantage:.2f}x")

        arch_results[arch] = {
            'anisotropy_mean': aniso_mean,
            'anisotropy_std': aniso_std,
            'sgd_loss': final_losses['sgd']['mean'],
            'muon_loss': final_losses['muon']['mean'],
            'advantage': advantage,
            'sgd_lr': final_losses['sgd']['lr'],
            'muon_lr': final_losses['muon']['lr'],
        }

    # ==========================================================================
    # CORRELATION ANALYSIS
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("RESULTS: Gradient Anisotropy vs Muon Advantage")
    print(f"{'=' * 100}")

    print(f"\n  {'Architecture':>15}  {'Anisotropy':>12}  {'SGD Loss':>14}  {'Muon Loss':>14}  {'Advantage':>12}")
    print("  " + "-" * 72)

    anisos = []
    advs = []
    for arch in ARCH_NAMES:
        r = arch_results[arch]
        anisos.append(r['anisotropy_mean'])
        advs.append(r['advantage'])
        print(f"  {arch:>15}  {r['anisotropy_mean']:>12.2f}  {r['sgd_loss']:>14.6e}  "
              f"{r['muon_loss']:>14.6e}  {r['advantage']:>12.2f}x")

    # Pearson correlation
    pearson_r = np.corrcoef(anisos, advs)[0, 1]
    # Spearman correlation
    spearman_r = spearman_correlation(anisos, advs)

    print(f"\n  Pearson correlation:  r = {pearson_r:.3f}")
    print(f"  Spearman correlation: r = {spearman_r:.3f}")

    # Log-scale correlations (anisotropy can span orders of magnitude)
    log_anisos = np.log10(np.array(anisos))
    log_advs = np.log10(np.clip(np.array(advs), 1e-10, None))
    pearson_log = np.corrcoef(log_anisos, log_advs)[0, 1]
    spearman_log = spearman_correlation(log_anisos, log_advs)
    print(f"  Pearson (log-log):    r = {pearson_log:.3f}")
    print(f"  Spearman (log-log):   r = {spearman_log:.3f}")

    # ==========================================================================
    # HYPOTHESIS TESTS
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    t1 = spearman_r > 0.5
    print(f"\n  T1: Spearman correlation r > 0.5?")
    print(f"      Spearman r = {spearman_r:.3f}")
    if t1:
        print(f"      --> PASS: Higher gradient anisotropy DOES predict greater Muon benefit.")
    else:
        print(f"      --> FAIL: Anisotropy is NOT a reliable predictor of Muon benefit.")

    # T2: Rank ordering matches
    aniso_ranking = np.argsort(anisos)[::-1]  # highest aniso first
    adv_ranking = np.argsort(advs)[::-1]      # highest advantage first
    top_aniso_arch = ARCH_NAMES[aniso_ranking[0]]
    top_adv_arch = ARCH_NAMES[adv_ranking[0]]
    t2 = top_aniso_arch == top_adv_arch
    print(f"\n  T2: Highest anisotropy = highest Muon advantage?")
    print(f"      Highest anisotropy: {top_aniso_arch} ({anisos[aniso_ranking[0]]:.2f})")
    print(f"      Highest advantage:  {top_adv_arch} ({advs[adv_ranking[0]]:.2f}x)")
    print(f"      Ranking by anisotropy: {[ARCH_NAMES[i] for i in aniso_ranking]}")
    print(f"      Ranking by advantage:  {[ARCH_NAMES[i] for i in adv_ranking]}")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    # T3: Low anisotropy => low advantage
    low_aniso_idx = aniso_ranking[-1]
    low_aniso_adv = advs[low_aniso_idx]
    median_adv = np.median(advs)
    t3 = low_aniso_adv < median_adv
    print(f"\n  T3: Lowest anisotropy has below-median Muon advantage?")
    print(f"      Lowest anisotropy: {ARCH_NAMES[low_aniso_idx]} (aniso={anisos[low_aniso_idx]:.2f}, adv={low_aniso_adv:.2f}x)")
    print(f"      Median advantage: {median_adv:.2f}x")
    print(f"      --> {'PASS' if t3 else 'FAIL'}")

    # ==========================================================================
    # CONCLUSION
    # ==========================================================================

    print(f"\n\n{'=' * 100}")
    print("CONCLUSION")
    print(f"{'=' * 100}")

    tests_passed = sum([t1, t2, t3])
    print(f"\n  Tests passed: {tests_passed}/3")
    print(f"  Spearman r = {spearman_r:.3f}")

    if t1:
        print(f"\n  CONFIRMED: Gradient anisotropy at initialization positively correlates")
        print(f"  with Muon's advantage (Spearman r = {spearman_r:.3f} > 0.5).")
        print(f"  Architectures with more anisotropic gradients benefit more from Muon's")
        print(f"  SV equalization, because there are more imbalanced SVs to correct.")
    else:
        print(f"\n  NOT CONFIRMED: Gradient anisotropy alone does not reliably predict")
        print(f"  Muon's advantage (Spearman r = {spearman_r:.3f} <= 0.5).")
        print(f"  Other factors (e.g., activation landscape, gradient alignment) may")
        print(f"  matter more than raw spectral anisotropy.")

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
        fig.suptitle('H17: Gradient Anisotropy Predicts Muon Benefit\n'
                     f'{NUM_STEPS} steps, {NUM_SEEDS} seeds per architecture',
                     fontsize=13, fontweight='bold')

        colors_map = {
            'deep_linear': '#4477AA',
            'relu_mlp': '#CC3311',
            'tanh_mlp': '#228B22',
            'bottleneck': '#9933CC',
            'wide': '#FF8800',
        }

        # (a) Anisotropy vs Advantage (linear scale)
        ax = axes[0]
        for i, arch in enumerate(ARCH_NAMES):
            ax.scatter(anisos[i], advs[i], c=colors_map[arch], s=120,
                       edgecolors='black', zorder=3, label=arch)
            ax.annotate(arch, (anisos[i], advs[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_xlabel('Gradient Anisotropy (sigma_1/sigma_min)')
        ax.set_ylabel('Muon Advantage (loss_SGD / loss_Muon)')
        ax.set_title(f'Linear Scale (Spearman r={spearman_r:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Fit line
        if len(anisos) > 1:
            z = np.polyfit(anisos, advs, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(anisos), max(anisos), 50)
            ax.plot(x_fit, p(x_fit), '--', color='gray', alpha=0.5, linewidth=1)

        # (b) Log-log scale
        ax = axes[1]
        for i, arch in enumerate(ARCH_NAMES):
            ax.scatter(anisos[i], advs[i], c=colors_map[arch], s=120,
                       edgecolors='black', zorder=3, label=arch)
            ax.annotate(arch, (anisos[i], advs[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_xlabel('Gradient Anisotropy (sigma_1/sigma_min)')
        ax.set_ylabel('Muon Advantage (loss_SGD / loss_Muon)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Log-Log Scale (Spearman r={spearman_log:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(SCRIPT_DIR, 'h17_anisotropy_predicts_benefit.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {plot_path}")
    except ImportError:
        print("\nWARNING: matplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
