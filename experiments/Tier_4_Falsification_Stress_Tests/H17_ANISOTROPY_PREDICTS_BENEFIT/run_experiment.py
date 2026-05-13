#!/usr/bin/env python3
"""
H17: INITIAL GRADIENT ANISOTROPY AS A TOY PREDICTOR OF MUON ADVANTAGE
========================================================================

This experiment asks a bounded predictor-style question:

    Across five small synthetic architectures, is larger first-layer gradient
    anisotropy at initialization associated with larger Muon-over-SGD benefit
    after a short learning-rate sweep?

Important scope note
--------------------
This is a small toy correlation study, not a significance study and not a
mechanism proof. It uses five hand-designed architectures, synthetic Gaussian
regression data, and learning rates chosen from predefined grids. The resulting
correlations should therefore be interpreted as descriptive evidence only.

Default protocol
----------------
For each architecture:
  1. Measure first-layer gradient anisotropy at initialization via
     sigma_max / sigma_min.
  2. Sweep candidate learning rates for SGD and Muon on a subset of seeds.
  3. Re-run the best candidate LR for each optimizer on all seeds.
  4. Define Muon advantage as mean_loss_SGD / mean_loss_Muon.

The module is import-safe and exposes `run_experiment()` for notebook use while
preserving normal CLI behavior through `main()`.
"""

from pathlib import Path
import time
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_FILENAME = 'h17_anisotropy_predicts_benefit.png'

DEFAULT_NUM_STEPS = 300
DEFAULT_MOMENTUM = 0.9
DEFAULT_NS_ITERS = 5
DEFAULT_NUM_SEEDS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_WEIGHT_SEED_OFFSET = 5000
DEFAULT_LR_SWEEP_NUM_SEEDS = 3
DEFAULT_DIVERGENCE_THRESHOLD = 1e10
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 12345

ARCH_NAMES = ['deep_linear', 'relu_mlp', 'tanh_mlp', 'bottleneck', 'wide']

LR_SGD_CANDIDATES = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]
LR_MUON_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]


# =============================================================================
# CONFIG HELPERS
# =============================================================================


def get_default_seeds(num_seeds=DEFAULT_NUM_SEEDS):
    return [42 + i * 137 for i in range(num_seeds)]


def get_default_config():
    return {
        'num_steps': DEFAULT_NUM_STEPS,
        'momentum': DEFAULT_MOMENTUM,
        'ns_iters': DEFAULT_NS_ITERS,
        'num_seeds': DEFAULT_NUM_SEEDS,
        'batch_size': DEFAULT_BATCH_SIZE,
        'weight_seed_offset': DEFAULT_WEIGHT_SEED_OFFSET,
        'lr_sweep_num_seeds': DEFAULT_LR_SWEEP_NUM_SEEDS,
        'divergence_threshold': DEFAULT_DIVERGENCE_THRESHOLD,
        'bootstrap_samples': DEFAULT_BOOTSTRAP_SAMPLES,
        'bootstrap_seed': DEFAULT_BOOTSTRAP_SEED,
        'arch_names': list(ARCH_NAMES),
        'lr_sgd_candidates': list(LR_SGD_CANDIDATES),
        'lr_muon_candidates': list(LR_MUON_CANDIDATES),
        'seeds': get_default_seeds(DEFAULT_NUM_SEEDS),
    }


# =============================================================================
# ARCHITECTURE DEFINITIONS
# =============================================================================


def get_arch_config(arch):
    """Return (dims_list, activation) for a named architecture."""
    if arch == 'deep_linear':
        dims = [(32, 32)] * 4
        return dims, 'linear'
    if arch == 'relu_mlp':
        dims = [(32, 32)] * 4
        return dims, 'relu'
    if arch == 'tanh_mlp':
        dims = [(32, 32)] * 4
        return dims, 'tanh'
    if arch == 'bottleneck':
        dims = [(8, 32), (32, 8), (8, 32), (32, 8)]
        return dims, 'relu'
    if arch == 'wide':
        dims = [(128, 128)] * 4
        return dims, 'linear'
    raise ValueError(f"Unknown architecture: {arch}")


def init_weights(arch, seed):
    rng = np.random.RandomState(seed)
    dims, _ = get_arch_config(arch)
    weights = []
    for rows, cols in dims:
        scale = np.sqrt(2.0 / (rows + cols))
        W = rng.randn(rows, cols) * scale
        if rows == cols:
            W += np.eye(rows) * 0.5
        weights.append(W)
    return weights


def apply_activation(x, act, is_last_layer):
    if is_last_layer or act == 'linear':
        return x
    if act == 'relu':
        return np.maximum(0, x)
    if act == 'tanh':
        return np.tanh(x)
    raise ValueError(f"Unknown activation: {act}")


def act_deriv(pre, act, is_last_layer):
    if is_last_layer or act == 'linear':
        return np.ones_like(pre)
    if act == 'relu':
        return (pre > 0).astype(float)
    if act == 'tanh':
        return 1.0 - np.tanh(pre) ** 2
    raise ValueError(f"Unknown activation: {act}")


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
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def compute_gradients(weights, X, Y, arch):
    _, act = get_arch_config(arch)
    L = len(weights)
    N = X.shape[1]

    post_acts = [X.copy()]
    pre_acts = []
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre)
        out = apply_activation(pre, act, idx == L - 1)
        post_acts.append(out)

    delta = (post_acts[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ post_acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * act_deriv(pre_acts[l - 1], act, l - 1 == L - 1)
    return grads


def newton_schulz(M, n_iters=DEFAULT_NS_ITERS):
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


def train(
    weights_init,
    X,
    Y,
    lr,
    optimizer,
    arch,
    num_steps=DEFAULT_NUM_STEPS,
    momentum=DEFAULT_MOMENTUM,
    ns_iters=DEFAULT_NS_ITERS,
    divergence_threshold=DEFAULT_DIVERGENCE_THRESHOLD,
):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(num_steps):
        loss = compute_loss(weights, X, Y, arch)
        if not np.isfinite(loss) or loss > divergence_threshold:
            return float('inf')
        grads = compute_gradients(weights, X, Y, arch)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = momentum * mom[i] + newton_schulz(grads[i], n_iters=ns_iters)
            else:
                mom[i] = momentum * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y, arch)


def make_data(arch, seed, batch_size=DEFAULT_BATCH_SIZE):
    rng = np.random.RandomState(seed)
    dims, _ = get_arch_config(arch)
    input_dim = dims[0][1]
    output_dim = dims[-1][0]
    X = rng.randn(input_dim, batch_size) * 0.3
    Y = rng.randn(output_dim, batch_size) * 0.3
    return X, Y


# =============================================================================
# MEASUREMENTS AND SUMMARIES
# =============================================================================


def effective_rank_from_singular_values(singular_values, eps=1e-12):
    total = np.sum(singular_values)
    if total <= eps:
        return 0.0
    p = singular_values / total
    p = p[p > eps]
    if len(p) == 0:
        return 0.0
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def summarize_numeric_list(values):
    finite_values = [float(v) for v in values if np.isfinite(v)]
    if not finite_values:
        return {
            'mean': float('inf'),
            'std': 0.0,
            'finite_count': 0,
            'diverged_count': len(values),
            'finite_values': [],
        }
    return {
        'mean': float(np.mean(finite_values)),
        'std': float(np.std(finite_values)) if len(finite_values) > 1 else 0.0,
        'finite_count': len(finite_values),
        'diverged_count': len(values) - len(finite_values),
        'finite_values': finite_values,
    }


def bootstrap_ratio_of_means(
    numerator_values,
    denominator_values,
    n_boot=DEFAULT_BOOTSTRAP_SAMPLES,
    seed=DEFAULT_BOOTSTRAP_SEED,
):
    num = np.array([v for v in numerator_values if np.isfinite(v)], dtype=float)
    den = np.array([v for v in denominator_values if np.isfinite(v)], dtype=float)
    if len(num) == 0 or len(den) == 0:
        return {
            'low': float('nan'),
            'high': float('nan'),
            'std': float('nan'),
            'median': float('nan'),
            'samples': [],
            'n_boot': int(n_boot),
            'note': 'No finite losses available for bootstrap.',
        }

    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_boot):
        boot_num = rng.choice(num, size=len(num), replace=True)
        boot_den = rng.choice(den, size=len(den), replace=True)
        samples.append(float(np.mean(boot_num) / max(np.mean(boot_den), 1e-30)))

    sample_array = np.array(samples, dtype=float)
    return {
        'low': float(np.percentile(sample_array, 2.5)),
        'high': float(np.percentile(sample_array, 97.5)),
        'std': float(np.std(sample_array)),
        'median': float(np.median(sample_array)),
        'samples': samples,
        'n_boot': int(n_boot),
        'note': 'Bootstrap CI over finite losses only; descriptive uncertainty, not a formal guarantee.',
    }


def measure_anisotropy(arch, seeds, batch_size=DEFAULT_BATCH_SIZE, weight_seed_offset=DEFAULT_WEIGHT_SEED_OFFSET):
    """Measure first-layer gradient anisotropy at initialization for each seed."""
    records = []
    for s in seeds:
        X, Y = make_data(arch, s, batch_size=batch_size)
        w = init_weights(arch, s + weight_seed_offset)
        grads = compute_gradients(w, X, Y, arch)
        G = grads[0]
        sv = np.linalg.svd(G, compute_uv=False)
        sigma_max = float(sv[0])
        sigma_min = float(max(sv[-1], 1e-15))
        anisotropy = sigma_max / sigma_min
        records.append({
            'seed': int(s),
            'anisotropy': float(anisotropy),
            'sigma_max': sigma_max,
            'sigma_min': sigma_min,
            'effective_rank': effective_rank_from_singular_values(sv),
            'singular_values': sv.tolist(),
        })

    anisotropies = [record['anisotropy'] for record in records]
    return {
        'per_seed': records,
        'mean': float(np.mean(anisotropies)),
        'std': float(np.std(anisotropies)) if len(anisotropies) > 1 else 0.0,
    }


def evaluate_lr_sweep(
    arch,
    seeds,
    optimizer,
    candidates,
    batch_size=DEFAULT_BATCH_SIZE,
    num_steps=DEFAULT_NUM_STEPS,
    momentum=DEFAULT_MOMENTUM,
    ns_iters=DEFAULT_NS_ITERS,
    divergence_threshold=DEFAULT_DIVERGENCE_THRESHOLD,
    weight_seed_offset=DEFAULT_WEIGHT_SEED_OFFSET,
):
    sweep_results = []
    best_lr = candidates[-1]
    best_loss = float('inf')

    for lr in candidates:
        per_seed = []
        losses = []
        for s in seeds:
            X, Y = make_data(arch, s, batch_size=batch_size)
            w = init_weights(arch, s + weight_seed_offset)
            final_loss = train(
                w,
                X,
                Y,
                lr,
                optimizer,
                arch,
                num_steps=num_steps,
                momentum=momentum,
                ns_iters=ns_iters,
                divergence_threshold=divergence_threshold,
            )
            per_seed.append({
                'seed': int(s),
                'final_loss': float(final_loss),
                'finite': bool(np.isfinite(final_loss)),
            })
            losses.append(final_loss)

        summary = summarize_numeric_list(losses)
        record = {
            'optimizer': optimizer,
            'lr': float(lr),
            'per_seed': per_seed,
            'mean_finite_loss': summary['mean'],
            'std_finite_loss': summary['std'],
            'finite_count': summary['finite_count'],
            'diverged_count': summary['diverged_count'],
        }
        sweep_results.append(record)

        if record['mean_finite_loss'] < best_loss:
            best_loss = record['mean_finite_loss']
            best_lr = lr

    return {
        'optimizer': optimizer,
        'candidates': [float(lr) for lr in candidates],
        'seed_subset': [int(s) for s in seeds],
        'results': sweep_results,
        'best_lr': float(best_lr),
        'best_mean_finite_loss': float(best_loss),
    }


def evaluate_optimizer(
    arch,
    seeds,
    optimizer,
    lr,
    batch_size=DEFAULT_BATCH_SIZE,
    num_steps=DEFAULT_NUM_STEPS,
    momentum=DEFAULT_MOMENTUM,
    ns_iters=DEFAULT_NS_ITERS,
    divergence_threshold=DEFAULT_DIVERGENCE_THRESHOLD,
    weight_seed_offset=DEFAULT_WEIGHT_SEED_OFFSET,
):
    per_seed = []
    losses = []
    for s in seeds:
        X, Y = make_data(arch, s, batch_size=batch_size)
        w = init_weights(arch, s + weight_seed_offset)
        final_loss = train(
            w,
            X,
            Y,
            lr,
            optimizer,
            arch,
            num_steps=num_steps,
            momentum=momentum,
            ns_iters=ns_iters,
            divergence_threshold=divergence_threshold,
        )
        per_seed.append({
            'seed': int(s),
            'final_loss': float(final_loss),
            'finite': bool(np.isfinite(final_loss)),
        })
        losses.append(final_loss)

    summary = summarize_numeric_list(losses)
    return {
        'optimizer': optimizer,
        'lr': float(lr),
        'per_seed': per_seed,
        'mean': summary['mean'],
        'std': summary['std'],
        'finite_count': summary['finite_count'],
        'diverged_count': summary['diverged_count'],
        'finite_values': summary['finite_values'],
    }


# =============================================================================
# CORRELATION AND TESTS
# =============================================================================


def safe_pearson_correlation(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x, y):
    """Compute Spearman rank correlation coefficient without scipy."""
    n = len(x)
    if n < 3:
        return float('nan')

    def rank_data(arr):
        order = np.argsort(arr)
        ranks = np.empty(len(arr), dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        for i in range(len(arr)):
            tied = np.where(arr == arr[i])[0]
            if len(tied) > 1:
                avg_rank = np.mean(ranks[tied])
                ranks[tied] = avg_rank
        return ranks

    rx = rank_data(np.array(x, dtype=float))
    ry = rank_data(np.array(y, dtype=float))
    return float(np.corrcoef(rx, ry)[0, 1])


def build_architecture_summary(arch, arch_result):
    sgd = arch_result['final_losses']['sgd']
    muon = arch_result['final_losses']['muon']
    aniso = arch_result['anisotropy']
    advantage = arch_result['advantage']
    return {
        'architecture': arch,
        'activation': arch_result['activation'],
        'dims': arch_result['dims'],
        'anisotropy_mean': aniso['mean'],
        'anisotropy_std': aniso['std'],
        'sgd_lr': arch_result['best_lrs']['sgd'],
        'muon_lr': arch_result['best_lrs']['muon'],
        'sgd_loss_mean': sgd['mean'],
        'sgd_loss_std': sgd['std'],
        'sgd_finite_count': sgd['finite_count'],
        'muon_loss_mean': muon['mean'],
        'muon_loss_std': muon['std'],
        'muon_finite_count': muon['finite_count'],
        'advantage_ratio_of_means': advantage['ratio_of_means'],
        'advantage_ci_low': advantage['bootstrap_ci']['low'],
        'advantage_ci_high': advantage['bootstrap_ci']['high'],
    }


def compute_correlation_summary(architecture_summaries):
    anisos = [row['anisotropy_mean'] for row in architecture_summaries]
    advs = [row['advantage_ratio_of_means'] for row in architecture_summaries]
    log_anisos = np.log10(np.clip(np.array(anisos, dtype=float), 1e-30, None))
    log_advs = np.log10(np.clip(np.array(advs, dtype=float), 1e-30, None))

    return {
        'linear': {
            'pearson_r': safe_pearson_correlation(anisos, advs),
            'spearman_r': spearman_correlation(anisos, advs),
        },
        'log10': {
            'pearson_r': safe_pearson_correlation(log_anisos, log_advs),
            'spearman_r': spearman_correlation(log_anisos, log_advs),
        },
        'arrays': {
            'anisotropy_means': [float(v) for v in anisos],
            'advantage_ratio_of_means': [float(v) for v in advs],
            'log10_anisotropy_means': log_anisos.tolist(),
            'log10_advantage_ratio_of_means': log_advs.tolist(),
        },
    }


def compute_hypothesis_tests(architecture_summaries, correlations):
    arch_names = [row['architecture'] for row in architecture_summaries]
    anisos = np.array([row['anisotropy_mean'] for row in architecture_summaries], dtype=float)
    advs = np.array([row['advantage_ratio_of_means'] for row in architecture_summaries], dtype=float)
    spearman_r = correlations['linear']['spearman_r']

    aniso_ranking = np.argsort(anisos)[::-1]
    adv_ranking = np.argsort(advs)[::-1]

    top_aniso_arch = arch_names[int(aniso_ranking[0])]
    top_adv_arch = arch_names[int(adv_ranking[0])]
    low_aniso_idx = int(aniso_ranking[-1])
    median_adv = float(np.median(advs))

    t1_pass = bool(np.isfinite(spearman_r) and spearman_r > 0.5)
    t2_pass = bool(top_aniso_arch == top_adv_arch)
    t3_pass = bool(advs[low_aniso_idx] < median_adv)

    return {
        'T1_spearman_gt_0_5': {
            'passed': t1_pass,
            'threshold': 0.5,
            'spearman_r': float(spearman_r),
            'interpretation': 'Descriptive architecture-level association only.',
        },
        'T2_top_arch_match': {
            'passed': t2_pass,
            'top_anisotropy_architecture': top_aniso_arch,
            'top_advantage_architecture': top_adv_arch,
            'anisotropy_ranking': [arch_names[int(i)] for i in aniso_ranking],
            'advantage_ranking': [arch_names[int(i)] for i in adv_ranking],
        },
        'T3_lowest_anisotropy_below_median_advantage': {
            'passed': t3_pass,
            'lowest_anisotropy_architecture': arch_names[low_aniso_idx],
            'lowest_anisotropy_advantage': float(advs[low_aniso_idx]),
            'median_advantage': median_adv,
        },
    }


def build_calibrated_conclusion(results):
    spearman_r = results['correlations']['linear']['spearman_r']
    spearman_text = f"{spearman_r:.3f}" if np.isfinite(spearman_r) else 'nan'
    tests = results['tests']
    tests_passed = sum(int(v['passed']) for v in tests.values())
    n_arch = len(results.get('architecture_summaries', []))
    sweep_phrase = f"{n_arch}-architecture toy sweep"

    if not np.isfinite(spearman_r):
        headline = (
            f"Within this {sweep_phrase}, the architecture-level monotonic association is not reliably "
            f"estimable from the available points (Spearman r={spearman_text})."
        )
    elif tests['T1_spearman_gt_0_5']['passed']:
        headline = (
            f"Within this {sweep_phrase}, higher initial gradient anisotropy is positively associated with "
            f"larger Muon advantage (Spearman r={spearman_text})."
        )
    else:
        headline = (
            f"Within this {sweep_phrase}, initial gradient anisotropy does not show a strong monotonic "
            f"association with Muon advantage (Spearman r={spearman_text})."
        )

    caveat = (
        "This should be treated as descriptive evidence only: the study uses synthetic data, "
        "predefined LR grids, and only a handful of architecture-level points. It does not by "
        "itself establish statistical significance or the causal mechanism behind Muon's behavior."
    )
    return {
        'tests_passed': tests_passed,
        'headline': headline,
        'caveat': caveat,
    }


# =============================================================================
# PLOTTING
# =============================================================================


def make_summary_plot(results, save_path=None, show=False):
    import matplotlib.pyplot as plt

    architecture_summaries = results['architecture_summaries']
    correlations = results['correlations']
    colors_map = {
        'deep_linear': '#4477AA',
        'relu_mlp': '#CC3311',
        'tanh_mlp': '#228B22',
        'bottleneck': '#9933CC',
        'wide': '#FF8800',
    }

    anisos = np.array([row['anisotropy_mean'] for row in architecture_summaries], dtype=float)
    aniso_stds = np.array([row['anisotropy_std'] for row in architecture_summaries], dtype=float)
    advs = np.array([row['advantage_ratio_of_means'] for row in architecture_summaries], dtype=float)
    adv_ci_low = np.array([row['advantage_ci_low'] for row in architecture_summaries], dtype=float)
    adv_ci_high = np.array([row['advantage_ci_high'] for row in architecture_summaries], dtype=float)

    lower_err = np.clip(advs - adv_ci_low, 0.0, None)
    upper_err = np.clip(adv_ci_high - advs, 0.0, None)
    yerr = np.vstack([lower_err, upper_err])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        'H17: Initial gradient anisotropy as a toy predictor of Muon advantage\n'
        f"{results['config']['num_steps']} steps, {results['config']['num_seeds']} seeds, descriptive only",
        fontsize=13,
        fontweight='bold',
    )

    for ax, use_log in zip(axes, [False, True]):
        for idx, row in enumerate(architecture_summaries):
            arch = row['architecture']
            ax.errorbar(
                anisos[idx],
                advs[idx],
                xerr=aniso_stds[idx],
                yerr=yerr[:, idx:idx + 1],
                fmt='o',
                color=colors_map.get(arch, '#333333'),
                ecolor=colors_map.get(arch, '#333333'),
                mec='black',
                ms=8,
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
            ax.annotate(arch, (anisos[idx], advs[idx]), textcoords='offset points', xytext=(6, 6), fontsize=8)

        if len(anisos) > 1 and np.std(anisos) > 1e-15:
            if use_log:
                x_fit = np.logspace(np.log10(np.min(anisos)), np.log10(np.max(anisos)), 100)
                z = np.polyfit(np.log10(anisos), np.log10(np.clip(advs, 1e-30, None)), 1)
                y_fit = 10 ** (z[1]) * x_fit ** z[0]
            else:
                z = np.polyfit(anisos, advs, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(np.min(anisos), np.max(anisos), 100)
                y_fit = p(x_fit)
            ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, linewidth=1.5, label='descriptive fit')

        ax.set_xlabel('Initial first-layer gradient anisotropy (sigma_max / sigma_min)')
        ax.set_ylabel('Muon advantage (mean SGD loss / mean Muon loss)')
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"Log scale; Spearman r={correlations['log10']['spearman_r']:.3f}")
        else:
            ax.set_title(f"Linear scale; Spearman r={correlations['linear']['spearman_r']:.3f}")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


# =============================================================================
# REPORTING
# =============================================================================


def print_results_report(results):
    config = results['config']
    print('=' * 100)
    print('H17: INITIAL GRADIENT ANISOTROPY AS A TOY PREDICTOR OF MUON ADVANTAGE')
    print('=' * 100)
    print('Scope: descriptive toy correlation study; not a significance or mechanism proof.')
    print(f"Architectures: {config['arch_names']}")
    print(f"Steps: {config['num_steps']}, Seeds: {config['num_seeds']}, LR sweep seeds: {config['lr_sweep_num_seeds']}")
    print(f"Batch size: {config['batch_size']}, Momentum: {config['momentum']}, Newton-Schulz iters: {config['ns_iters']}")
    print()

    for arch in config['arch_names']:
        arch_result = results['architecture_results'][arch]
        summary = arch_result['summary']
        print(f"{'=' * 80}")
        print(f"ARCHITECTURE: {arch}")
        print(f"Layers: {arch_result['dims']}, Activation: {arch_result['activation']}")
        print(f"Initial anisotropy: {summary['anisotropy_mean']:.2f} +/- {summary['anisotropy_std']:.2f}")
        print('LR sweep (mean finite loss, finite/total):')
        for opt in ['sgd', 'muon']:
            sweep = arch_result['lr_sweeps'][opt]
            print(f"  {opt.upper()} best LR among candidates: {arch_result['best_lrs'][opt]}")
            for row in sweep['results']:
                total = row['finite_count'] + row['diverged_count']
                print(
                    f"    lr={row['lr']:<8g} mean_finite_loss={row['mean_finite_loss']:.6e} "
                    f"finite={row['finite_count']}/{total}"
                )
        sgd = arch_result['final_losses']['sgd']
        muon = arch_result['final_losses']['muon']
        advantage = arch_result['advantage']
        print(
            f"Final SGD loss:  {sgd['mean']:.6e} +/- {sgd['std']:.6e} "
            f"(finite {sgd['finite_count']}/{len(sgd['per_seed'])}, lr={sgd['lr']})"
        )
        print(
            f"Final Muon loss: {muon['mean']:.6e} +/- {muon['std']:.6e} "
            f"(finite {muon['finite_count']}/{len(muon['per_seed'])}, lr={muon['lr']})"
        )
        print(
            f"Muon advantage (ratio of means): {advantage['ratio_of_means']:.2f}x "
            f"[bootstrap 95% CI {advantage['bootstrap_ci']['low']:.2f}, {advantage['bootstrap_ci']['high']:.2f}]"
        )
        print()

    print('=' * 100)
    print('ARCHITECTURE-LEVEL SUMMARY')
    print('=' * 100)
    print(
        f"{'Architecture':>15}  {'Aniso mean':>12}  {'Aniso std':>12}  {'SGD loss':>12}  {'Muon loss':>12}  {'Adv.':>8}"
    )
    print('  ' + '-' * 86)
    for row in results['architecture_summaries']:
        print(
            f"{row['architecture']:>15}  {row['anisotropy_mean']:>12.2f}  {row['anisotropy_std']:>12.2f}  "
            f"{row['sgd_loss_mean']:>12.6e}  {row['muon_loss_mean']:>12.6e}  {row['advantage_ratio_of_means']:>8.2f}x"
        )

    print('\nCorrelations (descriptive only):')
    print(f"  Pearson r:          {results['correlations']['linear']['pearson_r']:.3f}")
    print(f"  Spearman r:         {results['correlations']['linear']['spearman_r']:.3f}")
    print(f"  Pearson r (log10):  {results['correlations']['log10']['pearson_r']:.3f}")
    print(f"  Spearman r (log10): {results['correlations']['log10']['spearman_r']:.3f}")

    print('\nHypothesis checks:')
    t1 = results['tests']['T1_spearman_gt_0_5']
    t2 = results['tests']['T2_top_arch_match']
    t3 = results['tests']['T3_lowest_anisotropy_below_median_advantage']
    print(f"  T1 (Spearman > 0.5): {'PASS' if t1['passed'] else 'FAIL'} (r={t1['spearman_r']:.3f})")
    print(
        f"  T2 (top anisotropy arch = top advantage arch): {'PASS' if t2['passed'] else 'FAIL'} "
        f"({t2['top_anisotropy_architecture']} vs {t2['top_advantage_architecture']})"
    )
    print(
        f"  T3 (lowest anisotropy arch below median advantage): {'PASS' if t3['passed'] else 'FAIL'} "
        f"({t3['lowest_anisotropy_architecture']})"
    )

    conclusion = results['conclusion']
    print('\nConclusion:')
    print(f"  {conclusion['headline']}")
    print(f"  {conclusion['caveat']}")
    print(f"\nRuntime: {results['runtime']['elapsed_sec']:.2f} seconds")
    print('=' * 100)


# =============================================================================
# EXPERIMENT DRIVER
# =============================================================================


def run_experiment(
    arch_names=None,
    num_steps=DEFAULT_NUM_STEPS,
    momentum=DEFAULT_MOMENTUM,
    ns_iters=DEFAULT_NS_ITERS,
    num_seeds=DEFAULT_NUM_SEEDS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr_sgd_candidates=None,
    lr_muon_candidates=None,
    lr_sweep_num_seeds=DEFAULT_LR_SWEEP_NUM_SEEDS,
    divergence_threshold=DEFAULT_DIVERGENCE_THRESHOLD,
    weight_seed_offset=DEFAULT_WEIGHT_SEED_OFFSET,
    bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    seeds=None,
    make_plot=False,
    save_plot=False,
    show_plot=False,
    plot_path=None,
    verbose=True,
):
    start_time = time.time()

    arch_names = list(ARCH_NAMES if arch_names is None else arch_names)
    lr_sgd_candidates = list(LR_SGD_CANDIDATES if lr_sgd_candidates is None else lr_sgd_candidates)
    lr_muon_candidates = list(LR_MUON_CANDIDATES if lr_muon_candidates is None else lr_muon_candidates)
    seeds = list(get_default_seeds(num_seeds) if seeds is None else seeds)
    if len(seeds) != num_seeds:
        raise ValueError('Length of `seeds` must match `num_seeds`.')
    if lr_sweep_num_seeds > len(seeds):
        raise ValueError('`lr_sweep_num_seeds` cannot exceed the number of seeds.')

    config = {
        'num_steps': int(num_steps),
        'momentum': float(momentum),
        'ns_iters': int(ns_iters),
        'num_seeds': int(num_seeds),
        'batch_size': int(batch_size),
        'weight_seed_offset': int(weight_seed_offset),
        'lr_sweep_num_seeds': int(lr_sweep_num_seeds),
        'divergence_threshold': float(divergence_threshold),
        'bootstrap_samples': int(bootstrap_samples),
        'bootstrap_seed': int(bootstrap_seed),
        'arch_names': arch_names,
        'lr_sgd_candidates': [float(x) for x in lr_sgd_candidates],
        'lr_muon_candidates': [float(x) for x in lr_muon_candidates],
    }

    architecture_results = {}
    architecture_summaries = []

    if verbose:
        print('=' * 100)
        print('Running H17 as a descriptive toy correlation study.')
        print('=' * 100)

    for arch in arch_names:
        dims, activation = get_arch_config(arch)
        if verbose:
            print(f"\n[H17] Architecture: {arch} | dims={dims} | activation={activation}")

        anisotropy = measure_anisotropy(
            arch,
            seeds,
            batch_size=batch_size,
            weight_seed_offset=weight_seed_offset,
        )
        if verbose:
            print(
                f"  initial anisotropy mean +/- std: "
                f"{anisotropy['mean']:.2f} +/- {anisotropy['std']:.2f}"
            )

        lr_sweeps = {}
        best_lrs = {}
        lr_seed_subset = seeds[:lr_sweep_num_seeds]
        for optimizer, candidates in [('sgd', lr_sgd_candidates), ('muon', lr_muon_candidates)]:
            sweep = evaluate_lr_sweep(
                arch,
                lr_seed_subset,
                optimizer,
                candidates,
                batch_size=batch_size,
                num_steps=num_steps,
                momentum=momentum,
                ns_iters=ns_iters,
                divergence_threshold=divergence_threshold,
                weight_seed_offset=weight_seed_offset,
            )
            lr_sweeps[optimizer] = sweep
            best_lrs[optimizer] = sweep['best_lr']
            if verbose:
                print(f"  {optimizer.upper()} best LR among candidates: {sweep['best_lr']}")

        final_losses = {}
        for optimizer in ['sgd', 'muon']:
            final_eval = evaluate_optimizer(
                arch,
                seeds,
                optimizer,
                best_lrs[optimizer],
                batch_size=batch_size,
                num_steps=num_steps,
                momentum=momentum,
                ns_iters=ns_iters,
                divergence_threshold=divergence_threshold,
                weight_seed_offset=weight_seed_offset,
            )
            final_eval['lr'] = best_lrs[optimizer]
            final_losses[optimizer] = final_eval

        advantage_ratio = float(final_losses['sgd']['mean'] / max(final_losses['muon']['mean'], 1e-30))
        bootstrap_ci = bootstrap_ratio_of_means(
            final_losses['sgd']['finite_values'],
            final_losses['muon']['finite_values'],
            n_boot=bootstrap_samples,
            seed=bootstrap_seed,
        )
        paired_ratios = []
        for sgd_record, muon_record in zip(final_losses['sgd']['per_seed'], final_losses['muon']['per_seed']):
            if sgd_record['finite'] and muon_record['finite']:
                paired_ratios.append({
                    'seed': sgd_record['seed'],
                    'ratio': float(sgd_record['final_loss'] / max(muon_record['final_loss'], 1e-30)),
                })

        arch_result = {
            'architecture': arch,
            'dims': dims,
            'activation': activation,
            'anisotropy': anisotropy,
            'lr_sweeps': lr_sweeps,
            'best_lrs': best_lrs,
            'final_losses': final_losses,
            'advantage': {
                'ratio_of_means': advantage_ratio,
                'bootstrap_ci': bootstrap_ci,
                'paired_seed_ratios': paired_ratios,
                'paired_finite_count': len(paired_ratios),
                'definition': 'mean SGD loss / mean Muon loss',
            },
        }
        arch_result['summary'] = build_architecture_summary(arch, arch_result)
        architecture_results[arch] = arch_result
        architecture_summaries.append(arch_result['summary'])

        if verbose:
            sgd = final_losses['sgd']
            muon = final_losses['muon']
            print(
                f"  SGD final loss:  {sgd['mean']:.6e} +/- {sgd['std']:.6e} "
                f"(finite {sgd['finite_count']}/{len(sgd['per_seed'])})"
            )
            print(
                f"  Muon final loss: {muon['mean']:.6e} +/- {muon['std']:.6e} "
                f"(finite {muon['finite_count']}/{len(muon['per_seed'])})"
            )
            print(
                f"  Muon advantage: {advantage_ratio:.2f}x "
                f"[bootstrap 95% CI {bootstrap_ci['low']:.2f}, {bootstrap_ci['high']:.2f}]"
            )

    correlations = compute_correlation_summary(architecture_summaries)
    tests = compute_hypothesis_tests(architecture_summaries, correlations)

    results = {
        'experiment_id': 'H17_ANISOTROPY_PREDICTS_BENEFIT',
        'title': 'H17: Initial gradient anisotropy as a toy predictor of Muon advantage',
        'scope_note': (
            'Toy architecture-level correlation study on synthetic regression; '
            'descriptive only, not a mechanism proof.'
        ),
        'config': config,
        'seeds': [int(s) for s in seeds],
        'architecture_results': architecture_results,
        'architecture_summaries': architecture_summaries,
        'correlations': correlations,
        'tests': tests,
    }
    results['conclusion'] = build_calibrated_conclusion(results)
    results['runtime'] = {
        'elapsed_sec': float(time.time() - start_time),
    }

    if verbose:
        print('\n[H17] Descriptive correlation summary:')
        print(f"  Spearman r (linear): {correlations['linear']['spearman_r']:.3f}")
        print(f"  Spearman r (log10):  {correlations['log10']['spearman_r']:.3f}")
        print(f"  T1 passed? {tests['T1_spearman_gt_0_5']['passed']}")
        print(f"  T2 passed? {tests['T2_top_arch_match']['passed']}")

    if make_plot or save_plot or show_plot:
        resolved_plot_path = None
        if plot_path is not None:
            resolved_plot_path = Path(plot_path)
        elif save_plot:
            resolved_plot_path = SCRIPT_DIR / PLOT_FILENAME
        fig, axes = make_summary_plot(
            results,
            save_path=resolved_plot_path,
            show=show_plot,
        )
        results['plot'] = {
            'path': str(resolved_plot_path) if resolved_plot_path is not None else None,
            'figure_created': True,
        }
        if not show_plot:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
    else:
        results['plot'] = {
            'path': None,
            'figure_created': False,
        }

    return results


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================


def main():
    results = run_experiment(make_plot=True, save_plot=True, show_plot=False, verbose=True)
    print_results_report(results)
    if results['plot']['path']:
        print(f"Plot saved: {results['plot']['path']}")


if __name__ == '__main__':
    main()
