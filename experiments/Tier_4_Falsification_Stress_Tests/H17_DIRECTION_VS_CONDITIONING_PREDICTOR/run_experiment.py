#!/usr/bin/env python3
"""
H17: Initial Gradient Anisotropy as a Predictor of Muon Architecture Benefit
============================================================================

This experiment evaluates whether initial gradient anisotropy predicts how
much Muon outperforms momentum SGD across several small architectures on a
fixed Gaussian regression task.

What is measured:
  - initial gradient anisotropy = mean over layers of sigma_max(G_l)/sigma_min(G_l)
  - final training loss after optimizing with Muon and momentum SGD
  - Muon advantage = mean_finite_loss_sgd / mean_finite_loss_muon
  - architecture-level Pearson correlation between anisotropy and advantage

What is not yet measured:
  - no separate direction-quality metric
  - no direct mechanistic split between conditioning and update direction
  - no predictive train/test split beyond the architecture-level correlation

Default setup: 4 architectures, 4 layers, 32x32 weights, 500 steps, 5 seeds.
"""

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.abspath(__file__)

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
WEIGHT_SEED_OFFSET = 5000
LR_SWEEP_SEED_COUNT = 3

LR_MUON = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003]
LR_SGD = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003]
ARCHITECTURES = ['deep_linear', 'relu_net', 'tanh_net', 'bottleneck']


def default_seeds(num_seeds: int = NUM_SEEDS) -> List[int]:
    return [42 + i * 137 for i in range(num_seeds)]


def build_config(
    num_steps: int = NUM_STEPS,
    num_seeds: int = NUM_SEEDS,
    lr_sweep_seed_count: int = LR_SWEEP_SEED_COUNT,
    dim: int = DIM,
    num_layers: int = NUM_LAYERS,
    momentum: float = MOMENTUM,
    ns_iters: int = NS_ITERS,
    batch_size: int = BATCH_SIZE,
    lr_muon: Optional[List[float]] = None,
    lr_sgd: Optional[List[float]] = None,
    architectures: Optional[List[str]] = None,
    weight_seed_offset: int = WEIGHT_SEED_OFFSET,
) -> Dict[str, Any]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")

    seeds = default_seeds(num_seeds)
    lr_sweep_seed_count = max(1, min(lr_sweep_seed_count, len(seeds)))
    sweep_seeds = seeds[:lr_sweep_seed_count]
    archs = list(architectures) if architectures is not None else list(ARCHITECTURES)
    muon_lrs = list(lr_muon) if lr_muon is not None else list(LR_MUON)
    sgd_lrs = list(lr_sgd) if lr_sgd is not None else list(LR_SGD)

    estimated_train_calls = len(archs) * (
        (len(muon_lrs) + len(sgd_lrs)) * len(sweep_seeds) + 2 * len(seeds)
    )

    return {
        'dim': int(dim),
        'num_layers': int(num_layers),
        'num_steps': int(num_steps),
        'momentum': float(momentum),
        'ns_iters': int(ns_iters),
        'num_seeds': int(num_seeds),
        'batch_size': int(batch_size),
        'weight_seed_offset': int(weight_seed_offset),
        'lr_muon': muon_lrs,
        'lr_sgd': sgd_lrs,
        'architectures': archs,
        'seeds': seeds,
        'lr_sweep_seed_count': int(lr_sweep_seed_count),
        'lr_sweep_seeds': sweep_seeds,
        'estimated_train_calls': int(estimated_train_calls),
    }


def newton_schulz(M: np.ndarray, n_iters: int = NS_ITERS) -> np.ndarray:
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(
    arch: str,
    seed: int,
    dim: int = DIM,
    num_layers: int = NUM_LAYERS,
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    if arch in {'deep_linear', 'relu_net', 'tanh_net'}:
        return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(num_layers)]
    if arch == 'bottleneck':
        weights = []
        for _ in range(num_layers):
            W = rng.randn(dim, dim) * 0.1
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            s[dim // 2:] *= 0.01
            W = U @ np.diag(s) @ Vt
            W += np.eye(dim) * 0.5
            weights.append(W)
        return weights
    raise ValueError(f"Unknown architecture: {arch}")


def apply_act(x: np.ndarray, arch: str, layer_idx: int, num_layers: int) -> np.ndarray:
    if arch == 'deep_linear' or layer_idx == num_layers - 1:
        return x
    if arch in {'relu_net', 'bottleneck'}:
        return np.maximum(0, x)
    if arch == 'tanh_net':
        return np.tanh(x)
    raise ValueError(f"Unknown architecture: {arch}")


def act_deriv(pre: np.ndarray, arch: str, layer_idx: int, num_layers: int) -> np.ndarray:
    if arch == 'deep_linear' or layer_idx == num_layers - 1:
        return np.ones_like(pre)
    if arch in {'relu_net', 'bottleneck'}:
        return (pre > 0).astype(float)
    if arch == 'tanh_net':
        return 1 - np.tanh(pre) ** 2
    raise ValueError(f"Unknown architecture: {arch}")


def forward(weights: List[np.ndarray], X: np.ndarray, arch: str) -> np.ndarray:
    out = X.copy()
    for idx, W in enumerate(weights):
        out = W @ out
        out = apply_act(out, arch, idx, len(weights))
    return out


def compute_loss(weights: List[np.ndarray], X: np.ndarray, Y: np.ndarray, arch: str) -> float:
    pred = forward(weights, X, arch)
    return float(0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0)))


def compute_gradients(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    arch: str,
) -> List[np.ndarray]:
    L = len(weights)
    N = X.shape[1]
    acts_post = [X.copy()]
    pre_acts = []
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre)
        out = apply_act(pre, arch, idx, L)
        acts_post.append(out)
    delta = (acts_post[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts_post[l].T
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * act_deriv(pre_acts[l - 1], arch, l - 1, L)
    return grads


def gradient_spectrum_stats(
    weights: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    arch: str,
    sv_floor: float = 1e-12,
) -> List[Dict[str, float]]:
    grads = compute_gradients(weights, X, Y, arch)
    layer_stats = []
    for layer_idx, G in enumerate(grads):
        s = np.linalg.svd(G, compute_uv=False)
        sigma_max = float(s[0])
        sigma_min_raw = float(s[-1])
        sigma_min = sigma_min_raw if sigma_min_raw > sv_floor else float(sv_floor)
        anisotropy = float(sigma_max / sigma_min)
        layer_stats.append({
            'layer': int(layer_idx),
            'sigma_max': sigma_max,
            'sigma_min': sigma_min,
            'sigma_min_raw': sigma_min_raw,
            'anisotropy': anisotropy,
        })
    return layer_stats


def gradient_anisotropy(weights: List[np.ndarray], X: np.ndarray, Y: np.ndarray, arch: str) -> float:
    layer_stats = gradient_spectrum_stats(weights, X, Y, arch)
    return float(np.mean([layer['anisotropy'] for layer in layer_stats]))


def train(
    weights_init: List[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    lr: float,
    optimizer: str,
    arch: str,
    num_steps: int = NUM_STEPS,
    momentum: float = MOMENTUM,
    ns_iters: int = NS_ITERS,
) -> float:
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(num_steps):
        loss = compute_loss(weights, X, Y, arch)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y, arch)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = momentum * mom[i] + newton_schulz(grads[i], n_iters=ns_iters)
            elif optimizer == 'sgd':
                mom[i] = momentum * mom[i] + grads[i]
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y, arch)


def make_data(seed: int, dim: int = DIM, batch_size: int = BATCH_SIZE) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(dim, batch_size) * 0.3
    Y = rng.randn(dim, batch_size) * 0.3
    return X, Y


def summarize_losses(seeds: List[int], losses: List[float]) -> Dict[str, Any]:
    clean_losses = [float(loss) for loss in losses]
    finite_losses = [loss for loss in clean_losses if np.isfinite(loss)]
    finite_count = len(finite_losses)
    summary = {
        'losses': clean_losses,
        'seed_records': [
            {
                'data_seed': int(seed),
                'final_loss': float(loss),
                'is_finite': bool(np.isfinite(loss)),
            }
            for seed, loss in zip(seeds, clean_losses)
        ],
        'num_runs': len(clean_losses),
        'finite_count': finite_count,
        'divergent_count': len(clean_losses) - finite_count,
        'mean_finite_loss': float(np.mean(finite_losses)) if finite_losses else float('inf'),
        'std_finite_loss': float(np.std(finite_losses, ddof=1)) if len(finite_losses) > 1 else 0.0,
        'median_finite_loss': float(np.median(finite_losses)) if finite_losses else float('inf'),
    }
    return summary


def evaluate_lr_candidates(
    arch: str,
    optimizer: str,
    candidate_lrs: List[float],
    sweep_seeds: List[int],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    records = []
    best_lr = candidate_lrs[-1]
    best_mean_loss = float('inf')

    for lr in candidate_lrs:
        losses = []
        for data_seed in sweep_seeds:
            X, Y = make_data(data_seed, dim=config['dim'], batch_size=config['batch_size'])
            weights = init_weights(
                arch,
                data_seed + config['weight_seed_offset'],
                dim=config['dim'],
                num_layers=config['num_layers'],
            )
            final_loss = train(
                weights,
                X,
                Y,
                lr,
                optimizer,
                arch,
                num_steps=config['num_steps'],
                momentum=config['momentum'],
                ns_iters=config['ns_iters'],
            )
            losses.append(final_loss)

        summary = summarize_losses(sweep_seeds, losses)
        record = {'lr': float(lr), **summary}
        records.append(record)

        if summary['mean_finite_loss'] < best_mean_loss:
            best_mean_loss = summary['mean_finite_loss']
            best_lr = lr

    return {
        'optimizer': optimizer,
        'candidate_lrs': [float(lr) for lr in candidate_lrs],
        'records': records,
        'best_lr': float(best_lr),
        'best_mean_finite_loss': float(best_mean_loss),
    }


def evaluate_architecture(arch: str, config: Dict[str, Any]) -> Dict[str, Any]:
    seeds = config['seeds']
    sweep_seeds = config['lr_sweep_seeds']

    anisotropy_records = []
    for data_seed in seeds:
        X, Y = make_data(data_seed, dim=config['dim'], batch_size=config['batch_size'])
        weight_seed = data_seed + config['weight_seed_offset']
        weights = init_weights(
            arch,
            weight_seed,
            dim=config['dim'],
            num_layers=config['num_layers'],
        )
        layer_stats = gradient_spectrum_stats(weights, X, Y, arch)
        mean_anisotropy = float(np.mean([layer['anisotropy'] for layer in layer_stats]))
        anisotropy_records.append({
            'data_seed': int(data_seed),
            'weight_seed': int(weight_seed),
            'mean_anisotropy': mean_anisotropy,
            'layer_stats': layer_stats,
        })

    anisotropy_values = [record['mean_anisotropy'] for record in anisotropy_records]
    representative_diagnostic = anisotropy_records[0]

    lr_sweeps = {
        'muon': evaluate_lr_candidates(arch, 'muon', config['lr_muon'], sweep_seeds, config),
        'sgd': evaluate_lr_candidates(arch, 'sgd', config['lr_sgd'], sweep_seeds, config),
    }

    final_training = {}
    for optimizer in ['muon', 'sgd']:
        best_lr = lr_sweeps[optimizer]['best_lr']
        losses = []
        for data_seed in seeds:
            X, Y = make_data(data_seed, dim=config['dim'], batch_size=config['batch_size'])
            weights = init_weights(
                arch,
                data_seed + config['weight_seed_offset'],
                dim=config['dim'],
                num_layers=config['num_layers'],
            )
            final_loss = train(
                weights,
                X,
                Y,
                best_lr,
                optimizer,
                arch,
                num_steps=config['num_steps'],
                momentum=config['momentum'],
                ns_iters=config['ns_iters'],
            )
            losses.append(final_loss)
        final_training[optimizer] = {
            'optimizer': optimizer,
            'lr': float(best_lr),
            **summarize_losses(seeds, losses),
        }

    muon_mean = final_training['muon']['mean_finite_loss']
    sgd_mean = final_training['sgd']['mean_finite_loss']
    advantage_ratio_of_means = float(sgd_mean / max(muon_mean, 1e-30))

    per_seed_advantage_records = []
    for muon_record, sgd_record in zip(
        final_training['muon']['seed_records'],
        final_training['sgd']['seed_records'],
    ):
        muon_loss = muon_record['final_loss']
        sgd_loss = sgd_record['final_loss']
        both_finite = bool(np.isfinite(muon_loss) and np.isfinite(sgd_loss))
        per_seed_advantage_records.append({
            'data_seed': int(muon_record['data_seed']),
            'muon_loss': float(muon_loss),
            'sgd_loss': float(sgd_loss),
            'both_finite': both_finite,
            'advantage': float(sgd_loss / max(muon_loss, 1e-30)) if both_finite else None,
        })

    finite_advantages = [
        record['advantage']
        for record in per_seed_advantage_records
        if record['advantage'] is not None and np.isfinite(record['advantage'])
    ]

    return {
        'architecture': arch,
        'anisotropy_records': anisotropy_records,
        'anisotropy_per_seed': [float(value) for value in anisotropy_values],
        'anisotropy_mean': float(np.mean(anisotropy_values)),
        'anisotropy_std': float(np.std(anisotropy_values, ddof=1)) if len(anisotropy_values) > 1 else 0.0,
        'representative_diagnostic': representative_diagnostic,
        'lr_sweeps': lr_sweeps,
        'final_training': final_training,
        'advantage_ratio_of_means': advantage_ratio_of_means,
        'per_seed_advantage_records': per_seed_advantage_records,
        'mean_finite_per_seed_advantage': float(np.mean(finite_advantages)) if finite_advantages else float('nan'),
    }


def run_experiment(
    num_steps: int = NUM_STEPS,
    num_seeds: int = NUM_SEEDS,
    lr_sweep_seed_count: int = LR_SWEEP_SEED_COUNT,
    dim: int = DIM,
    num_layers: int = NUM_LAYERS,
    momentum: float = MOMENTUM,
    ns_iters: int = NS_ITERS,
    batch_size: int = BATCH_SIZE,
    lr_muon: Optional[List[float]] = None,
    lr_sgd: Optional[List[float]] = None,
    architectures: Optional[List[str]] = None,
    weight_seed_offset: int = WEIGHT_SEED_OFFSET,
    verbose: bool = False,
) -> Dict[str, Any]:
    config = build_config(
        num_steps=num_steps,
        num_seeds=num_seeds,
        lr_sweep_seed_count=lr_sweep_seed_count,
        dim=dim,
        num_layers=num_layers,
        momentum=momentum,
        ns_iters=ns_iters,
        batch_size=batch_size,
        lr_muon=lr_muon,
        lr_sgd=lr_sgd,
        architectures=architectures,
        weight_seed_offset=weight_seed_offset,
    )

    architecture_results: Dict[str, Dict[str, Any]] = {}
    for idx, arch in enumerate(config['architectures'], start=1):
        if verbose:
            print(f"[{idx}/{len(config['architectures'])}] Evaluating {arch}...")
        architecture_results[arch] = evaluate_architecture(arch, config)
        if verbose:
            arch_result = architecture_results[arch]
            print(
                f"    anisotropy={arch_result['anisotropy_mean']:.2f}, "
                f"advantage={arch_result['advantage_ratio_of_means']:.2f}x, "
                f"best_lrs=(muon {arch_result['lr_sweeps']['muon']['best_lr']}, "
                f"sgd {arch_result['lr_sweeps']['sgd']['best_lr']})"
            )

    anisos = [architecture_results[arch]['anisotropy_mean'] for arch in config['architectures']]
    advs = [architecture_results[arch]['advantage_ratio_of_means'] for arch in config['architectures']]
    corr = float(np.corrcoef(anisos, advs)[0, 1]) if len(config['architectures']) > 1 else float('nan')

    max_aniso_arch = config['architectures'][int(np.argmax(anisos))]
    max_adv_arch = config['architectures'][int(np.argmax(advs))]
    t1 = bool(corr > 0.5)
    t2 = bool(max_aniso_arch == max_adv_arch)

    summary_rows = []
    for arch in config['architectures']:
        arch_result = architecture_results[arch]
        summary_rows.append({
            'architecture': arch,
            'anisotropy_mean': arch_result['anisotropy_mean'],
            'anisotropy_std': arch_result['anisotropy_std'],
            'muon_best_lr': arch_result['lr_sweeps']['muon']['best_lr'],
            'sgd_best_lr': arch_result['lr_sweeps']['sgd']['best_lr'],
            'muon_mean_finite_loss': arch_result['final_training']['muon']['mean_finite_loss'],
            'sgd_mean_finite_loss': arch_result['final_training']['sgd']['mean_finite_loss'],
            'muon_divergent_count': arch_result['final_training']['muon']['divergent_count'],
            'sgd_divergent_count': arch_result['final_training']['sgd']['divergent_count'],
            'advantage_ratio_of_means': arch_result['advantage_ratio_of_means'],
            'mean_finite_per_seed_advantage': arch_result['mean_finite_per_seed_advantage'],
        })

    return {
        'title': 'H17: Initial Gradient Anisotropy as a Predictor of Muon Architecture Benefit',
        'script_path': SCRIPT_PATH,
        'script_dir': SCRIPT_DIR,
        'question': (
            'Does initial gradient anisotropy predict architecture-level Muon benefit '
            'across the tested architectures?'
        ),
        'limitations': [
            'This implementation does not compute a separate direction-quality metric.',
            'It does not directly isolate conditioning versus direction mechanisms.',
            'The architecture-level correlation is descriptive for this setup, not a held-out predictive test.',
            'The SGD baseline here is momentum SGD with momentum=0.9, not plain SGD.',
        ],
        'config': config,
        'architectures': architecture_results,
        'summary_rows': summary_rows,
        'correlation_analysis': {
            'anisotropy_values': [float(value) for value in anisos],
            'advantage_values': [float(value) for value in advs],
            'pearson_r': corr,
            't1_positive_correlation_gt_0_5': t1,
            't2_same_argmax_architecture': t2,
            'max_anisotropy_architecture': max_aniso_arch,
            'max_advantage_architecture': max_adv_arch,
        },
    }


def _fmt(value: float, precision: int = 3) -> str:
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return str(value)
    if not np.isfinite(value):
        return str(value)
    return f"{value:.{precision}f}"


def print_summary(results: Dict[str, Any]) -> None:
    config = results['config']
    corr = results['correlation_analysis']

    print("=" * 100)
    print(results['title'])
    print("=" * 100)
    print(f"Script: {results['script_path']}")
    print(f"Architectures: {config['architectures']}")
    print(
        f"Network: {config['num_layers']}-layer, {config['dim']}x{config['dim']}, "
        f"batch={config['batch_size']}, steps={config['num_steps']}"
    )
    print(f"Seeds: {config['seeds']}")
    print(f"LR sweep seeds: {config['lr_sweep_seeds']}")
    print(f"Estimated train() calls: {config['estimated_train_calls']}")
    print(f"Optimizers compared: Muon vs SGD+momentum (momentum={config['momentum']})")
    print()
    print("Measured quantities:")
    print("  - Initial mean gradient anisotropy across layers")
    print("  - Best-LR final loss for Muon and SGD+momentum")
    print("  - Architecture-level correlation between anisotropy and Muon advantage")
    print("Not measured here:")
    for limitation in results['limitations'][:3]:
        print(f"  - {limitation}")

    print(f"\n{'Architecture':>15}  {'Aniso mean':>12}  {'Muon loss':>12}  {'SGD loss':>12}  {'Advantage':>11}  {'Div(mu/sgd)':>12}")
    print("  " + "-" * 87)
    for row in results['summary_rows']:
        div_pair = f"{row['muon_divergent_count']}/{row['sgd_divergent_count']}"
        print(
            f"{row['architecture']:>15}  "
            f"{_fmt(row['anisotropy_mean'], 2):>12}  "
            f"{_fmt(row['muon_mean_finite_loss'], 4):>12}  "
            f"{_fmt(row['sgd_mean_finite_loss'], 4):>12}  "
            f"{_fmt(row['advantage_ratio_of_means'], 2) + 'x':>11}  "
            f"{div_pair:>12}"
        )

    print(f"\nPearson correlation (anisotropy vs advantage): r = {_fmt(corr['pearson_r'], 3)}")
    print(f"T1: positive correlation r > 0.5? {'PASS' if corr['t1_positive_correlation_gt_0_5'] else 'FAIL'}")
    print(
        "T2: highest-anisotropy architecture also has highest advantage? "
        f"{'PASS' if corr['t2_same_argmax_architecture'] else 'FAIL'}"
    )
    print(f"    max anisotropy architecture: {corr['max_anisotropy_architecture']}")
    print(f"    max advantage architecture:  {corr['max_advantage_architecture']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run H17: initial gradient anisotropy as a predictor of Muon architecture benefit.'
        )
    )
    parser.add_argument('--num-steps', type=int, default=NUM_STEPS, help='Training steps per run.')
    parser.add_argument('--num-seeds', type=int, default=NUM_SEEDS, help='Number of evaluation seeds.')
    parser.add_argument(
        '--lr-sweep-seeds',
        type=int,
        default=LR_SWEEP_SEED_COUNT,
        help='Number of seeds used in the learning-rate sweep.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable per-architecture progress messages during execution.',
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    results = run_experiment(
        num_steps=args.num_steps,
        num_seeds=args.num_seeds,
        lr_sweep_seed_count=args.lr_sweep_seeds,
        verbose=not args.quiet,
    )
    print_summary(results)
    return results


if __name__ == '__main__':
    main()
