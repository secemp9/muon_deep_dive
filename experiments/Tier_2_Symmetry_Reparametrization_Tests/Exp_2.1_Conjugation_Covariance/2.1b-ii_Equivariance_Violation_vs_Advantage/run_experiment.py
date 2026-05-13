#!/usr/bin/env python3
"""
2.1b-ii: Layerwise equivariance violation vs a Muon condition-number proxy.
============================================================================

This file keeps the original toy study intact as much as possible while making the
computation import-safe and reusable.

What is currently measured
--------------------------
For each layer of a deep linear network, we measure two quantities:
  1. Equivariance violation under orthogonal conjugation at initialization.
  2. A Muon "advantage" proxy defined as
         kappa_ratio = cond(W_sgd) / cond(W_muon)
     at the end of training.

A ratio > 1 means Muon produced a lower-condition-number layer on this specific
proxy. A ratio < 1 means SGD produced the lower-condition-number layer. This is a
narrow spectral proxy, not a general proof of optimizer advantage.

Primary toy question
--------------------
Do layers with larger equivariance violation also show larger values of this
condition-number proxy? The current experiment answers this only through a small,
layerwise correlation analysis on toy data.

Current verdict logic
---------------------
  T1: Pearson correlation on layer means is positive.
  T3: The Pearson correlation magnitude exceeds 0.5.

No formal significance claims are made beyond these descriptive thresholds.
"""

import os
import time
import numpy as np

try:
    from scipy.stats import spearmanr as scipy_spearmanr
except Exception:  # pragma: no cover - optional dependency fallback
    scipy_spearmanr = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(SCRIPT_DIR, 'run_experiment.ipynb')

DIM = 32
DEPTH = 8
NUM_STEPS = 300
LR_MUON = 0.01
LR_SGD = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
DATA_SCALE = 0.3
INIT_SCALE = 0.1
BASE_SEED = 42
SEED_STRIDE = 137
INIT_SEED_OFFSET = 5000
VIOLATION_SEED_STRIDE = 1000
DIVERGENCE_THRESHOLD = 1e10


def get_default_config():
    return {
        'dim': DIM,
        'depth': DEPTH,
        'num_steps': NUM_STEPS,
        'lr_muon': LR_MUON,
        'lr_sgd': LR_SGD,
        'momentum': MOMENTUM,
        'ns_iters': NS_ITERS,
        'num_seeds': NUM_SEEDS,
        'batch_size': BATCH_SIZE,
        'data_scale': DATA_SCALE,
        'init_scale': INIT_SCALE,
        'base_seed': BASE_SEED,
        'seed_stride': SEED_STRIDE,
        'init_seed_offset': INIT_SEED_OFFSET,
        'violation_seed_stride': VIOLATION_SEED_STRIDE,
        'divergence_threshold': DIVERGENCE_THRESHOLD,
    }


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


def init_weights(seed, dim=DIM, depth=DEPTH, init_scale=INIT_SCALE):
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * init_scale for _ in range(depth)]


def compute_loss_and_grads(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    loss = 0.5 * np.mean(np.sum((acts[-1] - Y) ** 2, axis=0))
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return loss, grads


def condition_number(W):
    svs = np.linalg.svd(W, compute_uv=False)
    return float(svs[0] / max(svs[-1], 1e-12))


def safe_pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float('nan')
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_fallback(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def safe_spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return {'correlation': float('nan'), 'pvalue': float('nan'), 'method': 'insufficient-data'}
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return {'correlation': float('nan'), 'pvalue': float('nan'), 'method': 'constant-input'}
    if scipy_spearmanr is not None:
        result = scipy_spearmanr(x, y)
        correlation = getattr(result, 'correlation', result[0])
        pvalue = getattr(result, 'pvalue', result[1])
        return {
            'correlation': float(correlation),
            'pvalue': float(pvalue),
            'method': 'scipy.stats.spearmanr',
        }
    rank_x = _rankdata_fallback(x)
    rank_y = _rankdata_fallback(y)
    return {
        'correlation': safe_pearson_corr(rank_x, rank_y),
        'pvalue': float('nan'),
        'method': 'rank-pearson-fallback',
    }


def _summary_stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'sem': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'count': 0,
        }
    std = float(np.std(arr))
    return {
        'mean': float(np.mean(arr)),
        'std': std,
        'sem': float(std / np.sqrt(arr.size)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': int(arr.size),
    }


def train(
    weights_init,
    X,
    Y,
    optimizer,
    lr,
    num_steps=NUM_STEPS,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
    divergence_threshold=DIVERGENCE_THRESHOLD,
):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    stopped_early = False
    steps_completed = 0

    for step in range(num_steps):
        loss, grads = compute_loss_and_grads(weights, X, Y)
        if not np.isfinite(loss) or loss > divergence_threshold:
            stopped_early = True
            break
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = momentum * mom[i] + newton_schulz(grads[i], n_iters=ns_iters)
            else:
                mom[i] = momentum * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
        steps_completed = step + 1

    final_loss, _ = compute_loss_and_grads(weights, X, Y)
    return {
        'weights': weights,
        'final_loss': float(final_loss),
        'steps_completed': int(steps_completed),
        'stopped_early': bool(stopped_early),
        'optimizer': optimizer,
        'lr': float(lr),
    }


def measure_equivariance_violation(
    weights_init,
    X,
    Y,
    target_layer,
    rng,
    reference_muon_weights=None,
    lr_muon=LR_MUON,
    num_steps=NUM_STEPS,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
    divergence_threshold=DIVERGENCE_THRESHOLD,
):
    dim = weights_init[target_layer].shape[0]
    R = random_orthogonal(dim, rng)
    S = random_orthogonal(dim, rng)

    if reference_muon_weights is None:
        reference_train = train(
            [W.copy() for W in weights_init],
            X,
            Y,
            'muon',
            lr_muon,
            num_steps=num_steps,
            momentum=momentum,
            ns_iters=ns_iters,
            divergence_threshold=divergence_threshold,
        )
        weights_A = reference_train['weights']
    else:
        weights_A = [W.copy() for W in reference_muon_weights]

    weights_conj = [W.copy() for W in weights_init]
    weights_conj[target_layer] = R @ weights_conj[target_layer] @ S.T
    conjugated_train = train(
        weights_conj,
        X,
        Y,
        'muon',
        lr_muon,
        num_steps=num_steps,
        momentum=momentum,
        ns_iters=ns_iters,
        divergence_threshold=divergence_threshold,
    )
    weights_B = conjugated_train['weights']

    expected = R @ weights_A[target_layer] @ S.T
    actual = weights_B[target_layer]
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(weights_A[target_layer]), 1e-30))


def _print_verbose_seed_header(seed_idx, num_seeds, seed, X, Y, weights_init):
    init_kappas = [condition_number(W) for W in weights_init]
    print(f"\n--- Seed {seed_idx + 1}/{num_seeds} (seed={seed}) ---")
    print(f"  Data: X shape={X.shape}, ||X||_F={np.linalg.norm(X):.3f}, ||Y||_F={np.linalg.norm(Y):.3f}")
    print(f"  Initial kappa range: [{min(init_kappas):.2f}, {max(init_kappas):.2f}]")


def run_experiment(
    dim=DIM,
    depth=DEPTH,
    num_steps=NUM_STEPS,
    lr_muon=LR_MUON,
    lr_sgd=LR_SGD,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
    num_seeds=NUM_SEEDS,
    batch_size=BATCH_SIZE,
    data_scale=DATA_SCALE,
    init_scale=INIT_SCALE,
    base_seed=BASE_SEED,
    seed_stride=SEED_STRIDE,
    init_seed_offset=INIT_SEED_OFFSET,
    violation_seed_stride=VIOLATION_SEED_STRIDE,
    divergence_threshold=DIVERGENCE_THRESHOLD,
    verbose=False,
):
    start_time = time.time()

    config = {
        'dim': int(dim),
        'depth': int(depth),
        'num_steps': int(num_steps),
        'lr_muon': float(lr_muon),
        'lr_sgd': float(lr_sgd),
        'momentum': float(momentum),
        'ns_iters': int(ns_iters),
        'num_seeds': int(num_seeds),
        'batch_size': int(batch_size),
        'data_scale': float(data_scale),
        'init_scale': float(init_scale),
        'base_seed': int(base_seed),
        'seed_stride': int(seed_stride),
        'init_seed_offset': int(init_seed_offset),
        'violation_seed_stride': int(violation_seed_stride),
        'divergence_threshold': float(divergence_threshold),
    }

    seeds = []
    violations = {l: [] for l in range(depth)}
    kappa_ratios = {l: [] for l in range(depth)}
    kappa_sgd_values = {l: [] for l in range(depth)}
    kappa_muon_values = {l: [] for l in range(depth)}
    seed_results = []
    violation_matrix = []
    kappa_ratio_matrix = []

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx * seed_stride
        seeds.append(int(seed))
        rng = np.random.RandomState(seed)
        X = rng.randn(dim, batch_size) * data_scale
        Y = rng.randn(dim, batch_size) * data_scale
        weights_init = init_weights(seed + init_seed_offset, dim=dim, depth=depth, init_scale=init_scale)

        if verbose:
            _print_verbose_seed_header(seed_idx, num_seeds, seed, X, Y, weights_init)

        sgd_train = train(
            [W.copy() for W in weights_init],
            X,
            Y,
            'sgd',
            lr_sgd,
            num_steps=num_steps,
            momentum=momentum,
            ns_iters=ns_iters,
            divergence_threshold=divergence_threshold,
        )
        muon_train = train(
            [W.copy() for W in weights_init],
            X,
            Y,
            'muon',
            lr_muon,
            num_steps=num_steps,
            momentum=momentum,
            ns_iters=ns_iters,
            divergence_threshold=divergence_threshold,
        )

        final_sgd = sgd_train['weights']
        final_muon = muon_train['weights']

        if verbose:
            print(
                f"  Final loss -- SGD: {sgd_train['final_loss']:.6e}, "
                f"Muon: {muon_train['final_loss']:.6e}"
            )

        seed_violations = []
        seed_kappa_ratios = []
        seed_kappa_sgd = []
        seed_kappa_muon = []

        for l in range(depth):
            kappa_sgd = condition_number(final_sgd[l])
            kappa_muon = condition_number(final_muon[l])
            kappa_ratio = kappa_sgd / max(kappa_muon, 1e-12)

            rng_v = np.random.RandomState(seed + l * violation_seed_stride)
            violation = measure_equivariance_violation(
                weights_init,
                X,
                Y,
                l,
                rng_v,
                reference_muon_weights=final_muon,
                lr_muon=lr_muon,
                num_steps=num_steps,
                momentum=momentum,
                ns_iters=ns_iters,
                divergence_threshold=divergence_threshold,
            )

            seed_violations.append(float(violation))
            seed_kappa_ratios.append(float(kappa_ratio))
            seed_kappa_sgd.append(float(kappa_sgd))
            seed_kappa_muon.append(float(kappa_muon))

            violations[l].append(float(violation))
            kappa_ratios[l].append(float(kappa_ratio))
            kappa_sgd_values[l].append(float(kappa_sgd))
            kappa_muon_values[l].append(float(kappa_muon))

        violation_matrix.append(seed_violations)
        kappa_ratio_matrix.append(seed_kappa_ratios)

        seed_spearman = safe_spearman_corr(seed_violations, seed_kappa_ratios)
        seed_results.append({
            'seed_idx': int(seed_idx),
            'seed': int(seed),
            'data_norms': {
                'X_fro': float(np.linalg.norm(X)),
                'Y_fro': float(np.linalg.norm(Y)),
            },
            'initial_condition_numbers': [float(condition_number(W)) for W in weights_init],
            'sgd': {
                'final_loss': float(sgd_train['final_loss']),
                'steps_completed': int(sgd_train['steps_completed']),
                'stopped_early': bool(sgd_train['stopped_early']),
            },
            'muon': {
                'final_loss': float(muon_train['final_loss']),
                'steps_completed': int(muon_train['steps_completed']),
                'stopped_early': bool(muon_train['stopped_early']),
            },
            'per_layer': [
                {
                    'layer': int(l),
                    'violation': float(seed_violations[l]),
                    'kappa_ratio': float(seed_kappa_ratios[l]),
                    'kappa_sgd': float(seed_kappa_sgd[l]),
                    'kappa_muon': float(seed_kappa_muon[l]),
                }
                for l in range(depth)
            ],
            'pearson_across_layers': float(safe_pearson_corr(seed_violations, seed_kappa_ratios)),
            'spearman_across_layers': float(seed_spearman['correlation']),
            'spearman_across_layers_pvalue': float(seed_spearman['pvalue']),
        })

    layer_summaries = []
    mean_violations = []
    mean_kappa_ratios = []

    for l in range(depth):
        viol_stats = _summary_stats(violations[l])
        ratio_stats = _summary_stats(kappa_ratios[l])
        kappa_sgd_stats = _summary_stats(kappa_sgd_values[l])
        kappa_muon_stats = _summary_stats(kappa_muon_values[l])
        mean_violations.append(viol_stats['mean'])
        mean_kappa_ratios.append(ratio_stats['mean'])
        layer_summaries.append({
            'layer': int(l),
            'violation_mean': float(viol_stats['mean']),
            'violation_std': float(viol_stats['std']),
            'violation_sem': float(viol_stats['sem']),
            'violation_min': float(viol_stats['min']),
            'violation_max': float(viol_stats['max']),
            'kappa_ratio_mean': float(ratio_stats['mean']),
            'kappa_ratio_std': float(ratio_stats['std']),
            'kappa_ratio_sem': float(ratio_stats['sem']),
            'kappa_ratio_min': float(ratio_stats['min']),
            'kappa_ratio_max': float(ratio_stats['max']),
            'kappa_sgd_mean': float(kappa_sgd_stats['mean']),
            'kappa_muon_mean': float(kappa_muon_stats['mean']),
            'muon_favored_fraction': float(np.mean(np.asarray(kappa_ratios[l]) > 1.0)),
        })

    pearson_layer_means = safe_pearson_corr(mean_violations, mean_kappa_ratios)
    spearman_layer_means = safe_spearman_corr(mean_violations, mean_kappa_ratios)

    seedwise_pearsons = [seed_result['pearson_across_layers'] for seed_result in seed_results]
    seedwise_spearmans = [seed_result['spearman_across_layers'] for seed_result in seed_results]
    sgd_losses = [seed_result['sgd']['final_loss'] for seed_result in seed_results]
    muon_losses = [seed_result['muon']['final_loss'] for seed_result in seed_results]
    mean_kappa_ratio_array = np.asarray(mean_kappa_ratios)
    layers_with_mean_kappa_ratio_gt_1 = int(np.sum(mean_kappa_ratio_array > 1.0))
    fraction_layers_with_mean_kappa_ratio_gt_1 = float(np.mean(mean_kappa_ratio_array > 1.0))

    tests = [
        {
            'id': 'T1',
            'description': 'Positive Pearson correlation between layerwise mean violation and layerwise mean kappa ratio',
            'statistic_name': 'pearson_r',
            'statistic_value': float(pearson_layer_means),
            'threshold': '> 0',
            'pass': bool(pearson_layer_means > 0),
        },
        {
            'id': 'T3',
            'description': 'Large-magnitude Pearson correlation on layer means',
            'statistic_name': '|pearson_r|',
            'statistic_value': float(abs(pearson_layer_means)),
            'threshold': '> 0.5',
            'pass': bool(abs(pearson_layer_means) > 0.5),
        },
    ]

    supports_positive_relationship = bool(pearson_layer_means > 0)
    verdict_summary = (
        'This toy run supports a positive relationship between layerwise equivariance '
        'violation and the condition-number proxy.'
        if supports_positive_relationship
        else 'This toy run does not support a positive relationship between layerwise '
             'equivariance violation and the condition-number proxy.'
    )

    elapsed_sec = time.time() - start_time
    results = {
        'experiment_id': '2.1b-ii_Equivariance_Violation_vs_Advantage',
        'title': 'Layerwise equivariance violation vs Muon condition-number proxy',
        'primary_question': (
            'Do layers with larger measured equivariance violation also show larger values '
            'of the condition-number ratio cond(W_sgd) / cond(W_muon)?'
        ),
        'analysis_notes': {
            'primary_statistic': 'Pearson correlation across layerwise mean pairs',
            'secondary_statistic': 'Spearman correlation across layerwise mean pairs',
            'layer_point_count': int(depth),
            'layer_point_note': (
                f'The headline layer-mean correlation uses {depth} aggregated layer points '
                '(one mean pair per layer).'
            ),
        },
        'script_dir': SCRIPT_DIR,
        'script_path': os.path.abspath(__file__),
        'notebook_path': NOTEBOOK_PATH,
        'scope_note': (
            'Toy layerwise correlation study: violation is measured in target-layer '
            'weight space after conjugation, and "advantage" is the condition-number '
            'ratio cond(SGD)/cond(Muon).'
        ),
        'advantage_proxy': {
            'name': 'kappa_ratio',
            'definition': 'cond(W_sgd) / cond(W_muon)',
            'interpretation': '> 1 favors Muon on this conditioning proxy; < 1 favors SGD on this proxy',
            'warning': 'This is not a full optimizer-advantage metric and does not measure per-layer loss contribution.',
        },
        'config': config,
        'runtime': {
            'elapsed_sec': float(elapsed_sec),
            'n_training_runs_total': int(num_seeds * (2 + depth)),
            'n_sgd_training_runs': int(num_seeds),
            'n_muon_training_runs': int(num_seeds * (1 + depth)),
        },
        'seeds': seeds,
        'per_layer_seedwise': {
            'violation': {str(l): [float(v) for v in violations[l]] for l in range(depth)},
            'kappa_ratio': {str(l): [float(v) for v in kappa_ratios[l]] for l in range(depth)},
            'kappa_sgd': {str(l): [float(v) for v in kappa_sgd_values[l]] for l in range(depth)},
            'kappa_muon': {str(l): [float(v) for v in kappa_muon_values[l]] for l in range(depth)},
        },
        'violation_matrix': [[float(v) for v in row] for row in violation_matrix],
        'kappa_ratio_matrix': [[float(v) for v in row] for row in kappa_ratio_matrix],
        'layer_summaries': layer_summaries,
        'seed_results': seed_results,
        'aggregate_summaries': {
            'sgd_final_loss': _summary_stats(sgd_losses),
            'muon_final_loss': _summary_stats(muon_losses),
            'seedwise_pearson': _summary_stats(seedwise_pearsons),
            'seedwise_spearman': _summary_stats(seedwise_spearmans),
            'mean_violation': _summary_stats(mean_violations),
            'mean_kappa_ratio': _summary_stats(mean_kappa_ratios),
            'layers_with_mean_kappa_ratio_gt_1': layers_with_mean_kappa_ratio_gt_1,
            'fraction_layers_with_mean_kappa_ratio_gt_1': fraction_layers_with_mean_kappa_ratio_gt_1,
        },
        'correlations': {
            'pearson_layer_means': float(pearson_layer_means),
            'spearman_layer_means': float(spearman_layer_means['correlation']),
            'spearman_layer_means_pvalue': float(spearman_layer_means['pvalue']),
            'spearman_method': spearman_layer_means['method'],
            'seedwise_pearson_values': [float(v) for v in seedwise_pearsons],
            'seedwise_spearman_values': [float(v) for v in seedwise_spearmans],
        },
        'tests': tests,
        'verdict': {
            'supports_positive_relationship': supports_positive_relationship,
            'primary_statistic': 'pearson_layer_means',
            'summary': verdict_summary,
            'caution': 'The run is small and descriptive; it should not be treated as mechanistic proof or as a formal significance result.',
        },
    }
    return results


def print_results(results):
    config = results['config']
    correlations = results['correlations']
    losses = results['aggregate_summaries']

    print('=' * 100)
    print('2.1b-ii: EQUIVARIANCE VIOLATION vs CONDITION-NUMBER PROXY')
    print('=' * 100)
    print(
        f"Network: {config['depth']}-layer, {config['dim']}x{config['dim']}, "
        f"{config['num_steps']} steps, {config['num_seeds']} seeds"
    )
    print(
        f"Muon LR = {config['lr_muon']}, SGD LR = {config['lr_sgd']}, "
        f"Momentum = {config['momentum']}, NS iters = {config['ns_iters']}"
    )
    print('Advantage proxy: kappa_ratio = cond(W_sgd) / cond(W_muon)')
    print('Interpretation: kappa_ratio > 1 favors Muon on this conditioning proxy.')
    print(f"Primary question: {results['primary_question']}")
    print(
        'Headline correlation sample size: '
        f"{results['analysis_notes']['layer_point_count']} aggregated layer points"
    )
    print()

    print(f"Runtime: {results['runtime']['elapsed_sec']:.2f}s")
    print(
        'Training runs: '
        f"{results['runtime']['n_training_runs_total']} total "
        f"({results['runtime']['n_sgd_training_runs']} SGD, "
        f"{results['runtime']['n_muon_training_runs']} Muon)"
    )

    print(f"\n{'=' * 100}")
    print('PER-LAYER SUMMARY (mean ± std across seeds)')
    print(f"{'=' * 100}")
    print(
        f"\n  {'Layer':>5}  {'Violation':>24}  {'kappa ratio':>24}  {'Frac ratio>1':>12}"
    )
    print('  ' + '-' * 78)
    for row in results['layer_summaries']:
        print(
            f"  {row['layer']:>5}  "
            f"{row['violation_mean']:>10.4e} ± {row['violation_std']:<10.4e}  "
            f"{row['kappa_ratio_mean']:>10.3f} ± {row['kappa_ratio_std']:<10.3f}  "
            f"{row['muon_favored_fraction']:>12.2f}"
        )

    print(f"\n{'=' * 100}")
    print('AGGREGATE DIAGNOSTICS')
    print(f"{'=' * 100}")
    print(
        f"\n  Final loss (SGD):  {losses['sgd_final_loss']['mean']:.6e} "
        f"± {losses['sgd_final_loss']['std']:.3e}"
    )
    print(
        f"  Final loss (Muon): {losses['muon_final_loss']['mean']:.6e} "
        f"± {losses['muon_final_loss']['std']:.3e}"
    )
    print(
        f"  Layers with mean kappa_ratio > 1: "
        f"{losses['layers_with_mean_kappa_ratio_gt_1']} / {config['depth']}"
    )

    print(f"\n{'=' * 100}")
    print('CORRELATION SUMMARY')
    print(f"{'=' * 100}")
    print(f"\n  Pearson(layer means)  = {correlations['pearson_layer_means']:.3f}")
    print(
        f"  Spearman(layer means) = {correlations['spearman_layer_means']:.3f} "
        f"(p={correlations['spearman_layer_means_pvalue']:.3g}, "
        f"method={correlations['spearman_method']})"
    )
    print(
        f"  Seedwise Pearson mean ± std = "
        f"{losses['seedwise_pearson']['mean']:.3f} ± {losses['seedwise_pearson']['std']:.3f}"
    )
    print(
        f"  Seedwise Spearman mean ± std = "
        f"{losses['seedwise_spearman']['mean']:.3f} ± {losses['seedwise_spearman']['std']:.3f}"
    )

    print(f"\n{'=' * 100}")
    print('TOY VERDICT TESTS')
    print(f"{'=' * 100}")
    for test in results['tests']:
        print(f"\n  {test['id']}: {test['description']}")
        print(
            f"      {test['statistic_name']} = {test['statistic_value']:.3f}; "
            f"threshold {test['threshold']}"
        )
        print(f"      --> {'PASS' if test['pass'] else 'FAIL'}")

    print(f"\n{'=' * 100}")
    print('CALIBRATED CONCLUSION')
    print(f"{'=' * 100}")
    print(f"\n  {results['verdict']['summary']}")
    print(f"  Caution: {results['verdict']['caution']}")
    print(f"\n{'=' * 100}")
    print('EXPERIMENT COMPLETE')
    print(f"{'=' * 100}")


def main():
    results = run_experiment()
    print_results(results)
    return results


if __name__ == '__main__':
    main()
