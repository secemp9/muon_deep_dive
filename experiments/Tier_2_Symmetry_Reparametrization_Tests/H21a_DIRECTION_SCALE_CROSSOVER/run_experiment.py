#!/usr/bin/env python3
"""
H21a: Toy relative-performance crossover under prescribed layer imbalance.

This script compares:
- Muon with a single global learning rate and Newton-Schulz-normalized gradients
- Momentum SGD with oracle-tuned per-layer learning rates

on a synthetic deep-linear regression task family. For each imbalance factor c,
hyperparameters are selected on deterministic selection seeds and then reported on
fresh held-out evaluation seeds.

Primary metric: held-out mean final training loss ratio Muon / Oracle.

This is a toy optimizer-comparison study under controlled reparameterization. It is
useful as a relative-performance crossover test, but it is not a direct proof of a
full direction-vs-scale mechanism.
"""

import argparse
import copy
import itertools
import math

import numpy as np


DEFAULT_CONFIG = {
    'dim': 32,
    'n_layers': 4,
    'num_steps': 300,
    'momentum': 0.9,
    'ns_iters': 5,
    'num_selection_seeds': 3,
    'num_eval_seeds': 3,
    'batch_size': 64,
    'c_values': [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
    'muon_lr_grid': [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001],
    'per_layer_lr_grid': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'selection_seed_start': 42,
    'evaluation_seed_start': 10042,
    'seed_step': 137,
    'weight_seed_offset': 5000,
    'divergence_threshold': 1e10,
}


def get_default_config():
    return copy.deepcopy(DEFAULT_CONFIG)


def normalize_config(config=None):
    merged = get_default_config()
    if config is not None:
        for key, value in config.items():
            merged[key] = copy.deepcopy(value)

    merged['dim'] = int(merged['dim'])
    merged['n_layers'] = int(merged['n_layers'])
    merged['num_steps'] = int(merged['num_steps'])
    merged['momentum'] = float(merged['momentum'])
    merged['ns_iters'] = int(merged['ns_iters'])
    merged['num_selection_seeds'] = int(merged['num_selection_seeds'])
    merged['num_eval_seeds'] = int(merged['num_eval_seeds'])
    merged['batch_size'] = int(merged['batch_size'])
    merged['c_values'] = [float(x) for x in merged['c_values']]
    merged['muon_lr_grid'] = [float(x) for x in merged['muon_lr_grid']]
    merged['per_layer_lr_grid'] = [float(x) for x in merged['per_layer_lr_grid']]
    merged['selection_seed_start'] = int(merged['selection_seed_start'])
    merged['evaluation_seed_start'] = int(merged['evaluation_seed_start'])
    merged['seed_step'] = int(merged['seed_step'])
    merged['weight_seed_offset'] = int(merged['weight_seed_offset'])
    merged['divergence_threshold'] = float(merged['divergence_threshold'])
    return merged


def make_smoke_config(base_config=None):
    config = normalize_config(base_config)
    config.update({
        'dim': 16,
        'num_steps': 40,
        'num_selection_seeds': 2,
        'num_eval_seeds': 2,
        'batch_size': 32,
        'c_values': [1.0, 10.0],
        'muon_lr_grid': [0.03, 0.01],
        'per_layer_lr_grid': [0.1, 0.001],
    })
    return normalize_config(config)


def make_seed_list(start, step, count):
    return [int(start + i * step) for i in range(count)]


def get_selection_seeds(config):
    return make_seed_list(config['selection_seed_start'], config['seed_step'], config['num_selection_seeds'])


def get_evaluation_seeds(config):
    return make_seed_list(config['evaluation_seed_start'], config['seed_step'], config['num_eval_seeds'])


def estimate_workload(config):
    config = normalize_config(config)
    oracle_grid_size = len(config['per_layer_lr_grid']) ** config['n_layers']
    muon_grid_size = len(config['muon_lr_grid'])
    selection_train_runs_per_c = config['num_selection_seeds'] * (muon_grid_size + oracle_grid_size)
    heldout_train_runs_per_c = config['num_eval_seeds'] * 2
    total_per_c = selection_train_runs_per_c + heldout_train_runs_per_c
    return {
        'num_c_values': len(config['c_values']),
        'muon_grid_size': muon_grid_size,
        'oracle_grid_size': oracle_grid_size,
        'selection_train_runs_per_c': selection_train_runs_per_c,
        'heldout_train_runs_per_c': heldout_train_runs_per_c,
        'total_train_runs_per_c': total_per_c,
        'total_train_runs': len(config['c_values']) * total_per_c,
    }


def format_loss(value):
    if value is None:
        return 'None'
    if np.isnan(value):
        return 'nan'
    if np.isposinf(value):
        return 'inf'
    if np.isneginf(value):
        return '-inf'
    return f'{value:.4e}'


def format_lr_list(values):
    return '[' + ', '.join(f'{float(v):.0e}' for v in values) + ']'


def newton_schulz(M, n_iters):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed, c, config):
    rng = np.random.RandomState(seed)
    dim = config['dim']
    n_layers = config['n_layers']
    weights = [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(n_layers)]
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
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def compute_gradients(weights, X, Y):
    n_layers = len(weights)
    batch_size = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / batch_size
    grads = [None] * n_layers
    for layer_idx in range(n_layers - 1, -1, -1):
        grads[layer_idx] = delta @ acts[layer_idx].T
        if layer_idx > 0:
            delta = weights[layer_idx].T @ delta
    return grads


def make_data(seed, config):
    rng = np.random.RandomState(seed)
    dim = config['dim']
    batch_size = config['batch_size']
    W_target = rng.randn(dim, dim) * 0.5
    X = rng.randn(dim, batch_size) * 0.3
    Y = W_target @ X
    return X, Y


def train_muon(w0, X, Y, lr, config):
    weights = [W.copy() for W in w0]
    momentum_buffers = [np.zeros_like(W) for W in weights]
    for _ in range(config['num_steps']):
        if compute_loss(weights, X, Y) > config['divergence_threshold']:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            momentum_buffers[i] = (
                config['momentum'] * momentum_buffers[i] +
                newton_schulz(grads[i], n_iters=config['ns_iters'])
            )
            weights[i] -= lr * momentum_buffers[i]
    final_loss = compute_loss(weights, X, Y)
    return float(final_loss) if np.isfinite(final_loss) else float('inf')


def train_sgd_per_layer(w0, X, Y, per_layer_lrs, config):
    weights = [W.copy() for W in w0]
    momentum_buffers = [np.zeros_like(W) for W in weights]
    for _ in range(config['num_steps']):
        if compute_loss(weights, X, Y) > config['divergence_threshold']:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            momentum_buffers[i] = config['momentum'] * momentum_buffers[i] + grads[i]
            weights[i] -= per_layer_lrs[i] * momentum_buffers[i]
    final_loss = compute_loss(weights, X, Y)
    return float(final_loss) if np.isfinite(final_loss) else float('inf')


def summarize_losses(losses, seeds):
    cleaned_losses = []
    finite_losses = []
    for loss in losses:
        if np.isfinite(loss):
            value = float(loss)
            finite_losses.append(value)
        else:
            value = float('inf')
        cleaned_losses.append(value)
    mean_finite_loss = float(np.mean(finite_losses)) if finite_losses else float('inf')
    return {
        'seeds': [int(seed) for seed in seeds],
        'losses': cleaned_losses,
        'finite_count': int(len(finite_losses)),
        'diverged_count': int(len(cleaned_losses) - len(finite_losses)),
        'mean_finite_loss': mean_finite_loss,
        'all_finite': bool(len(finite_losses) == len(cleaned_losses)),
    }


def candidate_sort_key(candidate):
    return (candidate['diverged_count'], candidate['mean_finite_loss'])


def evaluate_muon_on_seeds(seeds, c, lr, config):
    losses = []
    for seed in seeds:
        X, Y = make_data(seed, config)
        weights = init_weights(seed + config['weight_seed_offset'], c, config)
        losses.append(train_muon(weights, X, Y, lr, config))
    return summarize_losses(losses, seeds)


def evaluate_oracle_on_seeds(seeds, c, per_layer_lrs, config):
    losses = []
    for seed in seeds:
        X, Y = make_data(seed, config)
        weights = init_weights(seed + config['weight_seed_offset'], c, config)
        losses.append(train_sgd_per_layer(weights, X, Y, per_layer_lrs, config))
    return summarize_losses(losses, seeds)


def sweep_muon(selection_seeds, c, config):
    candidates = []
    best_candidate = None
    for lr in config['muon_lr_grid']:
        summary = evaluate_muon_on_seeds(selection_seeds, c, lr, config)
        candidate = {
            'lr': float(lr),
            **summary,
        }
        candidates.append(candidate)
        if best_candidate is None or candidate_sort_key(candidate) < candidate_sort_key(best_candidate):
            best_candidate = copy.deepcopy(candidate)
    return {
        'grid_size': len(config['muon_lr_grid']),
        'best_candidate': best_candidate,
        'candidates': candidates,
    }


def sweep_oracle_per_layer(selection_seeds, c, config):
    candidates = []
    best_candidate = None
    for combo in itertools.product(config['per_layer_lr_grid'], repeat=config['n_layers']):
        lrs = [float(lr) for lr in combo]
        summary = evaluate_oracle_on_seeds(selection_seeds, c, lrs, config)
        candidate = {
            'lrs': lrs,
            **summary,
        }
        candidates.append(candidate)
        if best_candidate is None or candidate_sort_key(candidate) < candidate_sort_key(best_candidate):
            best_candidate = copy.deepcopy(candidate)
    return {
        'grid_size': len(config['per_layer_lr_grid']) ** config['n_layers'],
        'best_candidate': best_candidate,
        'candidates': candidates,
    }


def compute_ratio(muon_mean, oracle_mean):
    muon_finite = np.isfinite(muon_mean)
    oracle_finite = np.isfinite(oracle_mean)
    if not muon_finite and not oracle_finite:
        return float('nan')
    if not muon_finite and oracle_finite:
        return float('inf')
    if muon_finite and not oracle_finite:
        return 0.0
    return float(muon_mean / max(oracle_mean, 1e-30))


def describe_regime(ratio):
    if np.isnan(ratio):
        return 'Both methods diverged on held-out evaluation'
    if ratio < 1.0:
        return 'Muon lower held-out loss'
    if ratio > 1.0:
        return 'Oracle lower held-out loss'
    return 'Held-out tie at tested precision'


def interpolate_crossover_log_segment(left_c, left_ratio, right_c, right_ratio):
    if not (np.isfinite(left_ratio) and np.isfinite(right_ratio)):
        return None
    if left_ratio <= 0 or right_ratio <= 0:
        return None
    if np.isclose(left_ratio, 1.0):
        return float(left_c)
    if np.isclose(right_ratio, 1.0):
        return float(right_c)
    log_r1 = math.log10(left_ratio)
    log_r2 = math.log10(right_ratio)
    if np.isclose(log_r1, log_r2):
        return None
    log_c1 = math.log10(left_c)
    log_c2 = math.log10(right_c)
    weight = (0.0 - log_r1) / (log_r2 - log_r1)
    log_c_star = log_c1 + weight * (log_c2 - log_c1)
    return float(10 ** log_c_star)


def estimate_crossover(per_c_results):
    ratios = [entry['ratio'] for entry in per_c_results]
    c_values = [entry['c'] for entry in per_c_results]

    exact_grid_hits = []
    brackets = []

    for c, ratio in zip(c_values, ratios):
        if np.isfinite(ratio) and np.isclose(ratio, 1.0):
            exact_grid_hits.append({'c': float(c), 'ratio': float(ratio)})

    for idx in range(len(per_c_results) - 1):
        left = per_c_results[idx]
        right = per_c_results[idx + 1]
        left_ratio = left['ratio']
        right_ratio = right['ratio']
        if not (np.isfinite(left_ratio) and np.isfinite(right_ratio)):
            continue
        if (left_ratio < 1.0 <= right_ratio) or (left_ratio > 1.0 >= right_ratio):
            brackets.append({
                'left_c': float(left['c']),
                'right_c': float(right['c']),
                'left_ratio': float(left_ratio),
                'right_ratio': float(right_ratio),
                'interpolated_c_star': interpolate_crossover_log_segment(
                    left['c'], left_ratio, right['c'], right_ratio
                ),
            })

    finite_ratios = [ratio for ratio in ratios if np.isfinite(ratio)]
    below = [ratio for ratio in finite_ratios if ratio < 1.0]
    above = [ratio for ratio in finite_ratios if ratio > 1.0]

    if exact_grid_hits:
        status = 'exact_grid_hit'
    elif brackets:
        status = 'bracketed'
    elif finite_ratios and not above:
        status = 'muon_better_all_tested'
    elif finite_ratios and not below:
        status = 'oracle_better_all_tested'
    elif not finite_ratios:
        status = 'no_finite_ratio'
    else:
        status = 'indeterminate'

    return {
        'status': status,
        'exact_grid_hits': exact_grid_hits,
        'observed_brackets': brackets,
        'primary_bracket': brackets[0] if brackets else None,
    }


def fit_loglog_trend(per_c_results):
    c_values = np.array([entry['c'] for entry in per_c_results], dtype=float)
    ratios = np.array([entry['ratio'] for entry in per_c_results], dtype=float)
    mask = np.isfinite(ratios) & (ratios > 0)
    if np.count_nonzero(mask) < 3:
        return None

    log_c = np.log10(c_values[mask])
    log_ratio = np.log10(ratios[mask])
    slope, intercept = np.polyfit(log_c, log_ratio, 1)
    predictions = slope * log_c + intercept
    ss_res = float(np.sum((log_ratio - predictions) ** 2))
    ss_tot = float(np.sum((log_ratio - np.mean(log_ratio)) ** 2))
    r_squared = 1.0 if np.isclose(ss_tot, 0.0) else float(1.0 - ss_res / ss_tot)

    extrapolated_c_star = None
    if slope > 0:
        extrapolated_c_star = float(10 ** (-intercept / slope))

    return {
        'num_points': int(np.count_nonzero(mask)),
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': r_squared,
        'extrapolated_c_star': extrapolated_c_star,
    }


def build_summary_rows(results):
    rows = []
    for entry in results['per_c_results']:
        muon_selection = entry['selection']['muon']['best_candidate']
        oracle_selection = entry['selection']['oracle']['best_candidate']
        muon_eval = entry['evaluation']['muon']
        oracle_eval = entry['evaluation']['oracle']
        ratio = entry['ratio']
        rows.append({
            'c': float(entry['c']),
            'muon_best_lr': float(muon_selection['lr']),
            'oracle_best_lrs': [float(lr) for lr in oracle_selection['lrs']],
            'muon_selection_mean_finite_loss': float(muon_selection['mean_finite_loss']),
            'oracle_selection_mean_finite_loss': float(oracle_selection['mean_finite_loss']),
            'muon_selection_finite_count': int(muon_selection['finite_count']),
            'oracle_selection_finite_count': int(oracle_selection['finite_count']),
            'muon_selection_diverged': int(muon_selection['diverged_count']),
            'oracle_selection_diverged': int(oracle_selection['diverged_count']),
            'muon_eval_mean_finite_loss': float(muon_eval['mean_finite_loss']),
            'oracle_eval_mean_finite_loss': float(oracle_eval['mean_finite_loss']),
            'muon_eval_losses': [float(x) for x in muon_eval['losses']],
            'oracle_eval_losses': [float(x) for x in oracle_eval['losses']],
            'muon_eval_finite_count': int(muon_eval['finite_count']),
            'oracle_eval_finite_count': int(oracle_eval['finite_count']),
            'muon_eval_diverged': int(muon_eval['diverged_count']),
            'oracle_eval_diverged': int(oracle_eval['diverged_count']),
            'ratio': float(ratio) if np.isfinite(ratio) else ratio,
            'regime': entry['regime'],
        })
    return rows


def run_crossover_experiment(config=None, verbose=False):
    config = normalize_config(config)
    selection_seeds = get_selection_seeds(config)
    evaluation_seeds = get_evaluation_seeds(config)
    workload = estimate_workload(config)

    if verbose:
        print('=' * 100)
        print('H21a: TOY RELATIVE-PERFORMANCE CROSSOVER UNDER PRESCRIBED IMBALANCE')
        print('=' * 100)
        print(f"Network: {config['n_layers']}-layer deep linear {config['dim']}x{config['dim']}")
        print(f"c values: {config['c_values']}")
        print(f"Steps: {config['num_steps']}, batch size: {config['batch_size']}")
        print(f"Selection seeds: {selection_seeds}")
        print(f"Held-out evaluation seeds: {evaluation_seeds}")
        print(
            f"Muon grid: {workload['muon_grid_size']} candidates | "
            f"Oracle grid: {workload['oracle_grid_size']} candidates | "
            f"Total train runs: {workload['total_train_runs']}"
        )
        print('Primary metric: held-out mean final loss ratio Muon / Oracle')

    per_c_results = []

    for c in config['c_values']:
        if verbose:
            print(f"\n[c={c:.0f}] selection on seeds {selection_seeds}")

        muon_sweep = sweep_muon(selection_seeds, c, config)
        oracle_sweep = sweep_oracle_per_layer(selection_seeds, c, config)

        best_muon = muon_sweep['best_candidate']
        best_oracle = oracle_sweep['best_candidate']

        muon_eval = evaluate_muon_on_seeds(evaluation_seeds, c, best_muon['lr'], config)
        oracle_eval = evaluate_oracle_on_seeds(evaluation_seeds, c, best_oracle['lrs'], config)

        muon_mean = muon_eval['mean_finite_loss']
        oracle_mean = oracle_eval['mean_finite_loss']
        ratio = compute_ratio(muon_mean, oracle_mean)
        regime = describe_regime(ratio)

        entry = {
            'c': float(c),
            'selection': {
                'muon': muon_sweep,
                'oracle': oracle_sweep,
            },
            'evaluation': {
                'muon': muon_eval,
                'oracle': oracle_eval,
            },
            'ratio': ratio,
            'regime': regime,
        }
        per_c_results.append(entry)

        if verbose:
            print(
                f"  Muon best LR={best_muon['lr']:.3g}, selection div={best_muon['diverged_count']}, "
                f"selection mean={format_loss(best_muon['mean_finite_loss'])}"
            )
            print(
                f"  Oracle best LRs={format_lr_list(best_oracle['lrs'])}, "
                f"selection div={best_oracle['diverged_count']}, "
                f"selection mean={format_loss(best_oracle['mean_finite_loss'])}"
            )
            print(
                f"  Held-out Muon={format_loss(muon_mean)} "
                f"(finite={muon_eval['finite_count']}, div={muon_eval['diverged_count']}) | "
                f"Oracle={format_loss(oracle_mean)} "
                f"(finite={oracle_eval['finite_count']}, div={oracle_eval['diverged_count']}) | "
                f"ratio={ratio:.4g}"
            )

    results = {
        'config': config,
        'selection_seeds': selection_seeds,
        'evaluation_seeds': evaluation_seeds,
        'workload': workload,
        'per_c_results': per_c_results,
    }
    results['summary_rows'] = build_summary_rows(results)
    results['crossover'] = estimate_crossover(per_c_results)
    results['trend_fit'] = fit_loglog_trend(per_c_results)
    return results


def summarize_results(results):
    config = results['config']
    workload = results['workload']
    summary_rows = results['summary_rows']

    print(f"\n\n{'=' * 100}")
    print('SUMMARY: TOY RELATIVE-PERFORMANCE CROSSOVER UNDER PRESCRIBED IMBALANCE')
    print(f"{'=' * 100}")
    print(f"Selection seeds: {results['selection_seeds']}")
    print(f"Held-out evaluation seeds: {results['evaluation_seeds']}")
    print(
        f"Grid sizes: Muon={workload['muon_grid_size']} | "
        f"Oracle={workload['oracle_grid_size']} | "
        f"Total train runs={workload['total_train_runs']}"
    )
    print(
        f"Default objective per c: select on {config['num_selection_seeds']} seeds, "
        f"report on {config['num_eval_seeds']} fresh held-out seeds"
    )
    print()
    print(
        f"{'c':>6}  {'Muon held-out':>14}  {'Oracle held-out':>14}  {'Ratio':>10}  "
        f"{'Muon LR':>8}  {'Oracle best LRs':>28}  {'sel div M/O':>11}  {'eval div M/O':>12}"
    )
    print('  ' + '-' * 125)
    for row in summary_rows:
        print(
            f"  {row['c']:>6.0f}  {format_loss(row['muon_eval_mean_finite_loss']):>14}  "
            f"{format_loss(row['oracle_eval_mean_finite_loss']):>14}  {row['ratio']:>10.4g}  "
            f"{row['muon_best_lr']:>8.3g}  {format_lr_list(row['oracle_best_lrs']):>28}  "
            f"{row['muon_selection_diverged']}/{row['oracle_selection_diverged']:>3}  "
            f"{row['muon_eval_diverged']}/{row['oracle_eval_diverged']:>3}"
        )

    crossover = results['crossover']
    print('\nCrossover verdict:')
    if crossover['status'] == 'exact_grid_hit':
        exact_cs = ', '.join(f"c={item['c']:.0f}" for item in crossover['exact_grid_hits'])
        print(f"  Exact grid hit(s) at ratio≈1 observed at {exact_cs}.")
    elif crossover['status'] == 'bracketed':
        primary = crossover['primary_bracket']
        print(
            f"  Observed in-range bracket between c={primary['left_c']:.0f} and c={primary['right_c']:.0f} "
            f"(ratio {primary['left_ratio']:.4g} -> {primary['right_ratio']:.4g})."
        )
        if primary['interpolated_c_star'] is not None:
            print(
                f"  Log-segment interpolation within that bracket gives c≈{primary['interpolated_c_star']:.3g}."
            )
        if len(crossover['observed_brackets']) > 1:
            print(f"  Note: multiple sign-change brackets were observed ({len(crossover['observed_brackets'])} total).")
    elif crossover['status'] == 'muon_better_all_tested':
        print(
            f"  Muon had lower held-out loss for all tested c values; "
            f"any crossover would lie above c={config['c_values'][-1]:.0f}."
        )
    elif crossover['status'] == 'oracle_better_all_tested':
        print(
            f"  Oracle had lower held-out loss for all tested c values; "
            f"any crossover would lie below c={config['c_values'][0]:.0f}."
        )
    elif crossover['status'] == 'no_finite_ratio':
        print('  No finite held-out ratio was available.')
    else:
        print('  No clean adjacent bracket could be stated from the tested grid.')

    trend_fit = results['trend_fit']
    print('\nExploratory log-log trend fit:')
    if trend_fit is None:
        print('  Not enough positive finite ratios for a log-log fit.')
    else:
        print(
            f"  log10(ratio) = {trend_fit['slope']:.3f} * log10(c) + {trend_fit['intercept']:.3f} "
            f"(R^2={trend_fit['r_squared']:.3f}, points={trend_fit['num_points']})"
        )
        if trend_fit['extrapolated_c_star'] is not None:
            print(f"  Positive-slope extrapolated c where ratio≈1: {trend_fit['extrapolated_c_star']:.3g}")
        else:
            print('  No positive-slope crossover extrapolation available from the global fit.')

    print('\nLimitations:')
    print('  - synthetic deep-linear toy task')
    print('  - coarse LR grids and coarse c grid')
    print('  - small seed counts due exhaustive oracle grid cost')
    print('  - final training loss comparison, not a direct mechanism decomposition')
    print(f"\n{'=' * 100}")
    print('EXPERIMENT COMPLETE')
    print(f"{'=' * 100}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the H21a toy relative-performance crossover experiment.'
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run a reduced configuration for quick path/sanity checking.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-c progress logging and only print the final summary.',
    )
    args = parser.parse_args(argv)

    config = make_smoke_config() if args.smoke else get_default_config()
    results = run_crossover_experiment(config=config, verbose=not args.quiet)
    summarize_results(results)
    return results


if __name__ == '__main__':
    main()
