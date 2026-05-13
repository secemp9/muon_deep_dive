#!/usr/bin/env python3
"""
H15b: Explicit SVD Clamping vs. Muon -- final-loss benchmark under post-step spectral controls
===========================================================================================

MOTIVATION (from H15 surprise):
  Matrix layers under Muon sometimes have WORSE kappa than SGD despite much better loss.
  If conditioning improvement is not the mechanism, then explicitly forcing better spectral
  conditioning after each SGD step should not reproduce Muon's final-loss behavior.

QUESTION:
  If we add explicit post-step SVD conditioning controls to SGD (clamp sigma_max/sigma_min
  to a target kappa, or equalize all singular values), do those controls match Muon's final
  loss after 500 steps on the same task?

SCOPE / LIMITATION:
  H15b measures final-loss behavior under these post-step spectral controls. It does not
  directly measure update-direction quality, and it does not by itself establish broader
  mechanistic claims beyond this setup.

PROTOCOL:
  Optimizers:
    (a) SGD -- baseline
    (b) Muon -- polar-factor / Newton-Schulz transformed gradient
    (c) SGD + SVD clamping -- after each SGD step, clamp SVs so kappa(W) <= target_kappa
    (d) SGD + SVD equalize -- after each SGD step, set all SVs to mean(S)
  Sweep target_kappa for (c) in {2, 5, 10, 50}.
  Select the best learning rate on a fixed grid using the first 3 seeds only.

KEY TESTS:
  T1: Does SGD + SVD clamping (kappa<=5) reach mean final loss within 2x of Muon?
  T2: Does SGD + SVD equalization reach mean final loss within 2x of Muon?
  T3: If both fail, then conditioning-only post-step spectral controls are insufficient
      to reproduce Muon under this setup and metric.

Default setup: 4-layer, 32x32, 500 steps, 10 seeds, LR swept per method.
"""

import argparse
import os
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
BATCH_SIZE = 64
DIVERGENCE_THRESHOLD = 1e10
SEED_START = 42
SEED_STRIDE = 137
WEIGHT_SEED_OFFSET = 5000
DATA_SCALE = 0.3
INIT_SCALE = 0.1
LR_SWEEP_NUM_SEEDS = 3

LR_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
KAPPA_TARGETS = [2, 5, 10, 50]

DEFAULT_CONFIG = {
    'dim': DIM,
    'num_layers': NUM_LAYERS,
    'num_steps': NUM_STEPS,
    'momentum': MOMENTUM,
    'ns_iters': NS_ITERS,
    'num_seeds': NUM_SEEDS,
    'batch_size': BATCH_SIZE,
    'divergence_threshold': DIVERGENCE_THRESHOLD,
    'seed_start': SEED_START,
    'seed_stride': SEED_STRIDE,
    'weight_seed_offset': WEIGHT_SEED_OFFSET,
    'data_scale': DATA_SCALE,
    'init_scale': INIT_SCALE,
    'lr_sweep_num_seeds': LR_SWEEP_NUM_SEEDS,
    'lr_candidates': list(LR_CANDIDATES),
    'kappa_targets': list(KAPPA_TARGETS),
}

SMOKE_CONFIG_OVERRIDES = {
    'num_steps': 30,
    'num_seeds': 4,
    'lr_sweep_num_seeds': 2,
    'lr_candidates': [0.03, 0.01, 0.003],
}


def make_config(overrides=None):
    config = {
        key: (value.copy() if isinstance(value, list) else value)
        for key, value in DEFAULT_CONFIG.items()
    }
    if overrides:
        for key, value in overrides.items():
            config[key] = value.copy() if isinstance(value, list) else value
    if config['lr_sweep_num_seeds'] > config['num_seeds']:
        raise ValueError('lr_sweep_num_seeds cannot exceed num_seeds')
    return config


def build_method_configs(config=None):
    config = make_config(config) if config is not None and config is not DEFAULT_CONFIG else (DEFAULT_CONFIG if config is None else config)
    configs = [
        {
            'name': 'sgd',
            'method': 'sgd',
            'kappa_target': None,
            'label': 'SGD',
        },
        {
            'name': 'muon',
            'method': 'muon',
            'kappa_target': None,
            'label': 'Muon',
        },
        {
            'name': 'sgd_equalize',
            'method': 'sgd_equalize',
            'kappa_target': None,
            'label': 'SGD + SVD equalize',
        },
    ]
    for kt in config['kappa_targets']:
        configs.append(
            {
                'name': f'sgd_clamp_k{kt}',
                'method': 'sgd_clamp',
                'kappa_target': kt,
                'label': f'SGD + SVD clamp (kappa<={kt})',
            }
        )
    return configs


def generate_seeds(config=None):
    config = DEFAULT_CONFIG if config is None else config
    return [config['seed_start'] + i * config['seed_stride'] for i in range(config['num_seeds'])]


def estimate_workload(config=None):
    config = DEFAULT_CONFIG if config is None else config
    method_configs = build_method_configs(config)
    num_methods = len(method_configs)
    num_lr = len(config['lr_candidates'])
    num_layers = config['num_layers']
    num_steps = config['num_steps']

    lr_sweep_train_runs = num_methods * num_lr * config['lr_sweep_num_seeds']
    full_phase_train_runs = num_methods * config['num_seeds']
    total_train_runs = lr_sweep_train_runs + full_phase_train_runs

    svd_method_count = 1 + len(config['kappa_targets'])
    muon_method_count = 1
    projections_per_run = num_steps * num_layers

    total_svd_projection_calls = svd_method_count * (
        num_lr * config['lr_sweep_num_seeds'] + config['num_seeds']
    ) * projections_per_run
    total_muon_projection_calls = muon_method_count * (
        num_lr * config['lr_sweep_num_seeds'] + config['num_seeds']
    ) * projections_per_run

    return {
        'num_methods': num_methods,
        'num_lr_candidates': num_lr,
        'lr_sweep_train_runs': lr_sweep_train_runs,
        'full_phase_train_runs': full_phase_train_runs,
        'total_train_runs': total_train_runs,
        'projection_calls_per_train_run': projections_per_run,
        'total_svd_projection_calls': total_svd_projection_calls,
        'total_muon_projection_calls': total_muon_projection_calls,
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


def svd_clamp(W, target_kappa):
    """Clamp singular values so kappa(W) <= target_kappa."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s_max = s[0]
    s_min_target = s_max / target_kappa
    s_clamped = np.maximum(s, s_min_target)
    return U @ np.diag(s_clamped) @ Vt


def svd_equalize(W):
    """Set all singular values to their mean."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s_eq = np.full_like(s, np.mean(s))
    return U @ np.diag(s_eq) @ Vt


def init_weights(seed, config=None):
    config = DEFAULT_CONFIG if config is None else config
    rng = np.random.RandomState(seed)
    return [
        np.eye(config['dim']) + rng.randn(config['dim'], config['dim']) * config['init_scale']
        for _ in range(config['num_layers'])
    ]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


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


def matrix_condition_number(W):
    singular_values = np.linalg.svd(W, compute_uv=False)
    return float(singular_values[0] / max(singular_values[-1], 1e-30))


def measure_kappas(weights):
    layer_kappas = [matrix_condition_number(W) for W in weights]
    product = np.eye(weights[0].shape[0])
    for W in weights:
        product = W @ product
    product_kappa = matrix_condition_number(product)
    return {
        'layer_kappas': [float(k) for k in layer_kappas],
        'product_kappa': float(product_kappa),
    }


def train(
    weights_init,
    X,
    Y,
    lr,
    method,
    kappa_target=None,
    config=None,
    return_details=False,
    record_history=False,
    record_conditioning=False,
):
    config = DEFAULT_CONFIG if config is None else config
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    history = [] if record_history else None

    for step in range(config['num_steps']):
        loss = compute_loss(weights, X, Y)
        if record_history:
            history.append(float(loss))
        if not np.isfinite(loss) or loss > config['divergence_threshold']:
            if record_history:
                history.append(float('inf'))
            details = {
                'final_loss': float('inf'),
                'converged': False,
                'termination_reason': 'diverged_before_update',
                'steps_completed': step,
                'loss_history': history,
                'conditioning': None,
            }
            return details if return_details else float('inf')

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if method == 'muon':
                mom[i] = config['momentum'] * mom[i] + newton_schulz(grads[i], n_iters=config['ns_iters'])
                weights[i] = weights[i] - lr * mom[i]
            elif method == 'sgd':
                mom[i] = config['momentum'] * mom[i] + grads[i]
                weights[i] = weights[i] - lr * mom[i]
            elif method == 'sgd_clamp':
                mom[i] = config['momentum'] * mom[i] + grads[i]
                weights[i] = weights[i] - lr * mom[i]
                weights[i] = svd_clamp(weights[i], kappa_target)
            elif method == 'sgd_equalize':
                mom[i] = config['momentum'] * mom[i] + grads[i]
                weights[i] = weights[i] - lr * mom[i]
                weights[i] = svd_equalize(weights[i])
            else:
                raise ValueError(f'Unknown method: {method}')

    final_loss = float(compute_loss(weights, X, Y))
    if record_history:
        history.append(final_loss)

    details = {
        'final_loss': final_loss,
        'converged': bool(np.isfinite(final_loss)),
        'termination_reason': 'completed',
        'steps_completed': config['num_steps'],
        'loss_history': history,
        'conditioning': measure_kappas(weights) if record_conditioning else None,
    }
    return details if return_details else final_loss


def make_data(seed, config=None):
    config = DEFAULT_CONFIG if config is None else config
    rng = np.random.RandomState(seed)
    X = rng.randn(config['dim'], config['batch_size']) * config['data_scale']
    Y = rng.randn(config['dim'], config['batch_size']) * config['data_scale']
    return X, Y


def sweep_lr(method, seeds, kappa_target=None, config=None):
    config = DEFAULT_CONFIG if config is None else config
    best_lr = config['lr_candidates'][-1]
    best_loss = float('inf')
    candidate_results = []

    for lr in config['lr_candidates']:
        losses = []
        for seed in seeds:
            X, Y = make_data(seed, config=config)
            weights = init_weights(seed + config['weight_seed_offset'], config=config)
            final_loss = train(weights, X, Y, lr, method, kappa_target, config=config)
            losses.append(float(final_loss))

        finite_losses = [loss for loss in losses if np.isfinite(loss)]
        mean_loss = float(np.mean(finite_losses)) if finite_losses else float('inf')
        candidate_results.append(
            {
                'lr': float(lr),
                'seed_losses': losses,
                'mean_final_loss_finite': mean_loss,
                'num_finite': int(len(finite_losses)),
                'num_total': int(len(losses)),
            }
        )
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_lr = lr

    return {
        'method': method,
        'kappa_target': kappa_target,
        'sweep_seeds': list(seeds),
        'candidate_results': candidate_results,
        'best_lr': float(best_lr),
        'best_mean_final_loss_finite': float(best_loss),
    }


def _finite_array(values):
    return np.asarray([value for value in values if np.isfinite(value)], dtype=float)


def summarize_seed_results(seed_results, num_layers):
    losses = [seed_result['final_loss'] for seed_result in seed_results]
    finite_losses = _finite_array(losses)
    num_finite = int(finite_losses.size)
    num_total = int(len(seed_results))

    if num_finite:
        mean_loss = float(np.mean(finite_losses))
        std_loss = float(np.std(finite_losses))
        sem_loss = float(np.std(finite_losses) / np.sqrt(num_finite))
        min_loss = float(np.min(finite_losses))
        max_loss = float(np.max(finite_losses))
    else:
        mean_loss = float('inf')
        std_loss = float('nan')
        sem_loss = float('nan')
        min_loss = float('nan')
        max_loss = float('nan')

    conditioning = [
        seed_result['conditioning']
        for seed_result in seed_results
        if seed_result['conditioning'] is not None and np.isfinite(seed_result['final_loss'])
    ]
    if conditioning:
        layer_kappas = np.asarray([entry['layer_kappas'] for entry in conditioning], dtype=float)
        product_kappas = np.asarray([entry['product_kappa'] for entry in conditioning], dtype=float)
        mean_layer_kappas = np.mean(layer_kappas, axis=0).astype(float).tolist()
        std_layer_kappas = np.std(layer_kappas, axis=0).astype(float).tolist()
        mean_product_kappa = float(np.mean(product_kappas))
        std_product_kappa = float(np.std(product_kappas))
    else:
        mean_layer_kappas = [float('nan')] * num_layers
        std_layer_kappas = [float('nan')] * num_layers
        mean_product_kappa = float('nan')
        std_product_kappa = float('nan')

    return {
        'mean_final_loss_finite': mean_loss,
        'std_final_loss_finite': std_loss,
        'sem_final_loss_finite': sem_loss,
        'min_final_loss_finite': min_loss,
        'max_final_loss_finite': max_loss,
        'num_finite': num_finite,
        'num_total': num_total,
        'num_diverged': num_total - num_finite,
        'mean_final_layer_kappas': mean_layer_kappas,
        'std_final_layer_kappas': std_layer_kappas,
        'mean_final_product_kappa': mean_product_kappa,
        'std_final_product_kappa': std_product_kappa,
    }


def _safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float('inf')
    return float(numerator / max(denominator, 1e-30))


def make_summary_rows(results):
    muon_loss = results['full_phase']['muon']['summary']['mean_final_loss_finite']
    sgd_loss = results['full_phase']['sgd']['summary']['mean_final_loss_finite']
    rows = []
    for method_config in results['method_configs']:
        name = method_config['name']
        summary = results['full_phase'][name]['summary']
        rows.append(
            {
                'name': name,
                'label': method_config['label'],
                'method': method_config['method'],
                'kappa_target': method_config['kappa_target'],
                'best_lr_in_grid': results['best_lrs'][name],
                'mean_final_loss': summary['mean_final_loss_finite'],
                'std_final_loss': summary['std_final_loss_finite'],
                'sem_final_loss': summary['sem_final_loss_finite'],
                'min_final_loss': summary['min_final_loss_finite'],
                'max_final_loss': summary['max_final_loss_finite'],
                'num_converged': summary['num_finite'],
                'num_total': summary['num_total'],
                'ratio_vs_muon': _safe_ratio(summary['mean_final_loss_finite'], muon_loss),
                'ratio_vs_sgd': _safe_ratio(summary['mean_final_loss_finite'], sgd_loss),
                'mean_final_product_kappa': summary['mean_final_product_kappa'],
                'mean_final_layer_kappas': summary['mean_final_layer_kappas'],
            }
        )
    return rows


def make_lr_sweep_rows(results):
    rows = []
    for method_config in results['method_configs']:
        name = method_config['name']
        sweep = results['lr_sweep'][name]
        best_lr = sweep['best_lr']
        for candidate in sweep['candidate_results']:
            rows.append(
                {
                    'name': name,
                    'label': method_config['label'],
                    'method': method_config['method'],
                    'kappa_target': method_config['kappa_target'],
                    'lr': candidate['lr'],
                    'selected_best_lr_in_grid': abs(candidate['lr'] - best_lr) < 1e-15,
                    'mean_final_loss_finite': candidate['mean_final_loss_finite'],
                    'num_finite': candidate['num_finite'],
                    'num_total': candidate['num_total'],
                    'seed_losses': candidate['seed_losses'],
                }
            )
    return rows


def evaluate_tests(results):
    muon_loss = results['full_phase']['muon']['summary']['mean_final_loss_finite']
    clamp5_loss = results['full_phase'].get('sgd_clamp_k5', {}).get('summary', {}).get('mean_final_loss_finite', float('inf'))
    equalize_loss = results['full_phase']['sgd_equalize']['summary']['mean_final_loss_finite']

    t1_ratio = _safe_ratio(clamp5_loss, muon_loss)
    t2_ratio = _safe_ratio(equalize_loss, muon_loss)
    t1_pass = bool(t1_ratio < 2.0)
    t2_pass = bool(t2_ratio < 2.0)
    t3_pass = bool((not t1_pass) and (not t2_pass))

    return {
        'T1': {
            'description': 'Does SGD + SVD clamping (kappa<=5) reach mean final loss within 2x of Muon?',
            'reference_method': 'muon',
            'candidate_method': 'sgd_clamp_k5',
            'candidate_loss': clamp5_loss,
            'reference_loss': muon_loss,
            'ratio': t1_ratio,
            'threshold': 2.0,
            'pass': t1_pass,
            'scoped_interpretation': (
                'Clamp-at-5 is within the pre-specified 2x window.'
                if t1_pass
                else 'Clamp-at-5 does not match Muon within the 2x window under this setup.'
            ),
        },
        'T2': {
            'description': 'Does SGD + SVD equalization reach mean final loss within 2x of Muon?',
            'reference_method': 'muon',
            'candidate_method': 'sgd_equalize',
            'candidate_loss': equalize_loss,
            'reference_loss': muon_loss,
            'ratio': t2_ratio,
            'threshold': 2.0,
            'pass': t2_pass,
            'scoped_interpretation': (
                'Equalization is within the pre-specified 2x window.'
                if t2_pass
                else 'Equalization does not match Muon within the 2x window under this setup.'
            ),
        },
        'T3': {
            'description': 'If T1 and T2 both fail, conditioning-only post-step spectral controls are insufficient here.',
            'depends_on': ['T1', 'T2'],
            'pass': t3_pass,
            'scoped_interpretation': (
                'Both conditioning-only controls fail the 2x benchmark, so H15b counts that as evidence that conditioning-only post-step spectral controls are insufficient under this setup.'
                if t3_pass
                else 'At least one conditioning-only control reaches the 2x benchmark, so this first-pass H15b benchmark does not rule out conditioning-only explanations under its own criterion.'
            ),
        },
    }


def run_experiment(
    config=None,
    verbose=False,
    record_full_phase_histories=False,
    record_conditioning=True,
):
    config = make_config(config)
    seeds = generate_seeds(config)
    lr_sweep_seeds = seeds[: config['lr_sweep_num_seeds']]
    method_configs = build_method_configs(config)
    workload = estimate_workload(config)

    if verbose:
        print('=' * 100)
        print('H15b: EXPLICIT SVD CLAMPING VS MUON -- final-loss benchmark under post-step spectral controls')
        print('=' * 100)
        print(
            f"Network: {config['num_layers']}-layer, {config['dim']}x{config['dim']}, "
            f"{config['num_steps']} steps, {config['num_seeds']} seeds"
        )
        print(f"Learning-rate grid: {config['lr_candidates']}")
        print(f"Clamp targets: {config['kappa_targets']}")
        print(
            f"Best-in-grid LR selection uses the first {config['lr_sweep_num_seeds']} seeds: {lr_sweep_seeds}"
        )
        print(
            'Measurement scope: final loss after training; no direct update-direction metric is measured here.'
        )
        print(
            f"Estimated train runs: {workload['total_train_runs']} "
            f"({workload['lr_sweep_train_runs']} sweep + {workload['full_phase_train_runs']} full-phase)"
        )
        print()

    start_time = time.time()
    lr_sweep = {}
    best_lrs = {}

    if verbose:
        print('Phase 1: best-in-grid learning-rate sweeps...')
    for method_config in method_configs:
        name = method_config['name']
        sweep = sweep_lr(
            method_config['method'],
            lr_sweep_seeds,
            kappa_target=method_config['kappa_target'],
            config=config,
        )
        lr_sweep[name] = sweep
        best_lrs[name] = sweep['best_lr']
        if verbose:
            print(
                f"  {name:>20}: best_lr_in_grid={sweep['best_lr']:.4f}, "
                f"sweep_mean_final_loss={sweep['best_mean_final_loss_finite']:.6e}"
            )

    full_phase = {}
    if verbose:
        print('\nPhase 2: full evaluation at selected best-in-grid LRs...')
    for method_config in method_configs:
        name = method_config['name']
        best_lr = best_lrs[name]
        seed_results = []
        if verbose:
            print(f"  {name:>20}: lr={best_lr:.4f}")
        for seed in seeds:
            X, Y = make_data(seed, config=config)
            weights = init_weights(seed + config['weight_seed_offset'], config=config)
            train_details = train(
                weights,
                X,
                Y,
                best_lr,
                method_config['method'],
                kappa_target=method_config['kappa_target'],
                config=config,
                return_details=True,
                record_history=record_full_phase_histories,
                record_conditioning=record_conditioning,
            )
            seed_results.append(
                {
                    'seed': int(seed),
                    'final_loss': float(train_details['final_loss']),
                    'converged': bool(train_details['converged']),
                    'termination_reason': train_details['termination_reason'],
                    'steps_completed': int(train_details['steps_completed']),
                    'loss_history': train_details['loss_history'],
                    'conditioning': train_details['conditioning'],
                }
            )

        full_phase[name] = {
            'method': method_config['method'],
            'kappa_target': method_config['kappa_target'],
            'best_lr_in_grid': float(best_lr),
            'seed_results': seed_results,
            'summary': summarize_seed_results(seed_results, num_layers=config['num_layers']),
        }

    runtime_seconds = float(time.time() - start_time)

    results = {
        'experiment_id': 'H15b_SVD_CLAMPING_VS_MUON',
        'script_path': os.path.abspath(__file__),
        'config': config,
        'seeds': seeds,
        'lr_sweep_seeds': lr_sweep_seeds,
        'method_configs': method_configs,
        'workload_estimate': workload,
        'notes': {
            'lr_selection': (
                f"best-in-grid by mean finite final loss across the first {config['lr_sweep_num_seeds']} seeds"
            ),
            'measurement_scope': (
                f"final loss after {config['num_steps']} steps under post-step spectral controls; "
                'no direct update-direction-quality metric is measured here'
            ),
        },
        'lr_sweep': lr_sweep,
        'best_lrs': best_lrs,
        'full_phase': full_phase,
        'runtime_seconds': runtime_seconds,
    }
    results['summary_rows'] = make_summary_rows(results)
    results['lr_sweep_rows'] = make_lr_sweep_rows(results)
    results['tests'] = evaluate_tests(results)
    return results


def print_cli_report(results):
    print(f"\n{'=' * 100}")
    print('RESULTS: mean final loss across finite runs')
    print(f"{'=' * 100}")
    print(
        f"{'Method':>24}  {'best LR':>8}  {'mean loss':>14}  {'std':>10}  {'conv':>9}  {'vs Muon':>9}  {'vs SGD':>9}"
    )
    print('  ' + '-' * 92)
    for row in results['summary_rows']:
        print(
            f"{row['name']:>24}  {row['best_lr_in_grid']:>8.4f}  {row['mean_final_loss']:>14.6e}  "
            f"{row['std_final_loss']:>10.3e}  {row['num_converged']:>2d}/{row['num_total']:<6d}  "
            f"{row['ratio_vs_muon']:>9.2f}  {row['ratio_vs_sgd']:>9.2f}"
        )

    print(f"\n{'=' * 100}")
    print('FINAL CONDITIONING DIAGNOSTICS (finite full-phase runs only)')
    print(f"{'=' * 100}")
    print(f"{'Method':>24}  {'mean product kappa':>20}  {'mean layer kappas':>32}")
    print('  ' + '-' * 84)
    for row in results['summary_rows']:
        layer_text = '[' + ', '.join(f'{value:.2f}' for value in row['mean_final_layer_kappas']) + ']'
        print(
            f"{row['name']:>24}  {row['mean_final_product_kappa']:>20.4f}  {layer_text:>32}"
        )

    print(f"\n{'=' * 100}")
    print('HYPOTHESIS TESTS')
    print(f"{'=' * 100}")
    for test_name in ['T1', 'T2', 'T3']:
        test = results['tests'][test_name]
        print(f"\n{test_name}: {test['description']}")
        if 'ratio' in test:
            print(
                f"  candidate={test['candidate_loss']:.6e}, reference={test['reference_loss']:.6e}, "
                f"ratio={test['ratio']:.2f}x, threshold={test['threshold']:.2f}x"
            )
        print(f"  verdict={'PASS' if test['pass'] else 'FAIL'}")
        print(f"  interpretation={test['scoped_interpretation']}")

    print(f"\nRuntime: {results['runtime_seconds']:.2f} s")
    print(f"Notes: {results['notes']['lr_selection']}; {results['notes']['measurement_scope']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='H15b: explicit SVD clamping vs Muon final-loss benchmark.'
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run a reduced configuration for quick code-path verification.',
    )
    parser.add_argument(
        '--record-full-phase-histories',
        action='store_true',
        help='Store per-step loss histories for the full-phase runs.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_overrides = SMOKE_CONFIG_OVERRIDES if args.smoke else None
    results = run_experiment(
        config=config_overrides,
        verbose=True,
        record_full_phase_histories=(args.record_full_phase_histories or args.smoke),
        record_conditioning=True,
    )
    print_cli_report(results)
    return results


if __name__ == '__main__':
    main()
