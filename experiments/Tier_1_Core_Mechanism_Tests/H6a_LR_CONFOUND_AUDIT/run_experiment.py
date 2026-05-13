#!/usr/bin/env python3
"""
H6a: LR confound audit of the D-TEST depth-scaling claim
=========================================================

This module runs a discrete learning-rate audit in the original 32x32 deep-linear
regression setting and compares two protocols:

1. A convergence-aware per-depth LR sweep for both SGD and Muon.
2. A D-TEST-style replica with formula SGD LR and fixed Muon LR.

Measured outputs are limited to this setting and include:
- per-LR final losses across seeds
- convergence counts and fractions
- selected best LRs under the discrete sweep
- log(advantage) vs depth linear fits
- temporal advantage at selected training steps

This experiment does not, by itself, measure asymptotic complexity classes or
establish RG / gauge-fixing mechanisms. It is a deep-linear LR-confound audit
under the specific sweep grids defined below.
"""

from pathlib import Path
import time

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# =============================================================================
# CONFIGURATION — preserve the original deep-linear audit setup
# =============================================================================

DIM = 32
DEPTHS = [2, 4, 8, 16]
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
BATCH_SIZE = 64
NUM_SEEDS = 3

SGD_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
MUON_LRS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]

MEASUREMENT_STEPS = [50, 100, 150, 200, 250, 300]
DTEST_MUON_LR = 0.005
SEED_BASE = 42
SEED_STRIDE = 137
INIT_SEED_OFFSET = 5000
DIVERGENCE_THRESHOLD = 1e10


# =============================================================================
# NETWORK AND TRAINING UTILITIES
# =============================================================================


def newton_schulz(M, n_iters=NS_ITERS):
    """Newton-Schulz iteration for an orthogonal polar factor."""
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X



def init_weights(dim, depth, seed):
    """Initialize near identity for stability (same as D-TEST)."""
    rng = np.random.RandomState(seed)
    return [np.eye(dim) + rng.randn(dim, dim) * 0.1 for _ in range(depth)]



def make_data(dim, seed, batch_size=BATCH_SIZE):
    """Generate target matrix and data (same as D-TEST: single random target)."""
    rng = np.random.RandomState(seed)
    W_target = rng.randn(dim, dim) * 0.5
    X = rng.randn(dim, batch_size) * 0.3
    Y = W_target @ X
    return X, Y



def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    diff = pred - Y
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))



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



def train(weights_init, X, Y, lr, optimizer, n_steps=NUM_STEPS):
    """Train and return (final_loss, loss_history)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    losses = []

    for step in range(n_steps):
        loss = compute_loss(weights, X, Y)
        losses.append(loss)
        if not np.isfinite(loss) or loss > DIVERGENCE_THRESHOLD:
            losses.extend([float('inf')] * (n_steps - step))
            return float('inf'), losses

        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if optimizer == 'muon':
                grad_update = newton_schulz(grads[i])
            else:
                grad_update = grads[i]
            mom[i] = MOMENTUM * mom[i] + grad_update
            weights[i] = weights[i] - lr * mom[i]

    final_loss = compute_loss(weights, X, Y)
    losses.append(final_loss)
    return final_loss, losses


# =============================================================================
# D-TEST comparison utility
# =============================================================================


def dtest_sgd_lr(depth, X, Y):
    """Replicate D-TEST's lr = 2/(lambda_max * L) formula."""
    rng_state = np.random.get_state()
    np.random.seed(42)
    test_weights = init_weights(DIM, depth, 42)
    np.random.set_state(rng_state)

    W_prod = np.eye(DIM)
    for W in test_weights:
        W_prod = W @ W_prod
    sv_prod = np.linalg.svd(W_prod, compute_uv=False)
    sv_X = np.linalg.svd(X, compute_uv=False)
    N = X.shape[1]
    lambda_max = (sv_prod[0] ** 2) * (sv_X[0] ** 2) / N
    lr = min(2.0 / (lambda_max * depth), 0.1)
    return lr


# =============================================================================
# RESULT HELPERS
# =============================================================================


def make_seeds(num_seeds=NUM_SEEDS, base=SEED_BASE, stride=SEED_STRIDE):
    return [base + i * stride for i in range(num_seeds)]



def expected_training_call_counts(config):
    num_depths = len(config['depths'])
    num_seeds = config['num_seeds']
    sweep_runs = num_depths * (len(config['sgd_lrs']) + len(config['muon_lrs'])) * num_seeds
    dtest_replica_runs = num_depths * num_seeds * 2
    temporal_runs = num_depths * num_seeds * 2
    return {
        'sweep_runs': sweep_runs,
        'dtest_replica_runs': dtest_replica_runs,
        'temporal_runs': temporal_runs,
        'total_training_calls': sweep_runs + dtest_replica_runs + temporal_runs,
    }



def get_default_config():
    config = {
        'dim': DIM,
        'depths': list(DEPTHS),
        'num_steps': NUM_STEPS,
        'momentum': MOMENTUM,
        'ns_iters': NS_ITERS,
        'batch_size': BATCH_SIZE,
        'num_seeds': NUM_SEEDS,
        'sgd_lrs': list(SGD_LRS),
        'muon_lrs': list(MUON_LRS),
        'measurement_steps': list(MEASUREMENT_STEPS),
        'dtest_muon_lr': DTEST_MUON_LR,
        'seed_base': SEED_BASE,
        'seed_stride': SEED_STRIDE,
        'init_seed_offset': INIT_SEED_OFFSET,
        'divergence_threshold': DIVERGENCE_THRESHOLD,
        'selection_rule': (
            'Prefer higher convergence count, then lower median finite final loss, '
            'then lower LR as a deterministic tie-break.'
        ),
        'scope_note': (
            'Deep-linear LR-confound audit under discrete sweep grids; not a direct '
            'test of asymptotic complexity classes or RG / gauge observables.'
        ),
        'script_path': str(SCRIPT_PATH),
        'experiment_dir': str(SCRIPT_DIR),
    }
    config['expected_training_calls'] = expected_training_call_counts(config)
    return config



def summarize_final_losses(final_losses, num_seeds):
    finite_losses = [float(loss) for loss in final_losses if np.isfinite(loss)]
    converged_count = len(finite_losses)
    if finite_losses:
        median_loss = float(np.median(finite_losses))
        mean_loss = float(np.mean(finite_losses))
        std_loss = float(np.std(finite_losses))
        sem_loss = float(np.std(finite_losses, ddof=1) / np.sqrt(len(finite_losses))) if len(finite_losses) > 1 else 0.0
    else:
        median_loss = float('inf')
        mean_loss = float('inf')
        std_loss = float('inf')
        sem_loss = float('inf')

    return {
        'final_losses': [float(loss) if np.isfinite(loss) else float('inf') for loss in final_losses],
        'finite_final_losses': finite_losses,
        'converged_count': converged_count,
        'convergence_fraction': converged_count / num_seeds,
        'median_finite_loss': median_loss,
        'mean_finite_loss': mean_loss,
        'std_finite_loss': std_loss,
        'sem_finite_loss': sem_loss,
        'all_converged': converged_count == num_seeds,
        'any_converged': bool(finite_losses),
    }



def select_best_lr(lr_results, lr_grid):
    """
    Select the best LR using a convergence-aware rule.

    Preferred policy:
      1. maximize converged seeds
      2. minimize median finite loss
      3. minimize LR as a deterministic tie-break

    Also return the legacy median-over-finite-only selection for transparency.
    """
    if not lr_results:
        raise ValueError('select_best_lr received an empty lr_results list')

    max_converged = max(result['converged_count'] for result in lr_results)
    convergence_candidates = [result for result in lr_results if result['converged_count'] == max_converged]
    selected = min(convergence_candidates, key=lambda result: (result['median_finite_loss'], result['lr']))
    legacy_selected = min(lr_results, key=lambda result: (result['median_finite_loss'], result['lr']))

    def annotate(result, policy_name):
        annotated = dict(result)
        annotated['selection_policy'] = policy_name
        annotated['boundary_hit'] = annotated['lr'] in (lr_grid[0], lr_grid[-1])
        annotated['grid_min_lr'] = lr_grid[0]
        annotated['grid_max_lr'] = lr_grid[-1]
        annotated['grid_size'] = len(lr_grid)
        return annotated

    selected_annotated = annotate(selected, 'convergence_first_then_median_finite')
    legacy_annotated = annotate(legacy_selected, 'median_finite_only')

    return {
        'selected': selected_annotated,
        'legacy_selected': legacy_annotated,
        'selection_changed_vs_legacy': (
            selected_annotated['lr'] != legacy_annotated['lr']
            or selected_annotated['converged_count'] != legacy_annotated['converged_count']
        ),
    }



def compute_advantage_summary(depth, sgd_summary, muon_summary, label):
    sgd_loss = sgd_summary['median_finite_loss']
    muon_loss = muon_summary['median_finite_loss']
    if np.isfinite(sgd_loss) and np.isfinite(muon_loss) and muon_loss > 1e-30:
        advantage = float(sgd_loss / muon_loss)
        log_advantage = float(np.log(advantage))
    else:
        advantage = float('inf')
        log_advantage = float('inf')

    return {
        'protocol': label,
        'depth': depth,
        'sgd_lr': float(sgd_summary['lr']),
        'muon_lr': float(muon_summary['lr']),
        'sgd_summary': dict(sgd_summary),
        'muon_summary': dict(muon_summary),
        'advantage': advantage,
        'log_advantage': log_advantage,
        'valid_for_fit': np.isfinite(log_advantage),
    }



def linear_fit(depths, log_advantages, label):
    if len(depths) < 2:
        return {
            'label': label,
            'n_points': len(depths),
            'depths': [int(depth) for depth in depths],
            'log_advantages': [float(value) for value in log_advantages],
            'slope': 0.0,
            'intercept': 0.0,
            'r2': 0.0,
            'per_layer_factor': 1.0,
            'predicted_log_advantages': [],
            'residuals': [],
            'valid': False,
        }

    d = np.array(depths, dtype=float)
    y = np.array(log_advantages, dtype=float)
    A = np.vstack([d, np.ones(len(d))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    predicted = slope * d + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0.0

    return {
        'label': label,
        'n_points': len(depths),
        'depths': [int(depth) for depth in d.tolist()],
        'log_advantages': [float(value) for value in y.tolist()],
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'per_layer_factor': float(np.exp(slope)),
        'predicted_log_advantages': [float(value) for value in predicted.tolist()],
        'residuals': [float(value) for value in (y - predicted).tolist()],
        'valid': True,
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_experiment(verbose=True):
    config = get_default_config()
    seeds = make_seeds(config['num_seeds'], config['seed_base'], config['seed_stride'])
    expected_calls = config['expected_training_calls']

    t_start = time.time()
    training_calls = 0

    if verbose:
        print()
        print('=' * 110)
        print('  H6a: LR CONFOUND AUDIT OF THE D-TEST DEPTH-SCALING CLAIM')
        print('=' * 110)
        print()
        print('  Scope: deep-linear LR-confound audit under discrete LR sweeps.')
        print(f"  Not directly measured: complexity classes or RG / gauge observables.")
        print()
        print(f"  Setup: {config['dim']}x{config['dim']} deep linear, {config['num_steps']} steps, {config['num_seeds']} seeds")
        print(f"  Depths: {config['depths']}")
        print(f"  SGD LR sweep:  {config['sgd_lrs']}")
        print(f"  Muon LR sweep: {config['muon_lrs']}")
        print(f"  Measurement steps: {config['measurement_steps']}")
        print(f"  Selection rule: {config['selection_rule']}")
        print(f"  Expected training calls: {expected_calls['total_training_calls']}")
        print()
        print('  Phase 1: sweeping LRs ...')
        print()

    sweep = {}
    best_by_depth = []
    legacy_selection_differences = []

    for depth in config['depths']:
        sweep[depth] = {}
        if verbose:
            print(f"  --- Depth L={depth} ---")

        for optimizer, lr_grid in [('sgd', config['sgd_lrs']), ('muon', config['muon_lrs'])]:
            lr_results = []
            for lr in lr_grid:
                final_losses = []
                for seed in seeds:
                    X, Y = make_data(config['dim'], seed, batch_size=config['batch_size'])
                    weights_init = init_weights(config['dim'], depth, seed + config['init_seed_offset'])
                    final_loss, _ = train(weights_init, X, Y, lr, optimizer, n_steps=config['num_steps'])
                    final_losses.append(final_loss)
                    training_calls += 1

                loss_summary = summarize_final_losses(final_losses, config['num_seeds'])
                lr_results.append({
                    'lr': float(lr),
                    **loss_summary,
                })

            selection = select_best_lr(lr_results, lr_grid)
            best_summary = selection['selected']
            legacy_best_summary = selection['legacy_selected']

            sweep[depth][optimizer] = {
                'lr_results': lr_results,
                'best': best_summary,
                'legacy_best': legacy_best_summary,
            }

            if selection['selection_changed_vs_legacy']:
                legacy_selection_differences.append({
                    'depth': depth,
                    'optimizer': optimizer,
                    'selected_lr': best_summary['lr'],
                    'selected_converged_count': best_summary['converged_count'],
                    'selected_median_finite_loss': best_summary['median_finite_loss'],
                    'legacy_lr': legacy_best_summary['lr'],
                    'legacy_converged_count': legacy_best_summary['converged_count'],
                    'legacy_median_finite_loss': legacy_best_summary['median_finite_loss'],
                })

            if verbose:
                print(
                    f"    {optimizer.upper():>5}: selected_lr={best_summary['lr']:.4f}  "
                    f"converged={best_summary['converged_count']}/{config['num_seeds']}  "
                    f"median_finite_loss={best_summary['median_finite_loss']:.6e}"
                )
                if selection['selection_changed_vs_legacy']:
                    print(
                        f"           legacy median-only would pick lr={legacy_best_summary['lr']:.4f} "
                        f"with {legacy_best_summary['converged_count']}/{config['num_seeds']} converged"
                    )

        sgd_best = sweep[depth]['sgd']['best']
        muon_best = sweep[depth]['muon']['best']
        sgd_legacy = sweep[depth]['sgd']['legacy_best']
        muon_legacy = sweep[depth]['muon']['legacy_best']
        advantage_summary = compute_advantage_summary(depth, sgd_best, muon_best, 'swept')
        best_by_depth.append({
            'depth': depth,
            'sgd_best_lr': sgd_best['lr'],
            'sgd_converged_count': sgd_best['converged_count'],
            'sgd_convergence_fraction': sgd_best['convergence_fraction'],
            'sgd_median_finite_loss': sgd_best['median_finite_loss'],
            'sgd_mean_finite_loss': sgd_best['mean_finite_loss'],
            'sgd_sem_finite_loss': sgd_best['sem_finite_loss'],
            'sgd_boundary_hit': sgd_best['boundary_hit'],
            'sgd_legacy_lr': sgd_legacy['lr'],
            'sgd_legacy_converged_count': sgd_legacy['converged_count'],
            'sgd_legacy_median_finite_loss': sgd_legacy['median_finite_loss'],
            'sgd_selection_changed_vs_legacy': sgd_best['lr'] != sgd_legacy['lr'],
            'muon_best_lr': muon_best['lr'],
            'muon_converged_count': muon_best['converged_count'],
            'muon_convergence_fraction': muon_best['convergence_fraction'],
            'muon_median_finite_loss': muon_best['median_finite_loss'],
            'muon_mean_finite_loss': muon_best['mean_finite_loss'],
            'muon_sem_finite_loss': muon_best['sem_finite_loss'],
            'muon_boundary_hit': muon_best['boundary_hit'],
            'muon_legacy_lr': muon_legacy['lr'],
            'muon_legacy_converged_count': muon_legacy['converged_count'],
            'muon_legacy_median_finite_loss': muon_legacy['median_finite_loss'],
            'muon_selection_changed_vs_legacy': muon_best['lr'] != muon_legacy['lr'],
            'advantage': advantage_summary['advantage'],
            'log_advantage': advantage_summary['log_advantage'],
            'valid_for_fit': advantage_summary['valid_for_fit'],
        })

    sweep_elapsed = time.time() - t_start

    swept_advantage = []
    swept_depths = []
    swept_log_advantages = []
    for row in best_by_depth:
        summary = {
            'protocol': 'swept',
            'depth': row['depth'],
            'sgd_lr': row['sgd_best_lr'],
            'muon_lr': row['muon_best_lr'],
            'advantage': row['advantage'],
            'log_advantage': row['log_advantage'],
            'valid_for_fit': row['valid_for_fit'],
        }
        swept_advantage.append(summary)
        if row['valid_for_fit']:
            swept_depths.append(row['depth'])
            swept_log_advantages.append(row['log_advantage'])

    dtest_replica = []
    dtest_depths = []
    dtest_log_advantages = []

    for depth in config['depths']:
        sgd_losses = []
        muon_losses = []
        sgd_formula_lrs = []

        for seed in seeds:
            X, Y = make_data(config['dim'], seed, batch_size=config['batch_size'])
            weights_init = init_weights(config['dim'], depth, seed + config['init_seed_offset'])
            sgd_lr_formula = float(dtest_sgd_lr(depth, X, Y))
            sgd_formula_lrs.append(sgd_lr_formula)

            final_loss_sgd, _ = train(weights_init, X, Y, sgd_lr_formula, 'sgd', n_steps=config['num_steps'])
            sgd_losses.append(final_loss_sgd)
            training_calls += 1

            final_loss_muon, _ = train(weights_init, X, Y, config['dtest_muon_lr'], 'muon', n_steps=config['num_steps'])
            muon_losses.append(final_loss_muon)
            training_calls += 1

        sgd_summary = {
            'lr': float(np.median(sgd_formula_lrs)),
            **summarize_final_losses(sgd_losses, config['num_seeds']),
            'formula_lrs_by_seed': [float(lr) for lr in sgd_formula_lrs],
            'formula_lr_mean': float(np.mean(sgd_formula_lrs)),
            'formula_lr_median': float(np.median(sgd_formula_lrs)),
            'formula_lr_min': float(np.min(sgd_formula_lrs)),
            'formula_lr_max': float(np.max(sgd_formula_lrs)),
        }
        muon_summary = {
            'lr': float(config['dtest_muon_lr']),
            **summarize_final_losses(muon_losses, config['num_seeds']),
        }

        advantage_summary = compute_advantage_summary(depth, sgd_summary, muon_summary, 'dtest_replica')
        replica_row = {
            'depth': depth,
            'sgd_formula_lrs_by_seed': [float(lr) for lr in sgd_formula_lrs],
            'sgd_formula_lr_mean': float(np.mean(sgd_formula_lrs)),
            'sgd_formula_lr_median': float(np.median(sgd_formula_lrs)),
            'sgd_formula_lr_min': float(np.min(sgd_formula_lrs)),
            'sgd_formula_lr_max': float(np.max(sgd_formula_lrs)),
            'sgd_summary': sgd_summary,
            'muon_summary': muon_summary,
            'advantage': advantage_summary['advantage'],
            'log_advantage': advantage_summary['log_advantage'],
            'valid_for_fit': advantage_summary['valid_for_fit'],
        }
        dtest_replica.append(replica_row)
        if replica_row['valid_for_fit']:
            dtest_depths.append(depth)
            dtest_log_advantages.append(replica_row['log_advantage'])

    fits = {
        'swept': linear_fit(
            swept_depths,
            swept_log_advantages,
            'Convergence-aware swept LR: log(advantage) vs depth',
        ),
        'dtest_replica': linear_fit(
            dtest_depths,
            dtest_log_advantages,
            'D-TEST replica: log(advantage) vs depth',
        ),
    }

    lr_scaling_by_depth = []
    for best_row, dtest_row in zip(best_by_depth, dtest_replica):
        dtest_median_lr = dtest_row['sgd_formula_lr_median']
        lr_scaling_by_depth.append({
            'depth': best_row['depth'],
            'sgd_best_lr': best_row['sgd_best_lr'],
            'muon_best_lr': best_row['muon_best_lr'],
            'dtest_formula_sgd_lr_median': dtest_median_lr,
            'dtest_formula_sgd_lr_min': dtest_row['sgd_formula_lr_min'],
            'dtest_formula_sgd_lr_max': dtest_row['sgd_formula_lr_max'],
            'sgd_swept_over_dtest_formula': (
                float(best_row['sgd_best_lr'] / dtest_median_lr) if dtest_median_lr > 0 else float('nan')
            ),
        })

    temporal_advantage = []
    for depth in config['depths']:
        sgd_lr = sweep[depth]['sgd']['best']['lr']
        muon_lr = sweep[depth]['muon']['best']['lr']
        sgd_curves = []
        muon_curves = []
        step_summaries = []

        for seed in seeds:
            X, Y = make_data(config['dim'], seed, batch_size=config['batch_size'])
            weights_init = init_weights(config['dim'], depth, seed + config['init_seed_offset'])

            _, sgd_losses_curve = train(weights_init, X, Y, sgd_lr, 'sgd', n_steps=config['num_steps'])
            sgd_curves.append([float(loss) if np.isfinite(loss) else float('inf') for loss in sgd_losses_curve])
            training_calls += 1

            _, muon_losses_curve = train(weights_init, X, Y, muon_lr, 'muon', n_steps=config['num_steps'])
            muon_curves.append([float(loss) if np.isfinite(loss) else float('inf') for loss in muon_losses_curve])
            training_calls += 1

        for step in config['measurement_steps']:
            sgd_losses_at_step = [curve[step] if step < len(curve) else curve[-1] for curve in sgd_curves]
            muon_losses_at_step = [curve[step] if step < len(curve) else curve[-1] for curve in muon_curves]
            sgd_step_summary = summarize_final_losses(sgd_losses_at_step, config['num_seeds'])
            muon_step_summary = summarize_final_losses(muon_losses_at_step, config['num_seeds'])

            if (
                np.isfinite(sgd_step_summary['median_finite_loss'])
                and np.isfinite(muon_step_summary['median_finite_loss'])
                and muon_step_summary['median_finite_loss'] > 1e-30
            ):
                advantage = float(
                    sgd_step_summary['median_finite_loss'] / muon_step_summary['median_finite_loss']
                )
                log_advantage = float(np.log(advantage))
            else:
                advantage = float('inf')
                log_advantage = float('inf')

            step_summaries.append({
                'step': step,
                'sgd_summary': sgd_step_summary,
                'muon_summary': muon_step_summary,
                'advantage': advantage,
                'log_advantage': log_advantage,
            })

        temporal_advantage.append({
            'depth': depth,
            'sgd_lr': float(sgd_lr),
            'muon_lr': float(muon_lr),
            'sgd_curves': sgd_curves,
            'muon_curves': muon_curves,
            'step_summaries': step_summaries,
        })

    total_elapsed = time.time() - t_start

    results = {
        'metadata': {
            'experiment_id': 'H6a_LR_CONFOUND_AUDIT',
            'title': 'H6a: LR confound audit of the D-TEST depth-scaling claim',
            'script_path': str(SCRIPT_PATH),
            'experiment_dir': str(SCRIPT_DIR),
        },
        'config': config,
        'seeds': seeds,
        'selection_policy': {
            'name': 'convergence_first_then_median_finite',
            'description': config['selection_rule'],
        },
        'sweep': sweep,
        'best_by_depth': best_by_depth,
        'legacy_selection_differences': legacy_selection_differences,
        'swept_advantage': swept_advantage,
        'dtest_replica': {
            'fixed_muon_lr': float(config['dtest_muon_lr']),
            'by_depth': dtest_replica,
        },
        'fits': fits,
        'lr_scaling': {
            'by_depth': lr_scaling_by_depth,
            'sgd_endpoint_ratio': (
                float(best_by_depth[-1]['sgd_best_lr'] / best_by_depth[0]['sgd_best_lr'])
                if best_by_depth and best_by_depth[0]['sgd_best_lr'] > 0 else float('nan')
            ),
            'muon_endpoint_ratio': (
                float(best_by_depth[-1]['muon_best_lr'] / best_by_depth[0]['muon_best_lr'])
                if best_by_depth and best_by_depth[0]['muon_best_lr'] > 0 else float('nan')
            ),
        },
        'temporal_advantage': {
            'measurement_steps': list(config['measurement_steps']),
            'by_depth': temporal_advantage,
        },
        'run_counts': {
            **expected_calls,
            'actual_training_calls': training_calls,
        },
        'timing': {
            'sweep_wall_time_seconds': float(sweep_elapsed),
            'total_wall_time_seconds': float(total_elapsed),
        },
    }

    if verbose:
        print_report(results)

    return results


# =============================================================================
# REPORTING
# =============================================================================


def print_report(results):
    config = results['config']
    best_by_depth = results['best_by_depth']
    dtest_rows = results['dtest_replica']['by_depth']
    fit_swept = results['fits']['swept']
    fit_dtest = results['fits']['dtest_replica']

    print()
    print('=' * 110)
    print('  PHASE 2: CONVERGENCE-AWARE BEST LR SUMMARY')
    print('=' * 110)
    print()
    print(
        f"  {'Depth':>5} | {'Best SGD LR':>12} {'Conv':>7} {'Median loss':>14} | "
        f"{'Best Muon LR':>12} {'Conv':>7} {'Median loss':>14} | {'Advantage':>10}"
    )
    print(
        f"  {'':->5}-+-{'':->12}-{'':->7}-{'':->14}-+-{'':->12}-{'':->7}-{'':->14}-+-{'':->10}"
    )
    for row in best_by_depth:
        print(
            f"  {row['depth']:>5} | {row['sgd_best_lr']:>12.4f} "
            f"{row['sgd_converged_count']:>3}/{config['num_seeds']:<3} {row['sgd_median_finite_loss']:>14.6e} | "
            f"{row['muon_best_lr']:>12.4f} "
            f"{row['muon_converged_count']:>3}/{config['num_seeds']:<3} {row['muon_median_finite_loss']:>14.6e} | "
            f"{row['advantage']:>9.2f}x"
        )

    print()
    if results['legacy_selection_differences']:
        print('  Legacy-selection differences (median-over-finite-only would have picked):')
        for item in results['legacy_selection_differences']:
            print(
                f"    depth={item['depth']:>2}  opt={item['optimizer']:<4}  "
                f"new_lr={item['selected_lr']:.4f} ({item['selected_converged_count']}/{config['num_seeds']} conv)  "
                f"legacy_lr={item['legacy_lr']:.4f} ({item['legacy_converged_count']}/{config['num_seeds']} conv)"
            )
    else:
        print('  No differences from the legacy median-over-finite-only selector on this run.')

    boundary_hits = [
        (row['depth'], 'sgd', row['sgd_best_lr'])
        for row in best_by_depth if row['sgd_boundary_hit']
    ] + [
        (row['depth'], 'muon', row['muon_best_lr'])
        for row in best_by_depth if row['muon_boundary_hit']
    ]
    if boundary_hits:
        print('  Boundary-hit warning(s):')
        for depth, optimizer, lr in boundary_hits:
            print(f"    depth={depth:>2}  opt={optimizer:<4}  selected LR sits on sweep edge at {lr:.4f}")

    print()
    print('=' * 110)
    print('  PHASE 3: D-TEST-STYLE REPLICA (formula SGD LR, fixed Muon LR = 0.005)')
    print('=' * 110)
    print()
    print(
        f"  {'Depth':>5} | {'Formula SGD LR':>14} {'Range':>20} | "
        f"{'SGD conv':>8} {'SGD median':>14} | {'Muon conv':>9} {'Muon median':>14} | {'Adv':>8}"
    )
    print(
        f"  {'':->5}-+-{'':->14}-{'':->20}-+-{'':->8}-{'':->14}-+-{'':->9}-{'':->14}-+-{'':->8}"
    )
    for row in dtest_rows:
        sgd_summary = row['sgd_summary']
        muon_summary = row['muon_summary']
        print(
            f"  {row['depth']:>5} | {row['sgd_formula_lr_median']:>14.6f} "
            f"[{row['sgd_formula_lr_min']:.6f}, {row['sgd_formula_lr_max']:.6f}] | "
            f"{sgd_summary['converged_count']:>3}/{config['num_seeds']:<3} {sgd_summary['median_finite_loss']:>14.6e} | "
            f"{muon_summary['converged_count']:>3}/{config['num_seeds']:<3} {muon_summary['median_finite_loss']:>14.6e} | "
            f"{row['advantage']:>7.2f}x"
        )

    print()
    print('=' * 110)
    print('  PHASE 4: FIT SUMMARY — log(advantage) vs depth')
    print('=' * 110)
    print()
    print(
        f"  {'Protocol':<48} {'Slope':>10} {'e^slope':>10} {'R^2':>10} {'Points':>8}"
    )
    print(
        f"  {'':-<48} {'':->10} {'':->10} {'':->10} {'':->8}"
    )
    for fit in [fit_swept, fit_dtest]:
        print(
            f"  {fit['label']:<48} {fit['slope']:>10.4f} {fit['per_layer_factor']:>10.4f} "
            f"{fit['r2']:>10.4f} {fit['n_points']:>8}"
        )

    print()
    print('=' * 110)
    print('  PHASE 5: LR SCALING WITH DEPTH')
    print('=' * 110)
    print()
    print(
        f"  {'Depth':>5} | {'Swept SGD':>10} | {'Swept Muon':>10} | {'D-TEST SGD med':>14} | {'Swept / D-TEST':>14}"
    )
    print(
        f"  {'':->5}-+-{'':->10}-+-{'':->10}-+-{'':->14}-+-{'':->14}"
    )
    for row in results['lr_scaling']['by_depth']:
        print(
            f"  {row['depth']:>5} | {row['sgd_best_lr']:>10.4f} | {row['muon_best_lr']:>10.4f} | "
            f"{row['dtest_formula_sgd_lr_median']:>14.6f} | {row['sgd_swept_over_dtest_formula']:>14.2f}x"
        )
    print()
    print(f"  SGD endpoint LR ratio (L={config['depths'][-1]} / L={config['depths'][0]}):  {results['lr_scaling']['sgd_endpoint_ratio']:.4f}")
    print(f"  Muon endpoint LR ratio (L={config['depths'][-1]} / L={config['depths'][0]}): {results['lr_scaling']['muon_endpoint_ratio']:.4f}")

    print()
    print('=' * 110)
    print('  PHASE 6: FULL LR LANDSCAPE (median finite loss, convergence shown explicitly)')
    print('=' * 110)
    for depth in config['depths']:
        print(f"\n  Depth L={depth}:")
        for optimizer in ['sgd', 'muon']:
            print(f"    {optimizer.upper()}:")
            for row in results['sweep'][depth][optimizer]['lr_results']:
                marker = ' <-- SELECTED' if row['lr'] == results['sweep'][depth][optimizer]['best']['lr'] else ''
                print(
                    f"      lr={row['lr']:.4f}  median_finite={row['median_finite_loss']:12.6e}  "
                    f"mean_finite={row['mean_finite_loss']:12.6e}  "
                    f"converged={row['converged_count']}/{config['num_seeds']}{marker}"
                )

    print()
    print('=' * 110)
    print('  PHASE 7: TEMPORAL ADVANTAGE AT SELECTED TRAINING STEPS (using selected best LRs)')
    print('=' * 110)
    print()
    print(f"  {'Depth':>5} |", end='')
    for step in results['temporal_advantage']['measurement_steps']:
        print(f"  Step {step:>3}", end='')
    print()
    print(f"  {'':->5}-+", end='')
    for _ in results['temporal_advantage']['measurement_steps']:
        print(f"{'':->10}", end='')
    print()
    for depth_row in results['temporal_advantage']['by_depth']:
        print(f"  {depth_row['depth']:>5} |", end='')
        for step_summary in depth_row['step_summaries']:
            advantage = step_summary['advantage']
            if np.isfinite(advantage):
                print(f"  {advantage:>7.2f}x", end='')
            else:
                print(f"  {'INF':>8}", end='')
        print()

    print()
    print('=' * 110)
    print('  CALIBRATED INTERPRETATION')
    print('=' * 110)
    print()
    print('  This audit measures final-loss ratios and convergence behavior under the discrete sweeps above.')
    print('  It does not directly prove complexity-class separation or RG / gauge explanations.')
    print()
    print(
        f"  D-TEST-style replica fit: slope={fit_dtest['slope']:.4f}, "
        f"e^slope={fit_dtest['per_layer_factor']:.4f}, R^2={fit_dtest['r2']:.4f}"
    )
    print(
        f"  Swept-LR audit fit:      slope={fit_swept['slope']:.4f}, "
        f"e^slope={fit_swept['per_layer_factor']:.4f}, R^2={fit_swept['r2']:.4f}"
    )
    print()

    slope_delta = fit_swept['slope'] - fit_dtest['slope']
    r2_delta = fit_swept['r2'] - fit_dtest['r2']
    if fit_swept['slope'] < fit_dtest['slope'] and fit_swept['r2'] < fit_dtest['r2']:
        print('  Under convergence-aware per-depth retuning, the depth trend is weaker than in the D-TEST-style replica.')
    elif fit_swept['slope'] > fit_dtest['slope'] and fit_swept['r2'] >= fit_dtest['r2']:
        print('  Under convergence-aware per-depth retuning, the depth trend is at least as strong as in the D-TEST-style replica.')
    else:
        print('  The relationship between the swept-LR audit and D-TEST-style replica is mixed in this run.')
    print(f"  Delta(slope) = {slope_delta:+.4f}, Delta(R^2) = {r2_delta:+.4f}")
    print()
    print('  Caveats: 4 depths, 3 seeds, discrete LR grids, no confidence intervals, and possible sweep-boundary effects.')
    print(f"  Total training calls executed: {results['run_counts']['actual_training_calls']}")
    print(f"  Total wall time: {results['timing']['total_wall_time_seconds']:.1f}s")
    print()
    print('=' * 110)
    print('  EXPERIMENT COMPLETE')
    print('=' * 110)
    print()



def main():
    run_experiment(verbose=True)


if __name__ == '__main__':
    main()
