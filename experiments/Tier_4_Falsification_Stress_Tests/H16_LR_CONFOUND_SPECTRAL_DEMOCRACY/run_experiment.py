#!/usr/bin/env python3
"""
H16: LR Confound Audit -- Final Effective Rank and Condition Number
===================================================================

This experiment asks whether the apparent spectral-democracy and conditioning
comparisons between Muon and SGD survive learning-rate matching.

Scope of the current H16 pair
-----------------------------
Computed at the final training step:
  - effective rank per layer
  - condition number per layer

Explicitly out of scope here:
  - WeightWatcher alpha
  - power-law tail fitting of eigenvalue spectra

The stale alpha wording in earlier H16 text was incorrect. Alpha-style analyses
belong in dedicated Exp 1.3c-style studies, not in this minimum LR-confound
falsification audit.

Protocol
--------
1. Sweep candidate learning rates separately for SGD and Muon on the first
   3 seeds and choose the LR with the lowest mean final loss.
2. Evaluate four configurations across 10 seeds:
      SGD @ default LR
      SGD @ chosen grid-best LR
      Muon @ default LR
      Muon @ chosen grid-best LR
3. Record final loss plus per-layer effective rank and condition number.
4. Apply the existing falsification heuristics:
      T1: the mean layerwise effective-rank advantage changes by >50%
      T2: the mean layerwise Muon/SGD condition-number ratio flips across 1

Setup: 4-layer, 32x32, 500 steps, momentum 0.9, 10 seeds.
"""

import numpy as np

EXPERIMENT_ID = 'H16_LR_CONFOUND_SPECTRAL_DEMOCRACY'
EXPERIMENT_TITLE = 'H16: LR Confound Audit -- Final Effective Rank and Condition Number'
SCOPE_NOTE = (
    'This pair audits LR confounds for final effective rank and condition number only. '
    'WeightWatcher/power-law alpha is explicitly out of scope here.'
)

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
BATCH_SIZE = 64
DATA_SCALE = 0.3
WEIGHT_INIT_SCALE = 0.1
WEIGHT_SEED_OFFSET = 5000
LR_SWEEP_NUM_SEEDS = 3

MUON_LRS = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
SGD_LRS = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]

DEFAULT_MUON_LR = 0.02
DEFAULT_SGD_LR = 0.01

CONFIG_LABELS = {
    'sgd_default': 'SGD @ default LR',
    'sgd_optimal': 'SGD @ chosen LR',
    'muon_default': 'Muon @ default LR',
    'muon_optimal': 'Muon @ chosen LR',
}


def get_default_seeds():
    return [42 + i * 137 for i in range(NUM_SEEDS)]


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * WEIGHT_INIT_SCALE for _ in range(NUM_LAYERS)]


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


def effective_rank(W):
    """Effective rank = exp(entropy of normalized singular values)."""
    s = np.linalg.svd(W, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / np.sum(s)
    H = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(H))


def condition_number(W):
    s = np.linalg.svd(W, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-12))


def train(weights_init, X, Y, lr, optimizer):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return None, float('inf')
        grads = compute_gradients(weights, X, Y)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return weights, float(compute_loss(weights, X, Y))


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * DATA_SCALE
    Y = rng.randn(DIM, BATCH_SIZE) * DATA_SCALE
    return X, Y


def summarize_values(values):
    if not values:
        return {
            'count': 0,
            'finite_count': 0,
            'diverged_count': 0,
            'mean_all': None,
            'mean_finite': None,
            'sd_finite': None,
            'sem_finite': None,
        }
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite_mean = None
        finite_sd = None
        finite_sem = None
    elif finite.size == 1:
        finite_mean = float(np.mean(finite))
        finite_sd = 0.0
        finite_sem = 0.0
    else:
        finite_mean = float(np.mean(finite))
        finite_sd = float(np.std(finite, ddof=1))
        finite_sem = float(finite_sd / np.sqrt(finite.size))
    return {
        'count': int(arr.size),
        'finite_count': int(finite.size),
        'diverged_count': int(arr.size - finite.size),
        'mean_all': float(np.mean(arr)),
        'mean_finite': finite_mean,
        'sd_finite': finite_sd,
        'sem_finite': finite_sem,
    }


def summarize_layer_lists(layer_lists):
    means = []
    sds = []
    sems = []
    counts = []
    for values in layer_lists:
        summary = summarize_values(values)
        means.append(summary['mean_finite'])
        sds.append(summary['sd_finite'])
        sems.append(summary['sem_finite'])
        counts.append(summary['finite_count'])
    return {
        'mean_by_layer': means,
        'sd_by_layer': sds,
        'sem_by_layer': sems,
        'count_by_layer': counts,
    }


def evaluate_learning_rate(lr, optimizer, seeds):
    losses = []
    seed_results = []
    for seed in seeds:
        X, Y = make_data(seed)
        weight_seed = seed + WEIGHT_SEED_OFFSET
        weights_init = init_weights(weight_seed)
        _, final_loss = train(weights_init, X, Y, lr, optimizer)
        losses.append(final_loss)
        seed_results.append({
            'seed': int(seed),
            'data_seed': int(seed),
            'weight_seed': int(weight_seed),
            'final_loss': float(final_loss),
            'converged': bool(np.isfinite(final_loss)),
        })
    loss_summary = summarize_values(losses)
    selection_score = loss_summary['mean_finite'] if loss_summary['finite_count'] > 0 else float('inf')
    return {
        'optimizer': optimizer,
        'lr': float(lr),
        'seed_results': seed_results,
        'losses_by_seed': [float(loss) for loss in losses],
        'finite_runs': loss_summary['finite_count'],
        'diverged_runs': loss_summary['diverged_count'],
        'mean_final_loss': float(selection_score),
        'loss_summary': loss_summary,
    }


def sweep_learning_rates(seeds_subset=None):
    if seeds_subset is None:
        seeds_subset = get_default_seeds()[:LR_SWEEP_NUM_SEEDS]
    seeds_subset = list(seeds_subset)
    sweep_results = {}
    for optimizer, candidates in [('muon', MUON_LRS), ('sgd', SGD_LRS)]:
        per_lr = []
        best_lr = candidates[-1]
        best_score = float('inf')
        for lr in candidates:
            evaluation = evaluate_learning_rate(lr, optimizer, seeds_subset)
            per_lr.append(evaluation)
            if evaluation['mean_final_loss'] < best_score:
                best_score = evaluation['mean_final_loss']
                best_lr = lr
        sweep_results[optimizer] = {
            'optimizer': optimizer,
            'seed_subset': [int(seed) for seed in seeds_subset],
            'candidates': [float(lr) for lr in candidates],
            'results': per_lr,
            'best_lr': float(best_lr),
            'best_mean_final_loss': float(best_score),
        }
    return sweep_results


def evaluate_config(name, optimizer, lr, seeds):
    layer_eranks = [[] for _ in range(NUM_LAYERS)]
    layer_kappas = [[] for _ in range(NUM_LAYERS)]
    final_losses = []
    seed_results = []

    for seed in seeds:
        X, Y = make_data(seed)
        weight_seed = seed + WEIGHT_SEED_OFFSET
        weights_init = init_weights(weight_seed)
        final_weights, final_loss = train(weights_init, X, Y, lr, optimizer)
        final_losses.append(final_loss)

        seed_record = {
            'seed': int(seed),
            'data_seed': int(seed),
            'weight_seed': int(weight_seed),
            'final_loss': float(final_loss),
            'converged': bool(final_weights is not None),
            'effective_rank_by_layer': None,
            'condition_number_by_layer': None,
        }

        if final_weights is not None:
            eranks = []
            kappas = []
            for layer_idx in range(NUM_LAYERS):
                erank = effective_rank(final_weights[layer_idx])
                kappa = condition_number(final_weights[layer_idx])
                eranks.append(erank)
                kappas.append(kappa)
                layer_eranks[layer_idx].append(erank)
                layer_kappas[layer_idx].append(kappa)
            seed_record['effective_rank_by_layer'] = eranks
            seed_record['condition_number_by_layer'] = kappas

        seed_results.append(seed_record)

    loss_summary = summarize_values(final_losses)
    erank_summary = summarize_layer_lists(layer_eranks)
    kappa_summary = summarize_layer_lists(layer_kappas)

    return {
        'name': name,
        'label': CONFIG_LABELS[name],
        'optimizer': optimizer,
        'lr': float(lr),
        'seed_results': seed_results,
        'summary': {
            'num_seeds': len(seed_results),
            'finite_runs': loss_summary['finite_count'],
            'diverged_runs': loss_summary['diverged_count'],
            'loss_mean': loss_summary['mean_all'],
            'loss_mean_finite': loss_summary['mean_finite'],
            'loss_sd_finite': loss_summary['sd_finite'],
            'loss_sem_finite': loss_summary['sem_finite'],
            'effective_rank_mean_by_layer': erank_summary['mean_by_layer'],
            'effective_rank_sd_by_layer': erank_summary['sd_by_layer'],
            'effective_rank_sem_by_layer': erank_summary['sem_by_layer'],
            'effective_rank_count_by_layer': erank_summary['count_by_layer'],
            'condition_number_mean_by_layer': kappa_summary['mean_by_layer'],
            'condition_number_sd_by_layer': kappa_summary['sd_by_layer'],
            'condition_number_sem_by_layer': kappa_summary['sem_by_layer'],
            'condition_number_count_by_layer': kappa_summary['count_by_layer'],
        },
    }


def compute_hypothesis_tests(config_results):
    default_erank_diff = float(np.mean([
        config_results['muon_default']['summary']['effective_rank_mean_by_layer'][layer_idx]
        - config_results['sgd_default']['summary']['effective_rank_mean_by_layer'][layer_idx]
        for layer_idx in range(NUM_LAYERS)
    ]))
    optimal_erank_diff = float(np.mean([
        config_results['muon_optimal']['summary']['effective_rank_mean_by_layer'][layer_idx]
        - config_results['sgd_optimal']['summary']['effective_rank_mean_by_layer'][layer_idx]
        for layer_idx in range(NUM_LAYERS)
    ]))
    absolute_change = abs(default_erank_diff - optimal_erank_diff)
    threshold = abs(default_erank_diff) * 0.5
    relative_change = float('inf') if threshold == 0 and absolute_change > 0 else (absolute_change / max(abs(default_erank_diff), 1e-12))
    t1_passed = absolute_change > threshold

    default_kappa_ratio = float(np.mean([
        config_results['muon_default']['summary']['condition_number_mean_by_layer'][layer_idx]
        / max(config_results['sgd_default']['summary']['condition_number_mean_by_layer'][layer_idx], 1e-12)
        for layer_idx in range(NUM_LAYERS)
    ]))
    optimal_kappa_ratio = float(np.mean([
        config_results['muon_optimal']['summary']['condition_number_mean_by_layer'][layer_idx]
        / max(config_results['sgd_optimal']['summary']['condition_number_mean_by_layer'][layer_idx], 1e-12)
        for layer_idx in range(NUM_LAYERS)
    ]))
    flip = (default_kappa_ratio < 1.0) != (optimal_kappa_ratio < 1.0)

    return {
        't1': {
            'name': 'Effective-rank difference changes materially under LR matching',
            'question': 'Does the mean layerwise Muon-SGD effective-rank difference change by >50% after LR matching?',
            'default_difference_mean': default_erank_diff,
            'optimal_difference_mean': optimal_erank_diff,
            'absolute_change': absolute_change,
            'relative_change_vs_default': float(relative_change),
            'threshold': float(threshold),
            'passed': bool(t1_passed),
            'interpretation': 'PASS (LR confound exists)' if t1_passed else 'FAIL (effective-rank comparison is relatively stable)',
        },
        't2': {
            'name': 'Condition-number ranking flips under LR matching',
            'question': 'Does the mean layerwise Muon/SGD condition-number ratio flip across 1.0 after LR matching?',
            'default_muon_over_sgd_ratio_mean': default_kappa_ratio,
            'optimal_muon_over_sgd_ratio_mean': optimal_kappa_ratio,
            'flip': bool(flip),
            'passed': bool(flip),
            'interpretation': 'PASS (ranking flips; previous comparison is LR-confounded)' if flip else 'FAIL (ranking remains qualitatively stable)',
        },
    }


def run_experiment(seeds=None):
    if seeds is None:
        seeds = get_default_seeds()
    seeds = [int(seed) for seed in seeds]
    lr_sweep_seeds = seeds[:LR_SWEEP_NUM_SEEDS]
    lr_sweep = sweep_learning_rates(lr_sweep_seeds)

    config_specs = [
        {'name': 'sgd_default', 'optimizer': 'sgd', 'lr': DEFAULT_SGD_LR},
        {'name': 'sgd_optimal', 'optimizer': 'sgd', 'lr': lr_sweep['sgd']['best_lr']},
        {'name': 'muon_default', 'optimizer': 'muon', 'lr': DEFAULT_MUON_LR},
        {'name': 'muon_optimal', 'optimizer': 'muon', 'lr': lr_sweep['muon']['best_lr']},
    ]

    config_results = {}
    for spec in config_specs:
        config_results[spec['name']] = evaluate_config(spec['name'], spec['optimizer'], spec['lr'], seeds)

    tests = compute_hypothesis_tests(config_results)

    return {
        'experiment_id': EXPERIMENT_ID,
        'title': EXPERIMENT_TITLE,
        'scope_note': SCOPE_NOTE,
        'scope': {
            'computed_metrics': ['final effective rank', 'final condition number'],
            'alpha_computed': False,
            'alpha_note': 'WeightWatcher/power-law alpha is explicitly out of scope for H16 and is not computed here.',
        },
        'config': {
            'dim': DIM,
            'num_layers': NUM_LAYERS,
            'num_steps': NUM_STEPS,
            'momentum': MOMENTUM,
            'ns_iters': NS_ITERS,
            'batch_size': BATCH_SIZE,
            'num_seeds': len(seeds),
            'seeds': seeds,
            'lr_sweep_num_seeds': LR_SWEEP_NUM_SEEDS,
            'lr_sweep_seeds': lr_sweep_seeds,
            'data_scale': DATA_SCALE,
            'weight_init_scale': WEIGHT_INIT_SCALE,
            'weight_seed_offset': WEIGHT_SEED_OFFSET,
            'default_lrs': {'sgd': DEFAULT_SGD_LR, 'muon': DEFAULT_MUON_LR},
            'candidate_lrs': {'sgd': [float(lr) for lr in SGD_LRS], 'muon': [float(lr) for lr in MUON_LRS]},
        },
        'config_order': [spec['name'] for spec in config_specs],
        'config_specs': [{**spec, 'label': CONFIG_LABELS[spec['name']]} for spec in config_specs],
        'lr_sweep': lr_sweep,
        'configs': config_results,
        'tests': tests,
    }


def format_number(value, precision=3):
    if value is None:
        return 'NA'
    value = float(value)
    if not np.isfinite(value):
        return 'inf' if value > 0 else '-inf'
    if value == 0:
        return '0'
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f'{value:.3e}'
    return f'{value:.{precision}f}'


def print_lr_sweep_summary(results):
    print('Phase 1: LR sweep (selection uses the first 3 seeds)')
    for optimizer in ['sgd', 'muon']:
        sweep = results['lr_sweep'][optimizer]
        print(f"  {optimizer.upper()}: best_lr={sweep['best_lr']:.4f}, best_mean_final_loss={format_number(sweep['best_mean_final_loss'], 4)}")
        for entry in sweep['results']:
            print(
                f"    lr={entry['lr']:.4f} | mean_final_loss={format_number(entry['mean_final_loss'], 4)} "
                f"| finite={entry['finite_runs']}/{len(sweep['seed_subset'])}"
            )


def print_metric_table(results, metric_key, metric_title):
    print()
    print(f'{metric_title}:')
    header = ['Config'] + [f'L{layer_idx}' for layer_idx in range(NUM_LAYERS)] + ['LossMeanFinite', 'Finite']
    rows = []
    for name in results['config_order']:
        config_result = results['configs'][name]
        summary = config_result['summary']
        metric_values = summary[f'{metric_key}_mean_by_layer']
        rows.append([
            config_result['label'],
            *[format_number(value, 2) for value in metric_values],
            format_number(summary['loss_mean_finite'], 4),
            f"{summary['finite_runs']}/{summary['num_seeds']}",
        ])

    widths = [max(len(str(row[col_idx])) for row in [header] + rows) for col_idx in range(len(header))]
    print('  ' + ' | '.join(str(header[col_idx]).ljust(widths[col_idx]) for col_idx in range(len(header))))
    print('  ' + '-+-'.join('-' * widths[col_idx] for col_idx in range(len(header))))
    for row in rows:
        print('  ' + ' | '.join(str(row[col_idx]).ljust(widths[col_idx]) for col_idx in range(len(row))))


def print_test_summary(results):
    t1 = results['tests']['t1']
    t2 = results['tests']['t2']
    print()
    print('Hypothesis tests:')
    print('  T1: Effective-rank difference changes >50% with LR matching?')
    print(f"      default_diff={format_number(t1['default_difference_mean'])}, optimal_diff={format_number(t1['optimal_difference_mean'])}")
    print(f"      abs_change={format_number(t1['absolute_change'])}, threshold={format_number(t1['threshold'])}")
    print(f"      --> {t1['interpretation']}")
    print('  T2: Muon/SGD condition-number ratio flips across 1.0 with LR matching?')
    print(f"      default_ratio={format_number(t2['default_muon_over_sgd_ratio_mean'])}, optimal_ratio={format_number(t2['optimal_muon_over_sgd_ratio_mean'])}")
    print(f"      --> {t2['interpretation']}")


def main():
    results = run_experiment()
    config = results['config']

    print('=' * 100)
    print(EXPERIMENT_TITLE)
    print('=' * 100)
    print(SCOPE_NOTE)
    print(
        f"Network: {config['num_layers']}-layer, {config['dim']}x{config['dim']}, "
        f"{config['num_steps']} steps, batch_size={config['batch_size']}, seeds={config['num_seeds']}"
    )
    print(f"Seeds: {config['seeds']}")
    print(f"LR sweep seeds: {config['lr_sweep_seeds']}")
    print()

    print_lr_sweep_summary(results)

    print()
    print('=' * 100)
    print('Phase 2: Final spectral metrics at default vs chosen LR')
    print('=' * 100)
    print_metric_table(results, 'effective_rank', 'Effective rank (final state)')
    print_metric_table(results, 'condition_number', 'Condition number (final state)')

    print()
    print('=' * 100)
    print('Phase 3: Heuristic falsification tests')
    print('=' * 100)
    print_test_summary(results)

    print()
    print('=' * 100)
    print('Experiment complete')
    print('=' * 100)


if __name__ == '__main__':
    main()
