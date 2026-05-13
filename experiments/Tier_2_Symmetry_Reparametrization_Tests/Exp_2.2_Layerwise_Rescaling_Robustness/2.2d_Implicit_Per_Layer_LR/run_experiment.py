#!/usr/bin/env python3
"""
2.2d: Toy Oracle Per-Layer LR Grid vs Single-LR Muon
====================================================

This experiment preserves the original 4-layer deep-linear c=100 toy study and
asks a narrow operational question:

    In this fixed setting, can exhaustive *discrete* per-layer learning-rate
    selection for SGD match or exceed the best *discrete single-LR* Muon run on
    final training loss?

Operational meaning of "oracle" in this file:
  - each of the 4 layers chooses from a fixed 5-value LR grid
  - the full 5^4 = 625 tuple grid is searched exhaustively
  - model selection uses the first 3 seeds only
  - the selection score is the mean of the *finite* final losses on those seeds
  - if a candidate never yields a finite loss on the selection seeds, it is not
    considered a valid selected candidate

This is a toy optimization comparison, not a general explanation of Muon.
"""

import itertools
import os

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_FILENAME = '2_2d_implicit_per_layer_lr.png'

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM_BETA = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
C = 100
SELECTION_SEED_COUNT = 3
WEIGHT_INIT_SEED_OFFSET = 5000
DIVERGENCE_THRESHOLD = 1e10

# 7 candidates for single-LR methods
SGD_LR_CANDIDATES = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
MUON_LR_CANDIDATES = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003]

# 5 candidates per layer for oracle grid search (5^4 = 625 combos)
PER_LAYER_LR_CANDIDATES = [0.1, 0.01, 0.001, 0.0001, 0.00001]

METHOD_SPECS = [
    ('sgd_single_lr', 'SGD (single LR)'),
    ('muon_single_lr', 'Muon (single LR)'),
    ('sgd_oracle_per_layer', 'SGD (oracle per-layer LR grid)'),
]


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
    for _ in range(N_LAYERS):
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
    for _ in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > DIVERGENCE_THRESHOLD:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + grads[l]
            weights[l] = weights[l] - lr * mom[l]
    return float(compute_loss(weights, X, Y))



def train_muon(weights_init, X, Y, lr):
    """Muon: orthogonalize gradient via Newton-Schulz, then momentum."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > DIVERGENCE_THRESHOLD:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + newton_schulz(grads[l])
            weights[l] = weights[l] - lr * mom[l]
    return float(compute_loss(weights, X, Y))



def train_sgd_per_layer(weights_init, X, Y, per_layer_lrs):
    """SGD with momentum, independent LR per layer."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > DIVERGENCE_THRESHOLD:
            return float('inf')
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM_BETA * mom[l] + grads[l]
            weights[l] = weights[l] - per_layer_lrs[l] * mom[l]
    return float(compute_loss(weights, X, Y))


# =============================================================================
# RESULT HELPERS
# =============================================================================

def get_default_seeds():
    return [42 + i * 137 for i in range(NUM_SEEDS)]



def build_config():
    oracle_grid_size = len(PER_LAYER_LR_CANDIDATES) ** N_LAYERS
    selection_train_runs = SELECTION_SEED_COUNT * (
        len(SGD_LR_CANDIDATES) + len(MUON_LR_CANDIDATES) + oracle_grid_size
    )
    evaluation_train_runs = NUM_SEEDS * len(METHOD_SPECS)
    return {
        'dim': DIM,
        'n_layers': N_LAYERS,
        'num_steps': NUM_STEPS,
        'batch_size': BATCH_SIZE,
        'momentum_beta': MOMENTUM_BETA,
        'ns_iters': NS_ITERS,
        'num_seeds': NUM_SEEDS,
        'selection_seed_count': SELECTION_SEED_COUNT,
        'weight_init_seed_offset': WEIGHT_INIT_SEED_OFFSET,
        'c': C,
        'divergence_threshold': DIVERGENCE_THRESHOLD,
        'sgd_lr_candidates': [float(x) for x in SGD_LR_CANDIDATES],
        'muon_lr_candidates': [float(x) for x in MUON_LR_CANDIDATES],
        'per_layer_lr_candidates': [float(x) for x in PER_LAYER_LR_CANDIDATES],
        'oracle_grid_size': oracle_grid_size,
        'selection_metric': 'mean of finite final losses across selection seeds',
        'selection_divergence_policy': (
            'divergent selection runs are excluded from the selection mean; '
            'if a candidate has no finite selection losses, it is not selected'
        ),
        'final_metric': f'final training loss after {NUM_STEPS} optimization steps',
        'selection_train_runs': selection_train_runs,
        'evaluation_train_runs': evaluation_train_runs,
        'total_train_runs': selection_train_runs + evaluation_train_runs,
    }



def format_number(value):
    if value is None:
        return 'None'
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return 'inf'
    return f'{float(value):.6e}'



def format_lr_spec(value):
    if value is None:
        return 'None'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(f'{float(v):.1e}' for v in value) + ']'
    return f'{float(value):.1e}'



def summarize_losses(losses):
    losses = [float(loss) for loss in losses]
    finite = [loss for loss in losses if np.isfinite(loss)]
    if finite:
        mean_loss = float(np.mean(finite))
        std_loss = float(np.std(finite)) if len(finite) > 1 else 0.0
    else:
        mean_loss = float('inf')
        std_loss = 0.0
    num_total = len(losses)
    num_finite = len(finite)
    finite_fraction = float(num_finite / num_total) if num_total else 0.0
    return {
        'mean_finite_loss': mean_loss,
        'std_finite_loss': std_loss,
        'num_finite': num_finite,
        'num_total': num_total,
        'finite_fraction': finite_fraction,
        'finite_percentage': 100.0 * finite_fraction,
        'all_finite': num_finite == num_total,
    }



def summarize_seed_rows(rows):
    return summarize_losses([row['loss'] for row in rows])



def run_training_on_seed(seed, train_callable):
    X, Y = make_data(seed)
    weights = init_weights(seed + WEIGHT_INIT_SEED_OFFSET, C)
    loss = float(train_callable(weights, X, Y))
    return {
        'seed': int(seed),
        'loss': loss,
        'finite': bool(np.isfinite(loss)),
    }



def evaluate_rows_for_single_lr(train_fn, seeds, lr):
    return [
        run_training_on_seed(seed, lambda w, X, Y, lr=lr: train_fn(w, X, Y, lr))
        for seed in seeds
    ]



def evaluate_rows_for_per_layer_lr(seeds, per_layer_lrs):
    return [
        run_training_on_seed(
            seed,
            lambda w, X, Y, per_layer_lrs=per_layer_lrs: train_sgd_per_layer(w, X, Y, per_layer_lrs),
        )
        for seed in seeds
    ]



def subset_rows(rows, subset_seeds):
    subset_set = set(subset_seeds)
    return [row for row in rows if row['seed'] in subset_set]



def build_single_lr_record(lr, rows):
    summary = summarize_seed_rows(rows)
    return {
        'lr': float(lr),
        'losses_by_seed': rows,
        **summary,
    }



def build_per_layer_lr_record(per_layer_lrs, rows):
    summary = summarize_seed_rows(rows)
    return {
        'per_layer_lrs': [float(x) for x in per_layer_lrs],
        'losses_by_seed': rows,
        **summary,
    }


# =============================================================================
# LR SWEEP HELPERS
# =============================================================================

def sweep_single_lr(train_fn, selection_seeds, candidates, method_label, verbose=False):
    records = []
    best_lr = None
    best_loss = float('inf')

    for lr in candidates:
        rows = evaluate_rows_for_single_lr(train_fn, selection_seeds, lr)
        record = build_single_lr_record(lr, rows)
        records.append(record)
        if record['num_finite'] > 0 and record['mean_finite_loss'] < best_loss:
            best_loss = record['mean_finite_loss']
            best_lr = float(lr)

    status = 'selected_finite_candidate' if best_lr is not None else 'no_finite_candidate'
    best_record = None
    if best_lr is not None:
        for record in records:
            if record['lr'] == best_lr:
                best_record = record
                break

    if verbose:
        print(f'  {method_label} single-LR sweep...')
        for record in records:
            print(
                '    '
                f"lr={record['lr']:<7g}  "
                f"mean_finite={format_number(record['mean_finite_loss'])}  "
                f"finite={record['num_finite']}/{record['num_total']}"
            )
        if best_lr is None:
            print('    No finite candidate found on the selection seeds.')
        else:
            print(
                '    Selected: '
                f"lr={best_lr} with selection mean {format_number(best_record['mean_finite_loss'])}"
            )

    return {
        'method_label': method_label,
        'selection_seeds': [int(seed) for seed in selection_seeds],
        'selection_status': status,
        'selection_metric': 'mean_finite_loss',
        'records': records,
        'num_candidates': len(records),
        'best_lr': best_lr,
        'best_mean_finite_loss': float(best_loss) if best_lr is not None else float('inf'),
    }



def sweep_oracle_per_layer(selection_seeds, verbose=False, progress_interval=100):
    combos = list(itertools.product(PER_LAYER_LR_CANDIDATES, repeat=N_LAYERS))
    records = []
    best_combo = None
    best_loss = float('inf')

    if verbose:
        print('  SGD oracle per-layer LR sweep...')
        print(f'    Exhaustive discrete grid: {len(combos)} combos')

    for idx, combo in enumerate(combos):
        per_layer_lrs = [float(x) for x in combo]
        rows = evaluate_rows_for_per_layer_lr(selection_seeds, per_layer_lrs)
        record = build_per_layer_lr_record(per_layer_lrs, rows)
        records.append(record)
        if record['num_finite'] > 0 and record['mean_finite_loss'] < best_loss:
            best_loss = record['mean_finite_loss']
            best_combo = per_layer_lrs

        if verbose and (idx + 1) % progress_interval == 0:
            print(
                f'    {idx + 1}/{len(combos)} combos evaluated; '
                f'best mean so far = {format_number(best_loss)}'
            )

    status = 'selected_finite_candidate' if best_combo is not None else 'no_finite_candidate'
    best_record = None
    if best_combo is not None:
        for record in records:
            if record['per_layer_lrs'] == best_combo:
                best_record = record
                break

    if verbose:
        if best_combo is None:
            print('    No finite per-layer LR tuple found on the selection seeds.')
        else:
            print(f'    Selected: lrs={format_lr_spec(best_combo)}')
            print(f"    Selection mean: {format_number(best_record['mean_finite_loss'])}")

    return {
        'method_label': 'SGD (oracle per-layer LR grid)',
        'selection_seeds': [int(seed) for seed in selection_seeds],
        'selection_status': status,
        'selection_metric': 'mean_finite_loss',
        'records': records,
        'num_candidates': len(records),
        'best_per_layer_lrs': best_combo,
        'best_mean_finite_loss': float(best_loss) if best_combo is not None else float('inf'),
    }


# =============================================================================
# EVALUATION / INTERPRETATION
# =============================================================================

def evaluate_selected_method(method_key, label, seeds, selection_seeds, held_out_seeds, selected_spec):
    if method_key == 'sgd_single_lr':
        selected_lr = selected_spec
        if selected_lr is None:
            rows = [{'seed': int(seed), 'loss': float('inf'), 'finite': False} for seed in seeds]
            status = 'skipped_no_selected_candidate'
        else:
            rows = evaluate_rows_for_single_lr(train_sgd, seeds, selected_lr)
            status = 'completed'
        selected_hyperparameter = selected_lr

    elif method_key == 'muon_single_lr':
        selected_lr = selected_spec
        if selected_lr is None:
            rows = [{'seed': int(seed), 'loss': float('inf'), 'finite': False} for seed in seeds]
            status = 'skipped_no_selected_candidate'
        else:
            rows = evaluate_rows_for_single_lr(train_muon, seeds, selected_lr)
            status = 'completed'
        selected_hyperparameter = selected_lr

    elif method_key == 'sgd_oracle_per_layer':
        selected_lrs = selected_spec
        if selected_lrs is None:
            rows = [{'seed': int(seed), 'loss': float('inf'), 'finite': False} for seed in seeds]
            status = 'skipped_no_selected_candidate'
        else:
            rows = evaluate_rows_for_per_layer_lr(seeds, selected_lrs)
            status = 'completed'
        selected_hyperparameter = [float(x) for x in selected_lrs] if selected_lrs is not None else None

    else:
        raise ValueError(f'Unknown method_key: {method_key}')

    return {
        'method_key': method_key,
        'label': label,
        'selected_hyperparameter': selected_hyperparameter,
        'selected_hyperparameter_display': format_lr_spec(selected_hyperparameter),
        'evaluation_status': status,
        'losses_by_seed': rows,
        'all_seeds': summarize_seed_rows(rows),
        'selection_seeds': summarize_seed_rows(subset_rows(rows, selection_seeds)),
        'held_out_seeds': summarize_seed_rows(subset_rows(rows, held_out_seeds)),
    }



def compute_initial_diagnostics(seed, oracle_lrs):
    X, Y = make_data(seed)
    weights = init_weights(seed + WEIGHT_INIT_SEED_OFFSET, C)
    grads = compute_gradients(weights, X, Y)
    layers = []
    for layer_idx in range(N_LAYERS):
        grad_norm = float(np.linalg.norm(grads[layer_idx], ord='fro'))
        oracle_lr = None if oracle_lrs is None else float(oracle_lrs[layer_idx])
        layers.append({
            'layer': layer_idx,
            'gradient_fro_norm': grad_norm,
            'oracle_lr': oracle_lr,
        })
    return {
        'seed': int(seed),
        'layers': layers,
    }



def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if np.isfinite(denominator):
        if not np.isfinite(numerator):
            return float('inf')
        return float(numerator / max(denominator, 1e-30))
    if np.isfinite(numerator):
        return 0.0
    return None



def compute_hypothesis_metrics(evaluations):
    sgd_mean = evaluations['sgd_single_lr']['all_seeds']['mean_finite_loss']
    muon_mean = evaluations['muon_single_lr']['all_seeds']['mean_finite_loss']
    oracle_mean = evaluations['sgd_oracle_per_layer']['all_seeds']['mean_finite_loss']

    ratio_oracle_to_muon = safe_ratio(oracle_mean, muon_mean)
    ratio_sgd_to_muon = safe_ratio(sgd_mean, muon_mean)
    oracle_matches_muon_within_2x = (
        ratio_oracle_to_muon is not None
        and np.isfinite(ratio_oracle_to_muon)
        and ratio_oracle_to_muon < 2.0
    )

    percent_explained = None
    percent_explained_status = 'not_computed'
    if ratio_sgd_to_muon is None or not np.isfinite(ratio_sgd_to_muon):
        percent_explained_status = 'undefined_nonfinite_single_lr_sgd_reference'
    elif ratio_sgd_to_muon <= 1.0:
        percent_explained_status = 'undefined_single_lr_sgd_not_worse_than_muon'
    elif ratio_oracle_to_muon is None or not np.isfinite(ratio_oracle_to_muon):
        percent_explained_status = 'undefined_nonfinite_oracle_ratio'
    elif ratio_oracle_to_muon <= 1.0:
        percent_explained = 100.0
        percent_explained_status = 'oracle_at_or_better_than_muon'
    else:
        percent_explained = float(
            (1.0 - (ratio_oracle_to_muon - 1.0) / (ratio_sgd_to_muon - 1.0)) * 100.0
        )
        percent_explained_status = 'computed'

    return {
        'ratio_oracle_to_muon': ratio_oracle_to_muon,
        'ratio_sgd_to_muon': ratio_sgd_to_muon,
        'oracle_matches_muon_within_2x': oracle_matches_muon_within_2x,
        'percent_explained_heuristic': percent_explained,
        'percent_explained_status': percent_explained_status,
        'operational_test': (
            'Does the selected discrete oracle per-layer LR SGD baseline achieve '
            'mean final loss within 2x of the selected discrete single-LR Muon baseline?'
        ),
    }



def build_interpretation(results):
    metrics = results['hypothesis_metrics']
    ratio_oracle_to_muon = metrics['ratio_oracle_to_muon']
    ratio_sgd_to_muon = metrics['ratio_sgd_to_muon']

    if metrics['oracle_matches_muon_within_2x']:
        if ratio_oracle_to_muon is not None and np.isfinite(ratio_oracle_to_muon) and ratio_oracle_to_muon <= 1.0:
            headline = (
                'In this fixed c=100 toy setting, the discrete oracle per-layer LR '
                'SGD baseline matches and in fact exceeds the selected single-LR Muon baseline '
                'on final training loss.'
            )
        else:
            headline = (
                'In this fixed c=100 toy setting, the discrete oracle per-layer LR '
                'SGD baseline matches the selected single-LR Muon baseline within the '
                'chosen 2x criterion on final training loss.'
            )
        summary_lines = [
            f'Oracle/Muon mean-loss ratio: {format_number(ratio_oracle_to_muon)}x.',
            'This supports only a narrow claim: in this setup and on this metric, '
            'per-layer LR tuning is sufficient to remove the observed Muon advantage.',
            'It does not establish that Muon is generally or mechanistically "just" '
            'implicit per-layer LR adaptation.',
        ]
        final_conclusion = (
            'In this fixed c=100 deep-linear toy study, exhaustive discrete per-layer LR '
            'tuning for SGD is enough to match or exceed the selected discrete single-LR '
            'Muon baseline on final training loss. This supports only the narrow claim that '
            'per-layer LR tuning can remove the observed advantage in this setup; it does '
            'not show that Muon is generally or mechanistically just implicit per-layer LR '
            'adaptation.'
        )
    else:
        headline = (
            'In this fixed c=100 toy setting, the discrete oracle per-layer LR SGD '
            'baseline does not match the selected single-LR Muon baseline on final training loss.'
        )
        summary_lines = [
            f'Oracle/Muon mean-loss ratio: {format_number(ratio_oracle_to_muon)}x.',
            'This suggests that simple per-layer LR retuning alone is insufficient here.',
            'Even so, this remains a toy result and does not by itself prove a general '
            'directional or mechanistic explanation for Muon.',
        ]
        final_conclusion = (
            'In this fixed c=100 deep-linear toy study, exhaustive discrete per-layer LR '
            'tuning for SGD does not match the selected discrete single-LR Muon baseline on '
            'final training loss. This suggests that simple per-layer LR retuning alone is '
            'insufficient here, but it still does not isolate or prove a general Muon '
            'mechanism beyond this toy setup.'
        )

    if ratio_sgd_to_muon is not None:
        summary_lines.append(f'Single-LR SGD / Muon mean-loss ratio: {format_number(ratio_sgd_to_muon)}x.')

    caveats = [
        'The comparison is limited to one architecture family, one synthetic task family, one step budget, and one rescaling value c=100.',
        'The oracle baseline is only an exhaustive search over the fixed discrete grid in this file; it is not a continuous optimum.',
        'Selection uses the first 3 seeds only, and the selection score averages only finite losses.',
        'The main reported metric is final training loss; there is no trajectory-level, test-set, or statistical uncertainty analysis beyond seed summaries.',
    ]

    return {
        'headline': headline,
        'summary_lines': summary_lines,
        'caveats': caveats,
        'final_conclusion': final_conclusion,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def ensure_matplotlib_backend():
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if os.environ.get('DISPLAY', '') == '' and 'inline' not in backend and 'agg' not in backend:
        matplotlib.use('Agg', force=True)
    return matplotlib



def proxy_plot_value(losses):
    finite_positive = [loss for loss in losses if np.isfinite(loss) and loss > 0]
    if finite_positive:
        return max(finite_positive) * 10.0
    return 1.0



def save_summary_plot(results, output_dir=None):
    try:
        ensure_matplotlib_backend()
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            'saved': False,
            'path': None,
            'status': 'matplotlib_unavailable',
        }

    output_dir = SCRIPT_DIR if output_dir is None else os.fspath(output_dir)
    plot_path = os.path.join(output_dir, PLOT_FILENAME)

    evaluations = results['evaluations']
    sgd_summary = evaluations['sgd_single_lr']['all_seeds']
    muon_summary = evaluations['muon_single_lr']['all_seeds']
    oracle_summary = evaluations['sgd_oracle_per_layer']['all_seeds']
    diagnostics = results['initial_diagnostics']['layers']

    labels = [
        'SGD\n(single LR)',
        'SGD\n(oracle per-layer)',
        'Muon\n(single LR)',
    ]
    summaries = [sgd_summary, oracle_summary, muon_summary]
    display_losses = [summary['mean_finite_loss'] for summary in summaries]
    plot_values = []
    annotations = []
    proxy = proxy_plot_value(display_losses)
    for summary in summaries:
        mean_loss = summary['mean_finite_loss']
        if np.isfinite(mean_loss) and mean_loss > 0:
            plot_values.append(mean_loss)
            annotations.append(f'{mean_loss:.2e}')
        else:
            plot_values.append(proxy)
            annotations.append(f'non-finite\n{summary["num_finite"]}/{summary["num_total"]} finite')

    stds = [
        summary['std_finite_loss'] if np.isfinite(summary['mean_finite_loss']) else 0.0
        for summary in summaries
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        '2.2d: Discrete Oracle Per-Layer LR Grid vs Single-LR Muon\n'
        f'Toy 4-layer linear network, c={C}, {NUM_STEPS} steps',
        fontsize=13,
        fontweight='bold',
    )

    ax = axes[0]
    colors = ['#4477AA', '#FF8800', '#CC3311']
    bars = ax.bar(range(3), plot_values, yerr=stds, color=colors, edgecolor='black', capsize=5, width=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean final training loss across finite runs')
    ax.set_yscale('log')
    ax.set_title('All-seed final-loss summary')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, annotation in zip(bars, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            annotation,
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

    ax = axes[1]
    x_pos = np.arange(N_LAYERS)
    gradient_norms = [layer['gradient_fro_norm'] for layer in diagnostics]
    oracle_lrs = [layer['oracle_lr'] if layer['oracle_lr'] is not None else np.nan for layer in diagnostics]
    ax2 = ax.twinx()
    ax.bar(x_pos - 0.15, gradient_norms, width=0.3, color='#4477AA', edgecolor='black', alpha=0.75, label='||G||_F at init')
    ax2.bar(x_pos + 0.15, oracle_lrs, width=0.3, color='#CC3311', edgecolor='black', alpha=0.75, label='Selected oracle LR')
    ax.set_xlabel('Layer')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Layer {layer_idx}' for layer_idx in range(N_LAYERS)])
    ax.set_ylabel('Gradient Frobenius norm', color='#4477AA')
    ax2.set_ylabel('Selected oracle LR', color='#CC3311')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_title('Initialization gradient scale vs selected per-layer LR')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'saved': True,
        'path': plot_path,
        'status': 'saved',
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_header(results):
    config = results['config']
    seeds = results['seeds']
    print('=' * 100)
    print('2.2d: TOY ORACLE PER-LAYER LR GRID VS SINGLE-LR MUON')
    print('=' * 100)
    print(f"Network: {config['n_layers']}-layer deep linear, {config['dim']}x{config['dim']}, {config['num_steps']} steps")
    print(f"Rescaling: layer 0 *= {config['c']}, layer {config['n_layers'] - 1} *= {1.0 / config['c']}")
    print(f"Seeds (all):        {seeds['all']}")
    print(f"Seeds (selection):  {seeds['selection']}")
    print(f"Seeds (held out):   {seeds['held_out']}")
    print(f"SGD LR candidates:  {config['sgd_lr_candidates']}")
    print(f"Muon LR candidates: {config['muon_lr_candidates']}")
    print(
        f"Oracle LR grid:     {config['per_layer_lr_candidates']} "
        f"({config['oracle_grid_size']} tuples)"
    )
    print(f"Selection metric:   {config['selection_metric']}")
    print(f"Selection policy:   {config['selection_divergence_policy']}")
    print()



def print_phase_two_summary(results):
    evaluations = results['evaluations']
    print('\nPhase 2: Full evaluation with selected hyperparameters...')
    for method_key, label in METHOD_SPECS:
        evaluation = evaluations[method_key]
        summary = evaluation['all_seeds']
        print(
            f"  {label:>34}: mean={format_number(summary['mean_finite_loss'])}  "
            f"std={format_number(summary['std_finite_loss'])}  "
            f"finite={summary['num_finite']}/{summary['num_total']}  "
            f"selected={evaluation['selected_hyperparameter_display']}"
        )



def print_results_tables(results):
    evaluations = results['evaluations']

    print(f"\n\n{'=' * 100}")
    print('RESULTS: ALL SEEDS')
    print(f"{'=' * 100}")
    print(
        f"\n  {'Method':>34}  {'Mean loss':>14}  {'Std':>14}  {'Finite':>10}  {'Selected LR(s)':>24}"
    )
    print('  ' + '-' * 108)
    for method_key, label in METHOD_SPECS:
        evaluation = evaluations[method_key]
        summary = evaluation['all_seeds']
        print(
            f"  {label:>34}  {format_number(summary['mean_finite_loss']):>14}  "
            f"{format_number(summary['std_finite_loss']):>14}  "
            f"{summary['num_finite']:>3}/{summary['num_total']:<6}  "
            f"{evaluation['selected_hyperparameter_display']:>24}"
        )

    print(f"\n\n{'=' * 100}")
    print('RESULTS: HELD-OUT SEEDS ONLY')
    print(f"{'=' * 100}")
    print(
        f"\n  {'Method':>34}  {'Mean loss':>14}  {'Std':>14}  {'Finite':>10}"
    )
    print('  ' + '-' * 82)
    for method_key, label in METHOD_SPECS:
        summary = evaluations[method_key]['held_out_seeds']
        print(
            f"  {label:>34}  {format_number(summary['mean_finite_loss']):>14}  "
            f"{format_number(summary['std_finite_loss']):>14}  "
            f"{summary['num_finite']:>3}/{summary['num_total']:<6}"
        )



def print_initial_diagnostics(results):
    diagnostics = results['initial_diagnostics']
    print(f"\n  Initialization diagnostics (seed {diagnostics['seed']}):")
    for layer in diagnostics['layers']:
        print(
            f"    Layer {layer['layer']}: ||G||_F = {layer['gradient_fro_norm']:.4e}  "
            f"(selected oracle LR = {format_lr_spec(layer['oracle_lr'])})"
        )



def print_hypothesis_section(results):
    metrics = results['hypothesis_metrics']
    print(f"\n\n{'=' * 100}")
    print('OPERATIONAL TOY-SCOPE VERDICT')
    print(f"{'=' * 100}")
    print(f"\n  Test: {metrics['operational_test']}")
    print(f"  Oracle / Muon ratio: {format_number(metrics['ratio_oracle_to_muon'])}x")
    print(f"  SGD / Muon ratio:    {format_number(metrics['ratio_sgd_to_muon'])}x")
    verdict = 'PASS' if metrics['oracle_matches_muon_within_2x'] else 'FAIL'
    print(f"  Within-2x verdict:   {verdict}")
    if metrics['percent_explained_heuristic'] is None:
        print(
            '  Percent-explained heuristic: not reported '
            f"({metrics['percent_explained_status']})"
        )
    else:
        print(
            '  Percent-explained heuristic: '
            f"{metrics['percent_explained_heuristic']:.2f}% "
            f"({metrics['percent_explained_status']})"
        )



def print_conclusion(results):
    interpretation = results['interpretation']
    print(f"\n\n{'=' * 100}")
    print('CALIBRATED CONCLUSION')
    print(f"{'=' * 100}")
    print(f"\n  {interpretation['headline']}")
    print(f"\n  Bottom line: {interpretation['final_conclusion']}")
    print('\n  What this result supports:')
    for line in interpretation['summary_lines']:
        print(f'    - {line}')
    print('\n  What this result does not establish:')
    for line in interpretation['caveats']:
        print(f'    - {line}')


# =============================================================================
# MAIN EXPERIMENT API
# =============================================================================

def run_experiment(save_plot=True, verbose=True, output_dir=None):
    seeds = get_default_seeds()
    selection_seeds = seeds[:SELECTION_SEED_COUNT]
    held_out_seeds = seeds[SELECTION_SEED_COUNT:]

    results = {
        'metadata': {
            'experiment_id': '2.2d_Implicit_Per_Layer_LR',
            'title': 'Toy Oracle Per-Layer LR Grid vs Single-LR Muon',
            'toy_scope_question': (
                'In the fixed c=100 deep-linear toy setting, can exhaustive discrete '
                'per-layer LR search for SGD match or exceed the best discrete single-LR '
                'Muon baseline on final training loss?'
            ),
            'oracle_definition': (
                'Exhaustive 5^4 discrete grid over per-layer learning-rate tuples, '
                'selected on the first 3 seeds.'
            ),
        },
        'paths': {
            'script_dir': SCRIPT_DIR,
            'default_plot_path': os.path.join(SCRIPT_DIR, PLOT_FILENAME),
        },
        'config': build_config(),
        'seeds': {
            'all': [int(seed) for seed in seeds],
            'selection': [int(seed) for seed in selection_seeds],
            'held_out': [int(seed) for seed in held_out_seeds],
        },
    }

    if verbose:
        print_header(results)
        print('Phase 1: Learning-rate selection...')

    sgd_sweep = sweep_single_lr(
        train_sgd,
        selection_seeds,
        SGD_LR_CANDIDATES,
        method_label='SGD',
        verbose=verbose,
    )
    muon_sweep = sweep_single_lr(
        train_muon,
        selection_seeds,
        MUON_LR_CANDIDATES,
        method_label='Muon',
        verbose=verbose,
    )
    oracle_sweep = sweep_oracle_per_layer(selection_seeds, verbose=verbose)

    results['sweeps'] = {
        'sgd_single_lr': sgd_sweep,
        'muon_single_lr': muon_sweep,
        'sgd_oracle_per_layer': oracle_sweep,
    }
    results['selected_hyperparameters'] = {
        'sgd_single_lr': sgd_sweep['best_lr'],
        'muon_single_lr': muon_sweep['best_lr'],
        'sgd_oracle_per_layer': oracle_sweep['best_per_layer_lrs'],
    }

    evaluations = {
        'sgd_single_lr': evaluate_selected_method(
            'sgd_single_lr',
            'SGD (single LR)',
            seeds,
            selection_seeds,
            held_out_seeds,
            sgd_sweep['best_lr'],
        ),
        'muon_single_lr': evaluate_selected_method(
            'muon_single_lr',
            'Muon (single LR)',
            seeds,
            selection_seeds,
            held_out_seeds,
            muon_sweep['best_lr'],
        ),
        'sgd_oracle_per_layer': evaluate_selected_method(
            'sgd_oracle_per_layer',
            'SGD (oracle per-layer LR grid)',
            seeds,
            selection_seeds,
            held_out_seeds,
            oracle_sweep['best_per_layer_lrs'],
        ),
    }
    results['evaluations'] = evaluations

    results['initial_diagnostics'] = compute_initial_diagnostics(
        seed=selection_seeds[0],
        oracle_lrs=oracle_sweep['best_per_layer_lrs'],
    )
    results['hypothesis_metrics'] = compute_hypothesis_metrics(evaluations)
    results['interpretation'] = build_interpretation(results)

    if save_plot:
        results['plot'] = save_summary_plot(results, output_dir=output_dir)
    else:
        results['plot'] = {
            'saved': False,
            'path': None,
            'status': 'not_requested',
        }

    if verbose:
        print_phase_two_summary(results)
        print_results_tables(results)
        print_initial_diagnostics(results)
        print_hypothesis_section(results)
        print_conclusion(results)
        if results['plot']['saved']:
            print(f"\nPlot saved: {results['plot']['path']}")
        elif results['plot']['status'] == 'matplotlib_unavailable':
            print('\nWARNING: matplotlib not available, skipping plot save.')

    return results



def main():
    run_experiment(save_plot=True, verbose=True, output_dir=SCRIPT_DIR)


if __name__ == '__main__':
    main()
