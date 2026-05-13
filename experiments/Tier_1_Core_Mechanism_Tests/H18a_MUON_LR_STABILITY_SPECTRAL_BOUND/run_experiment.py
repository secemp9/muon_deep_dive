#!/usr/bin/env python3
"""
H18a: toy LR robustness study with a step-1 spectral-bound diagnostic
========================================================================

This toy companion study compares momentum SGD and Muon on 32x32 deep linear
networks across depth. It performs discrete learning-rate sweeps, selects the
best grid LR by final loss, and records step-1 operator norms of the resulting
updates.

Important calibration:
- The reported learning rates are best grid LRs from a finite sweep, not true
  max-stable learning rates.
- The mechanistically relevant diagnostic is the normalized step-1 operator norm
  ||ΔW||_op / lr rather than the raw ||ΔW||_op.
- Results are intended as evidence consistent with a spectral-bound explanation
  in this toy setting, not as a proof or a universal depth-scaling law.
"""

import numpy as np

DIM = 32
DEPTHS = [2, 3, 4, 6, 8, 12, 16]
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64
SELECTION_SEED_COUNT = 3

# Dense LR grids: 20 candidates each, log-spaced
SGD_LR_GRID = np.logspace(-5, -1, 20)
MUON_LR_GRID = np.logspace(-4, -1, 20)


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(depth, seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(depth)]


def forward(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out


def compute_loss(weights, X, Y):
    pred = forward(weights, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def compute_gradients(weights, X, Y):
    depth = len(weights)
    batch_size = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / batch_size
    grads = [None] * depth
    for layer in range(depth - 1, -1, -1):
        grads[layer] = delta @ acts[layer].T
        if layer > 0:
            delta = weights[layer].T @ delta
    return grads


def make_data(seed):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


def train(weights_init, X, Y, lr, optimizer, num_steps=NUM_STEPS):
    """Train and return (final_loss, step1_update_op_norms)."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    step1_op_norms = None

    for step in range(num_steps):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf'), None

        grads = compute_gradients(weights, X, Y)
        deltas = []
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            delta = lr * mom[i]
            deltas.append(delta)
            weights[i] = weights[i] - delta

        if step == 0:
            step1_op_norms = [float(np.linalg.svd(d, compute_uv=False)[0]) for d in deltas]

    return float(compute_loss(weights, X, Y)), step1_op_norms


def _safe_mean(values, default=float('nan')):
    return float(np.mean(values)) if values else default


def _safe_std(values, default=float('nan')):
    return float(np.std(values)) if values else default


def _fit_lr_scaling(depths, best_lrs):
    depths_arr = np.asarray(depths, dtype=float)
    best_lrs_arr = np.asarray(best_lrs, dtype=float)
    log_depths = np.log(depths_arr)
    log_lrs = np.log(best_lrs_arr)

    slope, intercept = np.polyfit(log_depths, log_lrs, 1)
    predicted_log_lrs = slope * log_depths + intercept

    ss_res = np.sum((log_lrs - predicted_log_lrs) ** 2)
    ss_tot = np.sum((log_lrs - np.mean(log_lrs)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0.0

    return {
        'depths': [int(d) for d in depths_arr.tolist()],
        'best_lrs': [float(x) for x in best_lrs_arr.tolist()],
        'predicted_best_lrs': [float(x) for x in np.exp(predicted_log_lrs).tolist()],
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'lr_ratio': float(np.max(best_lrs_arr) / np.min(best_lrs_arr)),
        'min_best_lr': float(np.min(best_lrs_arr)),
        'max_best_lr': float(np.max(best_lrs_arr)),
    }


def _build_config(depths, sgd_lr_grid, muon_lr_grid, seeds, selection_seed_count, num_steps):
    estimated_training_runs = (
        len(depths) * (len(sgd_lr_grid) + len(muon_lr_grid)) * selection_seed_count
        + len(depths) * 2 * len(seeds)
    )
    return {
        'dim': DIM,
        'depths': [int(d) for d in depths],
        'num_steps': int(num_steps),
        'momentum': float(MOMENTUM),
        'ns_iters': int(NS_ITERS),
        'num_seeds': int(len(seeds)),
        'selection_seed_count': int(selection_seed_count),
        'batch_size': int(BATCH_SIZE),
        'seeds': [int(s) for s in seeds],
        'sgd_lr_grid': [float(x) for x in sgd_lr_grid],
        'muon_lr_grid': [float(x) for x in muon_lr_grid],
        'estimated_training_runs': int(estimated_training_runs),
        'scope_note': 'Toy deep linear network study; reports best grid LR from a discrete sweep.',
        'mechanism_note': 'Primary mechanism diagnostic is normalized step-1 operator norm ||ΔW||_op / lr.',
    }


def _build_analysis(depths, records):
    records_by_optimizer = {
        opt: sorted(
            [record for record in records if record['optimizer'] == opt],
            key=lambda record: record['depth'],
        )
        for opt in ('sgd', 'muon')
    }

    fits = {}
    for opt, opt_records in records_by_optimizer.items():
        best_lrs = [record['best_lr'] for record in opt_records]
        fits[opt] = _fit_lr_scaling(depths, best_lrs)

    sgd_ratio = fits['sgd']['lr_ratio']
    muon_ratio = fits['muon']['lr_ratio']
    variability_ratio = sgd_ratio / muon_ratio

    verdict = {
        'sgd_lr_ratio': float(sgd_ratio),
        'muon_lr_ratio': float(muon_ratio),
        'sgd_to_muon_variability_ratio': float(variability_ratio),
        'tests': {
            'T1_muon_lr_ratio_lt_5': {
                'description': 'Muon best-grid LR varies by less than 5x across depths.',
                'value': float(muon_ratio),
                'threshold': 5.0,
                'comparison': '<',
                'pass': bool(muon_ratio < 5.0),
            },
            'T2_sgd_lr_ratio_gt_20': {
                'description': 'SGD best-grid LR varies by more than 20x across depths.',
                'value': float(sgd_ratio),
                'threshold': 20.0,
                'comparison': '>',
                'pass': bool(sgd_ratio > 20.0),
            },
            'T3_variability_ratio_gt_10': {
                'description': 'SGD/Muon LR variability ratio exceeds 10.',
                'value': float(variability_ratio),
                'threshold': 10.0,
                'comparison': '>',
                'pass': bool(variability_ratio > 10.0),
            },
        },
    }

    return {
        'fits': fits,
        'verdict': verdict,
    }


def _print_header(config):
    print('=' * 100)
    print('H18a: TOY LR ROBUSTNESS STUDY WITH A STEP-1 SPECTRAL-BOUND DIAGNOSTIC')
    print('=' * 100)
    print(f"Depths: {config['depths']}")
    print(
        f"SGD LR grid: {len(config['sgd_lr_grid'])} candidates "
        f"[{config['sgd_lr_grid'][0]:.1e} .. {config['sgd_lr_grid'][-1]:.1e}]"
    )
    print(
        f"Muon LR grid: {len(config['muon_lr_grid'])} candidates "
        f"[{config['muon_lr_grid'][0]:.1e} .. {config['muon_lr_grid'][-1]:.1e}]"
    )
    print(f"Seeds: {config['seeds']}")
    print(f"Estimated training runs: {config['estimated_training_runs']}")
    print('Calibration: reports best grid LR from a discrete sweep, not a true max-stable LR boundary.')
    print('Mechanism diagnostic: mean step-1 max ||ΔW||_op / lr at the selected best grid LR.')


def _print_final_analysis(payload):
    config = payload['config']
    records = payload['results']['table']
    analysis = payload['analysis']

    print(f"\n\n{'=' * 100}")
    print('ANALYSIS: BEST GRID LR VS DEPTH')
    print(f"{'=' * 100}")

    for opt in ('sgd', 'muon'):
        fit = analysis['fits'][opt]
        print(f"\n  {opt.upper()}:")
        print(f"    log-log slope: {fit['slope']:.3f}  (exploratory discrete-grid fit)")
        print(f"    R^2: {fit['r2']:.4f}")
        print(
            f"    LR range: [{fit['min_best_lr']:.6f}, {fit['max_best_lr']:.6f}]  "
            f"ratio: {fit['lr_ratio']:.1f}x"
        )
        print('    Per-depth best grid LR: ', end='')
        for depth in config['depths']:
            record = next(r for r in records if r['depth'] == depth and r['optimizer'] == opt)
            print(f"L={depth}:{record['best_lr']:.5f}  ", end='')
        print()

    print(f"\n\n{'=' * 100}")
    print('ANALYSIS: NORMALIZED STEP-1 UPDATE OPERATOR NORM')
    print(f"{'=' * 100}")
    print('Primary mechanism diagnostic: mean max ||ΔW||_op / lr at the selected best grid LR.')

    for opt in ('sgd', 'muon'):
        print(f"\n  {opt.upper()} mean max ||ΔW||_op / lr:")
        for depth in config['depths']:
            record = next(r for r in records if r['depth'] == depth and r['optimizer'] == opt)
            print(f"    L={depth:>2}: {record['mean_max_step1_op_norm_over_lr']:.4e}")

    print(f"\n\n{'=' * 100}")
    print('VERDICT')
    print(f"{'=' * 100}")

    verdict = analysis['verdict']
    tests = verdict['tests']

    print(
        f"\n  T1: Muon LR varies < 5x across depths? ratio={verdict['muon_lr_ratio']:.1f}x  "
        f"--> {'PASS' if tests['T1_muon_lr_ratio_lt_5']['pass'] else 'FAIL'}"
    )
    print(
        f"  T2: SGD LR varies > 20x across depths?  ratio={verdict['sgd_lr_ratio']:.1f}x  "
        f"--> {'PASS' if tests['T2_sgd_lr_ratio_gt_20']['pass'] else 'FAIL'}"
    )
    print(
        f"  T3: SGD/Muon LR variability ratio > 10?  {verdict['sgd_to_muon_variability_ratio']:.1f}x  "
        f"--> {'PASS' if tests['T3_variability_ratio_gt_10']['pass'] else 'FAIL'}"
    )

    print('\nCalibration:')
    print(
        f"  - Observed Muon slope is {analysis['fits']['muon']['slope']:.3f}; "
        'this is less depth-sensitive than SGD, but not near-flat.'
    )
    print(
        '  - The relevant mechanism diagnostic here is normalized step-1 operator norm, '
        'not raw invariance of ||ΔW||_op.'
    )
    print(
        '  - Results are consistent with a spectral-bound explanation in this toy setting; '
        'they are not a proof or a universal law.'
    )

    print(f"\n{'=' * 100}")
    print('EXPERIMENT COMPLETE')
    print(f"{'=' * 100}")


def run_experiment(
    depths=None,
    sgd_lr_grid=None,
    muon_lr_grid=None,
    seeds=None,
    selection_seed_count=SELECTION_SEED_COUNT,
    num_steps=NUM_STEPS,
    verbose=False,
):
    """Run the toy LR sweep study and return structured results."""
    depths = list(DEPTHS if depths is None else depths)
    sgd_lr_grid = np.asarray(SGD_LR_GRID if sgd_lr_grid is None else sgd_lr_grid, dtype=float)
    muon_lr_grid = np.asarray(MUON_LR_GRID if muon_lr_grid is None else muon_lr_grid, dtype=float)

    if seeds is None:
        seeds = [42 + i * 137 for i in range(NUM_SEEDS)]
    seeds = [int(seed) for seed in seeds]
    selection_seed_count = min(int(selection_seed_count), len(seeds))
    selection_seeds = seeds[:selection_seed_count]

    config = _build_config(depths, sgd_lr_grid.tolist(), muon_lr_grid.tolist(), seeds, selection_seed_count, num_steps)

    if verbose:
        _print_header(config)

    records = []

    for depth in depths:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"  DEPTH L={depth}")
            print(f"{'=' * 80}")

        for optimizer, lr_grid in (('sgd', sgd_lr_grid), ('muon', muon_lr_grid)):
            sweep = []
            best_lr = float(lr_grid[0])
            best_selection_mean_final_loss = float('inf')

            for lr in lr_grid:
                selection_losses = []
                for seed in selection_seeds:
                    X, Y = make_data(seed)
                    weights_init = init_weights(depth, seed + 5000)
                    final_loss, _ = train(weights_init, X, Y, float(lr), optimizer, num_steps=num_steps)
                    selection_losses.append(float(final_loss))

                finite_selection_losses = [loss for loss in selection_losses if np.isfinite(loss)]
                selection_mean_final_loss = (
                    float(np.mean(finite_selection_losses))
                    if finite_selection_losses
                    else float('inf')
                )
                sweep.append(
                    {
                        'lr': float(lr),
                        'selection_seed_losses': selection_losses,
                        'selection_mean_final_loss': float(selection_mean_final_loss),
                        'selection_finite_count': int(len(finite_selection_losses)),
                    }
                )

                if selection_mean_final_loss < best_selection_mean_final_loss:
                    best_selection_mean_final_loss = float(selection_mean_final_loss)
                    best_lr = float(lr)

            final_seed_results = []
            final_seed_losses = []
            step1_max_op_norms = []
            step1_avg_op_norms = []

            for seed in seeds:
                X, Y = make_data(seed)
                weights_init = init_weights(depth, seed + 5000)
                final_loss, step1_op_norms = train(weights_init, X, Y, best_lr, optimizer, num_steps=num_steps)

                max_step1_op_norm = float(max(step1_op_norms)) if step1_op_norms is not None else float('nan')
                avg_step1_op_norm = float(np.mean(step1_op_norms)) if step1_op_norms is not None else float('nan')

                final_seed_results.append(
                    {
                        'seed': int(seed),
                        'final_loss': float(final_loss),
                        'step1_op_norms': [float(x) for x in step1_op_norms] if step1_op_norms is not None else None,
                        'max_step1_op_norm': max_step1_op_norm,
                        'avg_step1_op_norm': avg_step1_op_norm,
                    }
                )
                final_seed_losses.append(float(final_loss))
                if step1_op_norms is not None:
                    step1_max_op_norms.append(max_step1_op_norm)
                    step1_avg_op_norms.append(avg_step1_op_norm)

            finite_final_losses = [loss for loss in final_seed_losses if np.isfinite(loss)]
            mean_final_loss = _safe_mean(finite_final_losses, default=float('inf'))
            std_final_loss = _safe_std(finite_final_losses, default=float('nan'))
            mean_max_step1_op_norm = _safe_mean(step1_max_op_norms)
            std_max_step1_op_norm = _safe_std(step1_max_op_norms)
            mean_avg_step1_op_norm = _safe_mean(step1_avg_op_norms)
            std_avg_step1_op_norm = _safe_std(step1_avg_op_norms)

            record = {
                'depth': int(depth),
                'optimizer': optimizer,
                'lr_grid': [float(x) for x in lr_grid.tolist()],
                'sweep': sweep,
                'best_lr': float(best_lr),
                'best_selection_mean_final_loss': float(best_selection_mean_final_loss),
                'final_seed_results': final_seed_results,
                'final_seed_losses': final_seed_losses,
                'finite_final_loss_count': int(len(finite_final_losses)),
                'mean_final_loss': float(mean_final_loss),
                'std_final_loss': float(std_final_loss),
                'step1_max_op_norms': [float(x) for x in step1_max_op_norms],
                'step1_avg_op_norms': [float(x) for x in step1_avg_op_norms],
                'mean_max_step1_op_norm': float(mean_max_step1_op_norm),
                'std_max_step1_op_norm': float(std_max_step1_op_norm),
                'mean_avg_step1_op_norm': float(mean_avg_step1_op_norm),
                'std_avg_step1_op_norm': float(std_avg_step1_op_norm),
                'mean_max_step1_op_norm_over_lr': float(mean_max_step1_op_norm / best_lr)
                if np.isfinite(mean_max_step1_op_norm)
                else float('nan'),
                'mean_avg_step1_op_norm_over_lr': float(mean_avg_step1_op_norm / best_lr)
                if np.isfinite(mean_avg_step1_op_norm)
                else float('nan'),
            }
            records.append(record)

            if verbose:
                print(
                    f"  {optimizer.upper():>5}: best_grid_lr={record['best_lr']:.6f}  "
                    f"mean_final_loss={record['mean_final_loss']:.6e}  "
                    f"max_||ΔW||_op={record['mean_max_step1_op_norm']:.4e}  "
                    f"(max_||ΔW||_op)/lr={record['mean_max_step1_op_norm_over_lr']:.4e}"
                )

    records = sorted(records, key=lambda record: (record['optimizer'], record['depth']))
    analysis = _build_analysis(depths, records)
    by_optimizer = {
        opt: [record for record in records if record['optimizer'] == opt]
        for opt in ('sgd', 'muon')
    }

    payload = {
        'config': config,
        'results': {
            'table': records,
            'by_optimizer': by_optimizer,
        },
        'analysis': analysis,
    }

    if verbose:
        _print_final_analysis(payload)

    return payload


def main():
    """CLI entrypoint that preserves the standard script behavior."""
    return run_experiment(verbose=True)


if __name__ == '__main__':
    main()
