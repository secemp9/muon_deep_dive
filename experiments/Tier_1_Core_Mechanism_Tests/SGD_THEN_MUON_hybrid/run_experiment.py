#!/usr/bin/env python3
"""
Experiment 3.13: SGD->Muon Hybrid Fine-tuning
=============================================

Toy deep-linear switch-point sweep for SGD->Muon fine-tuning after a target
perturbation.

Measured scope only:
  - final fine-tuning loss
  - steps to reach 50% of the initial fine-tuning loss
  - comparison of pure Muon, pure SGD, and intermediate SGD->Muon switch points

The "shift" here is a toy protocol: randomly replacing 20% of the entries in
an otherwise fixed target matrix. That is useful for a controlled perturbation
study in this deep-linear setting, but it is not a realistic distribution shift
and it does not directly prove a mechanistic spectral/gauge phase transition.
"""

import numpy as np


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_SEED = 42
DEFAULT_DIM = 32
DEFAULT_NUM_LAYERS = 4
DEFAULT_BATCH_SIZE = 64

# Pre-training
DEFAULT_PRETRAIN_STEPS = 500
DEFAULT_PRETRAIN_LR = 0.01

# Fine-tuning
DEFAULT_FINETUNE_STEPS = 200
DEFAULT_SGD_FT_LR = 0.01
DEFAULT_MUON_FT_LR = 0.005

# Muon parameters
DEFAULT_MOMENTUM = 0.9
DEFAULT_NS_ITERS = 5

# Target modification
DEFAULT_MODIFY_FRAC = 0.20

# Sweep
DEFAULT_SWITCH_POINTS = [0, 5, 10, 20, 50, 100, 200]
DEFAULT_NUM_SEEDS = 5
DEFAULT_SEED_STRIDE = 137
DEFAULT_THRESHOLD_FRAC = 0.5
DEFAULT_SNAPSHOT_STEPS = [0, 5, 10, 20, 50, 100, 150, 200]


# =============================================================================
# CONFIG HELPERS
# =============================================================================


def get_default_config():
    return {
        'seed': DEFAULT_SEED,
        'dim': DEFAULT_DIM,
        'num_layers': DEFAULT_NUM_LAYERS,
        'batch_size': DEFAULT_BATCH_SIZE,
        'pretrain_steps': DEFAULT_PRETRAIN_STEPS,
        'pretrain_lr': DEFAULT_PRETRAIN_LR,
        'finetune_steps': DEFAULT_FINETUNE_STEPS,
        'sgd_ft_lr': DEFAULT_SGD_FT_LR,
        'muon_ft_lr': DEFAULT_MUON_FT_LR,
        'momentum': DEFAULT_MOMENTUM,
        'ns_iters': DEFAULT_NS_ITERS,
        'modify_frac': DEFAULT_MODIFY_FRAC,
        'switch_points': list(DEFAULT_SWITCH_POINTS),
        'num_seeds': DEFAULT_NUM_SEEDS,
        'seed_stride': DEFAULT_SEED_STRIDE,
        'threshold_frac': DEFAULT_THRESHOLD_FRAC,
        'snapshot_steps': list(DEFAULT_SNAPSHOT_STEPS),
    }



def normalize_config(config=None):
    merged = get_default_config()
    if config is not None:
        merged.update(config)

    merged['switch_points'] = sorted(dict.fromkeys(int(s) for s in merged['switch_points']))
    merged['snapshot_steps'] = [int(s) for s in merged['snapshot_steps']]

    if merged['num_seeds'] < 1:
        raise ValueError('num_seeds must be at least 1.')
    if merged['finetune_steps'] < 1:
        raise ValueError('finetune_steps must be at least 1.')
    if 0 not in merged['switch_points']:
        raise ValueError('switch_points must include 0 for the pure Muon baseline.')
    if merged['finetune_steps'] not in merged['switch_points']:
        raise ValueError(
            'switch_points must include finetune_steps for the pure SGD baseline.'
        )
    if any(s < 0 or s > merged['finetune_steps'] for s in merged['switch_points']):
        raise ValueError('Each switch point must satisfy 0 <= S <= finetune_steps.')

    return merged



def build_fixed_data(dim, batch_size, seed):
    """Reproduce the original fixed-data setup without global RNG side effects."""
    rng = np.random.RandomState(seed)
    X_data = rng.randn(dim, batch_size) * 0.3
    W_target_original = rng.randn(dim, dim) * 0.5
    return X_data, W_target_original



def switch_description(switch_step, finetune_steps):
    if switch_step == 0:
        return 'Pure Muon'
    if switch_step == finetune_steps:
        return 'Pure SGD'
    return f'SGD({switch_step})->Muon({finetune_steps - switch_step})'


# =============================================================================
# NETWORK HELPERS
# =============================================================================


def init_weights(num_layers, dim, rng):
    weights = []
    for _ in range(num_layers):
        W = np.eye(dim) + rng.randn(dim, dim) * 0.1
        weights.append(W.copy())
    return weights



def copy_weights(weights):
    return [W.copy() for W in weights]



def forward_linear(weights, X):
    out = X.copy()
    for W in weights:
        out = W @ out
    return out



def compute_loss(weights, X, Y_target):
    Y_pred = forward_linear(weights, X)
    diff = Y_pred - Y_target
    return 0.5 * np.mean(np.sum(diff ** 2, axis=0))



def compute_gradients(weights, X, Y_target):
    num_layers = len(weights)
    N = X.shape[1]
    activations = [X.copy()]
    for W in weights:
        activations.append(W @ activations[-1])
    delta = (activations[-1] - Y_target) / N
    grads = [None] * num_layers
    for l in range(num_layers - 1, -1, -1):
        grads[l] = delta @ activations[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return grads



def make_modified_target(W_original, frac, rng):
    W_mod = W_original.copy()
    n_entries = W_mod.size
    n_change = int(frac * n_entries)
    indices = rng.choice(n_entries, size=n_change, replace=False)
    flat = W_mod.ravel()
    flat[indices] = rng.randn(n_change) * 0.5
    return W_mod


# =============================================================================
# NEWTON-SCHULZ ORTHOGONALIZATION
# =============================================================================


def newton_schulz_ortho(M, n_iters):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M / (np.linalg.norm(M, ord='fro') + 1e-7)
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    Id = np.eye(X.shape[0])
    for _ in range(n_iters):
        A = X @ X.T
        X = (a * Id + b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X


# =============================================================================
# TRAINING LOOPS
# =============================================================================


def pretrain_sgd(weights_init, W_target, X, n_steps, lr, momentum):
    weights = copy_weights(weights_init)
    Y_target = W_target @ X
    velocities = [np.zeros_like(w) for w in weights]

    for _ in range(n_steps):
        grads = compute_gradients(weights, X, Y_target)
        for i in range(len(weights)):
            velocities[i] = momentum * velocities[i] + grads[i]
            weights[i] = weights[i] - lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    return weights, float(final_loss)



def train_hybrid(
    weights_init,
    W_target,
    X,
    n_steps,
    switch_step,
    sgd_ft_lr,
    muon_ft_lr,
    momentum,
    ns_iters,
):
    """
    Train for n_steps total:
      - Steps 0..switch_step-1: SGD with momentum
      - Steps switch_step..n_steps-1: Muon with momentum

    Returns a loss trajectory of length n_steps + 1 where index 0 is the initial
    fine-tuning loss and the last entry is the post-training final loss.
    """
    weights = copy_weights(weights_init)
    Y_target = W_target @ X
    velocities = [np.zeros_like(w) for w in weights]

    losses = []

    for step in range(n_steps):
        loss = compute_loss(weights, X, Y_target)
        losses.append(loss)

        if np.isnan(loss) or loss > 1e10:
            losses.extend([loss] * (n_steps - step - 1))
            break

        grads = compute_gradients(weights, X, Y_target)

        if step < switch_step:
            for i in range(len(weights)):
                velocities[i] = momentum * velocities[i] + grads[i]
                weights[i] = weights[i] - sgd_ft_lr * velocities[i]
        else:
            if step == switch_step and switch_step > 0:
                velocities = [np.zeros_like(w) for w in weights]

            for i in range(len(weights)):
                ortho_grad = newton_schulz_ortho(grads[i], n_iters=ns_iters)
                velocities[i] = momentum * velocities[i] + ortho_grad
                weights[i] = weights[i] - muon_ft_lr * velocities[i]

    final_loss = compute_loss(weights, X, Y_target)
    losses.append(final_loss)

    return np.array(losses, dtype=float)



def steps_to_threshold(losses, frac=0.5):
    """Return the first index where losses reach frac * initial_loss."""
    if len(losses) == 0:
        return len(losses)
    threshold = losses[0] * frac
    for i, l in enumerate(losses):
        if l <= threshold:
            return i
    return len(losses)


# =============================================================================
# SUMMARIZATION
# =============================================================================


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    sem = float(std / np.sqrt(len(values))) if len(values) > 0 else float('nan')
    ci95 = float(1.96 * sem) if len(values) > 0 else float('nan')
    return mean, std, sem, ci95



def build_results(config, X_data, W_target_original, seeds, seed_runs, all_results):
    finetune_steps = config['finetune_steps']
    switch_points = config['switch_points']

    per_switch = {}
    summary_table = []
    mean_finals = {}
    mean_half_steps = {}

    for S in switch_points:
        final_losses = np.asarray(all_results[S]['final_losses'], dtype=float)
        half_steps = np.asarray(all_results[S]['half_steps'], dtype=float)
        loss_curves = np.asarray(all_results[S]['loss_curves'], dtype=float)

        mean_final, std_final, sem_final, ci95_final = summarize_values(final_losses)
        mean_half, std_half, sem_half, ci95_half = summarize_values(half_steps)
        mean_curve = np.mean(loss_curves, axis=0)
        std_curve = np.std(loss_curves, axis=0)
        sem_curve = std_curve / np.sqrt(loss_curves.shape[0])
        ci95_curve = 1.96 * sem_curve

        mean_finals[S] = mean_final
        mean_half_steps[S] = mean_half

        row = {
            'S': int(S),
            'description': switch_description(S, finetune_steps),
            'mean_final_loss': mean_final,
            'std_final_loss': std_final,
            'sem_final_loss': sem_final,
            'ci95_final_loss': ci95_final,
            'mean_half_steps': mean_half,
            'std_half_steps': std_half,
            'sem_half_steps': sem_half,
            'ci95_half_steps': ci95_half,
        }
        summary_table.append(row)

        per_switch[S] = {
            'S': int(S),
            'description': row['description'],
            'final_losses': final_losses,
            'half_steps': half_steps,
            'loss_curves': loss_curves,
            'mean_final_loss': mean_final,
            'std_final_loss': std_final,
            'sem_final_loss': sem_final,
            'ci95_final_loss': ci95_final,
            'mean_half_steps': mean_half,
            'std_half_steps': std_half,
            'sem_half_steps': sem_half,
            'ci95_half_steps': ci95_half,
            'mean_loss_curve': mean_curve,
            'std_loss_curve': std_curve,
            'sem_loss_curve': sem_curve,
            'ci95_loss_curve': ci95_curve,
        }

    best_overall_S = min(switch_points, key=lambda s: mean_finals[s])
    nonpure_switch_points = [S for S in switch_points if 0 < S < finetune_steps]
    best_nonpure_hybrid_S = (
        min(nonpure_switch_points, key=lambda s: mean_finals[s])
        if nonpure_switch_points
        else None
    )
    fastest_half_step_S = min(switch_points, key=lambda s: mean_half_steps[s])

    pure_muon_loss = mean_finals[0]
    pure_sgd_loss = mean_finals[finetune_steps]
    pure_muon_half = mean_half_steps[0]
    pure_sgd_half = mean_half_steps[finetune_steps]

    if best_nonpure_hybrid_S is not None:
        best_nonpure_loss = mean_finals[best_nonpure_hybrid_S]
        best_nonpure_half = mean_half_steps[best_nonpure_hybrid_S]
        ratio_best_nonpure_to_muon = best_nonpure_loss / (pure_muon_loss + 1e-12)
    else:
        best_nonpure_loss = None
        best_nonpure_half = None
        ratio_best_nonpure_to_muon = None

    tests = {
        'T1': {
            'question': 'Pure SGD reaches 50% loss faster than pure Muon?',
            'passed': bool(pure_sgd_half < pure_muon_half),
            'details': (
                f"Pure SGD mean half-steps = {pure_sgd_half:.1f}; "
                f"Pure Muon mean half-steps = {pure_muon_half:.1f}"
            ),
        },
        'T2': {
            'question': 'Pure Muon achieves better mean final loss than pure SGD?',
            'passed': bool(pure_muon_loss < pure_sgd_loss),
            'details': (
                f"Pure Muon mean final loss = {pure_muon_loss:.6f}; "
                f"Pure SGD mean final loss = {pure_sgd_loss:.6f}"
            ),
        },
        'T3': {
            'question': 'Best non-pure hybrid is within 5% of pure Muon final loss?',
            'passed': bool(
                best_nonpure_hybrid_S is not None
                and best_nonpure_loss <= pure_muon_loss * 1.05
            ),
            'details': (
                'No non-pure hybrid switch points were supplied.'
                if best_nonpure_hybrid_S is None
                else (
                    f"Best non-pure hybrid S={best_nonpure_hybrid_S} has mean final loss "
                    f"{best_nonpure_loss:.6f}; ratio to pure Muon = "
                    f"{ratio_best_nonpure_to_muon:.4f}"
                )
            ),
        },
        'T4': {
            'question': 'Best non-pure hybrid reaches 50% loss faster than pure Muon?',
            'passed': bool(
                best_nonpure_hybrid_S is not None
                and best_nonpure_half < pure_muon_half
            ),
            'details': (
                'No non-pure hybrid switch points were supplied.'
                if best_nonpure_hybrid_S is None
                else (
                    f"Best non-pure hybrid S={best_nonpure_hybrid_S} mean half-steps = "
                    f"{best_nonpure_half:.1f}; Pure Muon mean half-steps = {pure_muon_half:.1f}"
                )
            ),
        },
    }
    tests['total_passed'] = sum(int(t['passed']) for t in tests.values() if isinstance(t, dict))
    tests['caveat'] = (
        'These are heuristic checks on sample means across seeds, not formal '
        'statistical hypothesis tests.'
    )

    ranking_by_final_loss = []
    sorted_switch_points = sorted(switch_points, key=lambda s: mean_finals[s])
    for rank, S in enumerate(sorted_switch_points, 1):
        ranking_by_final_loss.append({
            'rank': rank,
            'S': int(S),
            'description': switch_description(S, finetune_steps),
            'mean_final_loss': mean_finals[S],
            'mean_half_steps': mean_half_steps[S],
            'is_best_overall': bool(S == best_overall_S),
            'is_best_nonpure_hybrid': bool(
                best_nonpure_hybrid_S is not None and S == best_nonpure_hybrid_S
            ),
        })

    snapshot_steps = [
        step for step in config['snapshot_steps']
        if 0 <= step < len(per_switch[switch_points[0]]['mean_loss_curve'])
    ]
    snapshot_table = []
    for step in snapshot_steps:
        row = {'step': int(step)}
        for S in switch_points:
            row[S] = float(per_switch[S]['mean_loss_curve'][step])
        snapshot_table.append(row)

    target_singular_values = np.linalg.svd(W_target_original, compute_uv=False)
    fixed_data = {
        'X_data': X_data,
        'W_target_original': W_target_original,
        'X_shape': tuple(X_data.shape),
        'target_shape': tuple(W_target_original.shape),
        'X_mean': float(np.mean(X_data)),
        'X_std': float(np.std(X_data)),
        'target_mean': float(np.mean(W_target_original)),
        'target_std': float(np.std(W_target_original)),
        'target_fro_norm': float(np.linalg.norm(W_target_original, ord='fro')),
        'target_singular_values': target_singular_values,
    }

    if best_nonpure_hybrid_S is None:
        verdict = (
            'Only pure baselines were supplied, so no non-pure hybrid conclusion '
            'can be drawn.'
        )
    elif best_overall_S == 0:
        verdict = (
            f"Pure Muon (S=0) is best overall by mean final loss. The best non-pure "
            f"hybrid is S={best_nonpure_hybrid_S}, which remains close in final loss "
            f"while reaching the 50%-loss threshold in {best_nonpure_half:.1f} steps "
            f"versus {pure_muon_half:.1f} for pure Muon under this toy target-"
            f"perturbation protocol."
        )
    else:
        verdict = (
            f"Best overall by mean final loss is S={best_overall_S}; the best non-pure "
            f"hybrid is S={best_nonpure_hybrid_S}. Interpret this as loss-based "
            f"evidence in a toy deep-linear setting, not a mechanistic proof."
        )

    return {
        'config': config,
        'fixed_data': fixed_data,
        'seeds': seeds,
        'seed_runs': seed_runs,
        'per_switch': per_switch,
        'summary_table': summary_table,
        'snapshot_steps': snapshot_steps,
        'snapshot_table': snapshot_table,
        'mean_finals': mean_finals,
        'mean_half_steps': mean_half_steps,
        'best_overall_S': int(best_overall_S),
        'best_overall_description': switch_description(best_overall_S, finetune_steps),
        'best_nonpure_hybrid_S': (
            int(best_nonpure_hybrid_S) if best_nonpure_hybrid_S is not None else None
        ),
        'best_nonpure_hybrid_description': (
            switch_description(best_nonpure_hybrid_S, finetune_steps)
            if best_nonpure_hybrid_S is not None
            else None
        ),
        'fastest_half_step_S': int(fastest_half_step_S),
        'fastest_half_step_description': switch_description(
            fastest_half_step_S, finetune_steps
        ),
        'ranking_by_final_loss': ranking_by_final_loss,
        'tests': tests,
        'verdict': verdict,
    }


# =============================================================================
# MAIN EXPERIMENT API
# =============================================================================


def print_experiment_header(config):
    print('=' * 85)
    print('Experiment 3.13: SGD->Muon Hybrid Fine-tuning')
    print('=' * 85)
    print('Toy scope: deep-linear switch-point sweep under a target perturbation.')
    print('Measured outputs: loss curves, final loss, and steps-to-50%-loss.')
    print(
        'Toy shift caveat: 20% random target-entry replacement is a controlled '
        'perturbation, not a realistic dataset shift.'
    )
    print(f"Setup: {config['num_layers']}-layer deep linear ({config['dim']}x{config['dim']})")
    print(
        f"Pre-train: {config['pretrain_steps']} steps SGD, then modify "
        f"{int(config['modify_frac'] * 100)}% of target"
    )
    print(
        f"Fine-tune: {config['finetune_steps']} steps, "
        f"SGD lr={config['sgd_ft_lr']}, Muon lr={config['muon_ft_lr']}"
    )
    print(f"Switch points S: {config['switch_points']}")
    print(f"Seeds: {config['num_seeds']} (base seed {config['seed']})")
    print('=' * 85)



def run_experiment(config=None, verbose=False):
    config = normalize_config(config)

    X_data, W_target_original = build_fixed_data(
        dim=config['dim'],
        batch_size=config['batch_size'],
        seed=config['seed'],
    )

    if verbose:
        print_experiment_header(config)

    all_results = {
        S: {'final_losses': [], 'half_steps': [], 'loss_curves': []}
        for S in config['switch_points']
    }
    seeds = []
    seed_runs = []

    for seed_idx in range(config['num_seeds']):
        run_seed = config['seed'] + seed_idx * config['seed_stride']
        rng = np.random.RandomState(run_seed)
        seeds.append(int(run_seed))

        if verbose:
            print(f"\n--- Seed {run_seed} ---")

        weights_init = init_weights(config['num_layers'], config['dim'], rng)
        checkpoint, pretrain_final_loss = pretrain_sgd(
            weights_init=weights_init,
            W_target=W_target_original,
            X=X_data,
            n_steps=config['pretrain_steps'],
            lr=config['pretrain_lr'],
            momentum=config['momentum'],
        )

        if verbose:
            print(f"  Pre-train final loss: {pretrain_final_loss:.6f}")

        W_target_modified = make_modified_target(
            W_target_original,
            config['modify_frac'],
            rng,
        )

        seed_record = {
            'seed': int(run_seed),
            'pretrain_final_loss': float(pretrain_final_loss),
            'final_losses': {},
            'half_steps': {},
        }

        for S in config['switch_points']:
            loss_curve = train_hybrid(
                checkpoint,
                W_target_modified,
                X_data,
                config['finetune_steps'],
                S,
                config['sgd_ft_lr'],
                config['muon_ft_lr'],
                config['momentum'],
                config['ns_iters'],
            )
            final_loss = float(loss_curve[-1])
            half_step = int(steps_to_threshold(loss_curve, frac=config['threshold_frac']))

            all_results[S]['final_losses'].append(final_loss)
            all_results[S]['half_steps'].append(half_step)
            all_results[S]['loss_curves'].append(loss_curve)

            seed_record['final_losses'][S] = final_loss
            seed_record['half_steps'][S] = half_step

        seed_runs.append(seed_record)

        if verbose:
            final_str = '  Finals: ' + ', '.join(
                [f"S={S}:{seed_record['final_losses'][S]:.6f}" for S in config['switch_points']]
            )
            print(final_str)

    return build_results(config, X_data, W_target_original, seeds, seed_runs, all_results)


# =============================================================================
# REPORTING
# =============================================================================


def print_report(results):
    config = results['config']
    switch_points = config['switch_points']
    finetune_steps = config['finetune_steps']
    per_switch = results['per_switch']

    print(f"\n\n{'=' * 85}")
    print('AGGREGATE RESULTS')
    print(f"{'=' * 85}")

    print(
        f"\n{'Switch S':<12} {'Description':<25} {'Mean Final Loss':>16} "
        f"{'Std':>10} {'Mean Half-Steps':>16} {'Std':>10}"
    )
    print('-' * 96)

    for row in results['summary_table']:
        print(
            f"{row['S']:<12} {row['description']:<25} "
            f"{row['mean_final_loss']:>16.6f} {row['std_final_loss']:>10.6f} "
            f"{row['mean_half_steps']:>16.1f} {row['std_half_steps']:>10.1f}"
        )

    print(f"\n\n{'=' * 85}")
    print('LOSS CURVES (mean over seeds at selected trajectory indices)')
    print(f"{'=' * 85}")
    print(
        'Indexing note: index 0 is the initial fine-tuning loss; index '
        f"{finetune_steps} is the post-training final loss."
    )

    header_parts = [f"{'Step':>6}"]
    for S in switch_points:
        if S == 0:
            label = 'Muon'
        elif S == finetune_steps:
            label = 'SGD'
        else:
            label = f'S={S}'
        header_parts.append(f"{label:>12}")
    print('  '.join(header_parts))
    print('-' * (8 + 14 * len(switch_points)))

    for snapshot_row in results['snapshot_table']:
        parts = [f"{snapshot_row['step']:>6}"]
        for S in switch_points:
            parts.append(f"{snapshot_row[S]:>12.6f}")
        print('  '.join(parts))

    print(f"\n\n{'=' * 85}")
    print('SWITCH-POINT ANALYSIS')
    print(f"{'=' * 85}")

    best_overall_S = results['best_overall_S']
    best_nonpure_S = results['best_nonpure_hybrid_S']

    print(
        f"\n  Best overall by mean final loss: S={best_overall_S} "
        f"({results['best_overall_description']}) -> "
        f"{per_switch[best_overall_S]['mean_final_loss']:.6f}"
    )
    if best_nonpure_S is not None:
        print(
            f"  Best non-pure hybrid by mean final loss: S={best_nonpure_S} "
            f"({results['best_nonpure_hybrid_description']}) -> "
            f"{per_switch[best_nonpure_S]['mean_final_loss']:.6f}"
        )
    print(
        f"  Pure Muon (S=0) mean final loss: {per_switch[0]['mean_final_loss']:.6f}"
    )
    print(
        f"  Pure SGD (S={finetune_steps}) mean final loss: "
        f"{per_switch[finetune_steps]['mean_final_loss']:.6f}"
    )
    print(
        f"  Fastest to 50% loss: S={results['fastest_half_step_S']} "
        f"({results['fastest_half_step_description']}) -> "
        f"{per_switch[results['fastest_half_step_S']]['mean_half_steps']:.1f} steps"
    )

    print('\n  Ranking by mean final loss:')
    for row in results['ranking_by_final_loss']:
        markers = []
        if row['is_best_overall']:
            markers.append('BEST OVERALL')
        if row['is_best_nonpure_hybrid']:
            markers.append('BEST NON-PURE')
        marker_text = f" <-- {', '.join(markers)}" if markers else ''
        print(
            f"    #{row['rank']}: {row['description']:<25} "
            f"loss={row['mean_final_loss']:.6f}  "
            f"half-steps={row['mean_half_steps']:.1f}{marker_text}"
        )

    print(f"\n\n{'=' * 85}")
    print('HEURISTIC TESTS')
    print(f"{'=' * 85}")
    print(f"Caveat: {results['tests']['caveat']}")

    for test_name in ['T1', 'T2', 'T3', 'T4']:
        test = results['tests'][test_name]
        print(f"\n{test_name}: {test['question']}")
        print(f"    {test['details']}")
        print(f"    --> {'PASS' if test['passed'] else 'FAIL'}")

    print(f"\n\n{'=' * 85}")
    print('FINAL CONCLUSION')
    print(f"{'=' * 85}")
    print(f"Tests passed: {results['tests']['total_passed']}/4")
    print(
        f"Best overall by mean final loss: S={best_overall_S} "
        f"({results['best_overall_description']})"
    )
    if best_nonpure_S is not None:
        print(
            f"Best non-pure hybrid by mean final loss: S={best_nonpure_S} "
            f"({results['best_nonpure_hybrid_description']})"
        )
    print(f"Conclusion: {results['verdict']}")
    print('=' * 85)



def main():
    results = run_experiment(verbose=True)
    print_report(results)
    return results


if __name__ == '__main__':
    main()
