#!/usr/bin/env python3
"""
H6b: Activation-dependent Muon vs SGD final-loss comparison
===========================================================

Motivation:
  H6 suggested that Muon can retain a sizable advantage over SGD in a deep
  linear toy problem after separate learning-rate tuning. This H6b follow-up
  keeps the same small synthetic setting and asks how the final-loss gap
  changes across activation functions.

Observed metric:
  For each activation in {linear, ReLU, tanh, GELU}:
    For each optimizer in {SGD, Muon}:
      Sweep a fixed LR grid on 3 seeds, training for 500 steps.
      Select the LR with the lowest mean finite final loss.
    Re-evaluate the chosen LR on 10 seeds.
  Report advantage = mean_final_loss_SGD / mean_final_loss_Muon.

Important scope limits:
  - This measures activation-dependent final-loss ratios in a 4-layer 32x32
    Gaussian regression toy problem under separately tuned learning rates.
  - It does NOT directly measure directional alignment, singular-value rescue,
    or causal mechanisms.
  - The linear activation is an internal control inside this protocol; whether
    it reproduces any prior ~7x baseline is an empirical outcome, not an
    assumption.
"""

import time
import numpy as np

EXPERIMENT_ID = 'H6b_RESIDUAL_DIRECTIONAL_ADVANTAGE'
EXPERIMENT_TITLE = 'H6b: Activation-dependent Muon vs SGD final-loss comparison'
EXPERIMENT_SCOPE = (
    'Activation-dependent Muon-vs-SGD final-loss comparison under separately '
    'tuned learning rates in a small synthetic regression task.'
)
SELECTION_RULE = (
    'Choose the LR with the lowest mean finite final loss on the sweep seeds. '
    'Partial divergence is reported but not separately penalized unless every '
    'sweep seed diverges, in which case the candidate mean is inf.'
)

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
BATCH_SIZE = 64
SWEEP_NUM_SEEDS = 3
DIVERGENCE_THRESHOLD = 1e10
WEIGHT_INIT_SEED_OFFSET = 5000
SEED_BASE = 42
SEED_STRIDE = 137

MUON_LRS = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
SGD_LRS = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]

ACTIVATIONS = ['linear', 'relu', 'tanh', 'gelu']
OPTIMIZERS = ['muon', 'sgd']


def get_default_config():
    return {
        'dim': DIM,
        'num_layers': NUM_LAYERS,
        'num_steps': NUM_STEPS,
        'momentum': MOMENTUM,
        'ns_iters': NS_ITERS,
        'num_seeds': NUM_SEEDS,
        'batch_size': BATCH_SIZE,
        'sweep_num_seeds': SWEEP_NUM_SEEDS,
        'divergence_threshold': DIVERGENCE_THRESHOLD,
        'weight_init_seed_offset': WEIGHT_INIT_SEED_OFFSET,
        'seed_base': SEED_BASE,
        'seed_stride': SEED_STRIDE,
        'muon_lrs': list(MUON_LRS),
        'sgd_lrs': list(SGD_LRS),
        'activations': list(ACTIVATIONS),
        'optimizers': list(OPTIMIZERS),
        'selection_rule': SELECTION_RULE,
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


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_deriv(x):
    s = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    t = np.tanh(s)
    ds = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * ds


def apply_act(x, act_name):
    if act_name == 'linear':
        return x
    if act_name == 'relu':
        return np.maximum(0, x)
    if act_name == 'tanh':
        return np.tanh(x)
    if act_name == 'gelu':
        return gelu(x)
    raise ValueError(f'Unknown activation: {act_name}')


def apply_act_deriv(pre, act_name):
    if act_name == 'linear':
        return np.ones_like(pre)
    if act_name == 'relu':
        return (pre > 0).astype(float)
    if act_name == 'tanh':
        return 1 - np.tanh(pre)**2
    if act_name == 'gelu':
        return gelu_deriv(pre)
    raise ValueError(f'Unknown activation: {act_name}')


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]


def forward(weights, X, act):
    pre_acts = []
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre)
        if idx < len(weights) - 1:
            out = apply_act(pre, act)
        else:
            out = pre
    return out, pre_acts


def compute_loss(weights, X, Y, act):
    pred, _ = forward(weights, X, act)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def compute_gradients(weights, X, Y, act):
    L = len(weights)
    N = X.shape[1]
    acts_post = [X.copy()]
    pre_acts = []
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre)
        if idx < L - 1:
            out = apply_act(pre, act)
        else:
            out = pre
        acts_post.append(out)
    delta = (acts_post[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts_post[l].T
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * apply_act_deriv(pre_acts[l - 1], act)
    return grads


def train(weights_init, X, Y, lr, optimizer, act, num_steps=NUM_STEPS,
          divergence_threshold=DIVERGENCE_THRESHOLD):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for _ in range(num_steps):
        loss = compute_loss(weights, X, Y, act)
        if not np.isfinite(loss) or loss > divergence_threshold:
            return float('inf')
        grads = compute_gradients(weights, X, Y, act)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            elif optimizer == 'sgd':
                mom[i] = MOMENTUM * mom[i] + grads[i]
            else:
                raise ValueError(f'Unknown optimizer: {optimizer}')
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y, act)


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3
    return X, Y


def generate_seeds(num_seeds=NUM_SEEDS, seed_base=SEED_BASE, seed_stride=SEED_STRIDE):
    return [seed_base + i * seed_stride for i in range(num_seeds)]


def summarize_losses(seeds, losses):
    seed_records = [
        {'seed': int(seed), 'final_loss': float(loss)}
        for seed, loss in zip(seeds, losses)
    ]
    finite = [float(loss) for loss in losses if np.isfinite(loss)]
    mean_loss = float(np.mean(finite)) if finite else float('inf')
    std_loss = float(np.std(finite)) if finite else float('inf')
    return {
        'seed_records': seed_records,
        'seed_losses': [float(loss) for loss in losses],
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'n_diverged': int(len(losses) - len(finite)),
        'n_finite': int(len(finite)),
        'n_total': int(len(losses)),
    }


def evaluate_candidate_lr(act, optimizer, lr, seeds, num_steps=NUM_STEPS,
                          divergence_threshold=DIVERGENCE_THRESHOLD):
    losses = []
    for seed in seeds:
        X, Y = make_data(seed)
        weights = init_weights(seed + WEIGHT_INIT_SEED_OFFSET)
        final_loss = train(
            weights, X, Y, lr, optimizer, act,
            num_steps=num_steps,
            divergence_threshold=divergence_threshold,
        )
        losses.append(final_loss)
    summary = summarize_losses(seeds, losses)
    summary.update({
        'lr': float(lr),
        'activation': act,
        'optimizer': optimizer,
    })
    return summary


def select_best_lr(sweep_rows, fallback_lr):
    best_lr = float(fallback_lr)
    best_mean_loss = float('inf')
    for row in sweep_rows:
        if row['mean_loss'] < best_mean_loss:
            best_mean_loss = row['mean_loss']
            best_lr = row['lr']
    return best_lr, best_mean_loss


def compute_advantage(muon_loss, sgd_loss):
    if np.isfinite(muon_loss) and np.isfinite(sgd_loss):
        return float(sgd_loss / max(muon_loss, 1e-30))
    return float('nan')


def run_experiment(activations=None, num_seeds=NUM_SEEDS,
                   sweep_num_seeds=SWEEP_NUM_SEEDS, num_steps=NUM_STEPS,
                   muon_lrs=None, sgd_lrs=None, verbose=True):
    config = get_default_config()
    if activations is None:
        activations = list(config['activations'])
    else:
        activations = list(activations)
    if muon_lrs is None:
        muon_lrs = list(config['muon_lrs'])
    else:
        muon_lrs = list(muon_lrs)
    if sgd_lrs is None:
        sgd_lrs = list(config['sgd_lrs'])
    else:
        sgd_lrs = list(sgd_lrs)

    seeds = generate_seeds(num_seeds=num_seeds)
    sweep_seeds = seeds[:sweep_num_seeds]
    started_at = time.time()

    results_by_activation = {}

    if verbose:
        print('=' * 100)
        print(EXPERIMENT_TITLE.upper())
        print('=' * 100)
        print(EXPERIMENT_SCOPE)
        print(f'Activations: {activations}')
        print(
            f'Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {num_steps} steps, '
            f'{num_seeds} seeds ({sweep_num_seeds} sweep seeds)'
        )
        print(f'Selection rule: {SELECTION_RULE}')
        print()

    for act in activations:
        if verbose:
            print(f'\n--- Activation: {act.upper()} ---')

        activation_results = {'optimizers': {}}
        for optimizer, candidates in [('muon', muon_lrs), ('sgd', sgd_lrs)]:
            sweep_rows = [
                evaluate_candidate_lr(
                    act,
                    optimizer,
                    lr,
                    sweep_seeds,
                    num_steps=num_steps,
                    divergence_threshold=config['divergence_threshold'],
                )
                for lr in candidates
            ]
            best_lr, best_sweep_mean = select_best_lr(sweep_rows, candidates[-1])
            for row in sweep_rows:
                row['selected'] = bool(np.isclose(row['lr'], best_lr))

            full_eval = evaluate_candidate_lr(
                act,
                optimizer,
                best_lr,
                seeds,
                num_steps=num_steps,
                divergence_threshold=config['divergence_threshold'],
            )
            full_eval['selected'] = True

            activation_results['optimizers'][optimizer] = {
                'candidate_lrs': [float(lr) for lr in candidates],
                'lr_sweep': sweep_rows,
                'selected_lr': float(best_lr),
                'selected_lr_sweep_mean_loss': float(best_sweep_mean),
                'selection_rule': SELECTION_RULE,
                'full_eval': full_eval,
            }

            if verbose:
                print(
                    f"  {optimizer:>5} best_lr={best_lr:.4f} "
                    f"sweep_mean={best_sweep_mean:.4e} "
                    f"full_mean={full_eval['mean_loss']:.4e} "
                    f"diverged={full_eval['n_diverged']}/{full_eval['n_total']}"
                )

        muon_mean = activation_results['optimizers']['muon']['full_eval']['mean_loss']
        sgd_mean = activation_results['optimizers']['sgd']['full_eval']['mean_loss']
        activation_results['advantage'] = compute_advantage(muon_mean, sgd_mean)
        results_by_activation[act] = activation_results

    runtime_seconds = float(time.time() - started_at)
    advantages = {
        act: float(results_by_activation[act]['advantage'])
        for act in activations
    }
    valid_advs = [value for value in advantages.values() if np.isfinite(value)]
    t1 = all(value > 1.0 for value in valid_advs)
    t2 = all(0.5 < value < 20.0 for value in valid_advs) if valid_advs else False
    t3 = advantages.get('tanh', 0.0) > advantages.get('relu', float('inf'))

    tests = {
        'T1': {
            'description': 'Muon mean final loss is lower than SGD mean final loss for every finite activation result.',
            'passed': bool(t1),
            'criterion': 'advantage > 1 for all finite activation advantages',
            'observed_advantages': advantages,
        },
        'T2': {
            'description': 'All finite activation advantages fall inside the broad consistency band (0.5, 20).',
            'passed': bool(t2),
            'criterion': '0.5 < advantage < 20 for all finite activation advantages',
            'observed_advantages': advantages,
        },
        'T3': {
            'description': 'tanh shows larger advantage than ReLU under this proxy comparison.',
            'passed': bool(t3),
            'criterion': 'advantage(tanh) > advantage(relu)',
            'observed_tanh_advantage': float(advantages.get('tanh', float('nan'))),
            'observed_relu_advantage': float(advantages.get('relu', float('nan'))),
        },
    }

    results = {
        'metadata': {
            'experiment_id': EXPERIMENT_ID,
            'title': EXPERIMENT_TITLE,
            'scope': EXPERIMENT_SCOPE,
            'selection_rule': SELECTION_RULE,
            'notes': [
                'This is a final-loss comparison after separate LR sweeps, not a direct directional or spectral diagnostic.',
                'The linear activation is treated as an internal control within this protocol; any ~7x baseline must be checked empirically.',
            ],
        },
        'config': {
            **config,
            'num_steps': int(num_steps),
            'num_seeds': int(num_seeds),
            'sweep_num_seeds': int(sweep_num_seeds),
            'activations': activations,
            'muon_lrs': muon_lrs,
            'sgd_lrs': sgd_lrs,
        },
        'seeds': [int(seed) for seed in seeds],
        'sweep_seeds': [int(seed) for seed in sweep_seeds],
        'results_by_activation': results_by_activation,
        'advantages': advantages,
        'tests': tests,
        'runtime_seconds': runtime_seconds,
    }

    if verbose:
        print(f"\n\n{'=' * 100}")
        print('RESULTS: Muon vs SGD at separately tuned LR per activation')
        print(f"{'=' * 100}")
        print(
            f"\n  {'Activation':>10}  {'Muon loss':>12} {'(lr)':>8}  {'SGD loss':>12} {'(lr)':>8}  {'Advantage':>12}"
        )
        print('  ' + '-' * 78)
        for act in activations:
            muon_eval = results_by_activation[act]['optimizers']['muon']['full_eval']
            sgd_eval = results_by_activation[act]['optimizers']['sgd']['full_eval']
            print(
                f"  {act:>10}  {muon_eval['mean_loss']:>12.4e} "
                f"{results_by_activation[act]['optimizers']['muon']['selected_lr']:>8.4f}  "
                f"{sgd_eval['mean_loss']:>12.4e} "
                f"{results_by_activation[act]['optimizers']['sgd']['selected_lr']:>8.4f}  "
                f"{results_by_activation[act]['advantage']:>12.1f}x"
            )

        print(f"\n\n{'=' * 100}")
        print('HYPOTHESIS TESTS')
        print(f"{'=' * 100}")
        print(f"\n  T1: Muon beats SGD for all finite activation results? --> {'PASS' if tests['T1']['passed'] else 'FAIL'}")
        print(f"  T2: Advantage lies in the broad 0.5-20x band? --> {'PASS' if tests['T2']['passed'] else 'FAIL'}")
        print(f"  T3: tanh advantage > ReLU advantage? --> {'PASS' if tests['T3']['passed'] else 'FAIL'}")
        print(
            '       '
            f"tanh={tests['T3']['observed_tanh_advantage']:.1f}x, "
            f"ReLU={tests['T3']['observed_relu_advantage']:.1f}x"
        )
        print(f"\nRuntime: {runtime_seconds:.2f}s")
        print(f"\n{'=' * 100}")
        print('EXPERIMENT COMPLETE')
        print(f"{'=' * 100}")

    return results


def main():
    run_experiment(verbose=True)


if __name__ == '__main__':
    main()
