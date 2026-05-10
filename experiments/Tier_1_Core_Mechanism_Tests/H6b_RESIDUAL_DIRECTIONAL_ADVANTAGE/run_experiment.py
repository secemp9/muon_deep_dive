#!/usr/bin/env python3
"""
H6b: Residual 7x Directional Advantage Across Activations
============================================================

MOTIVATION (from H6 surprise):
  After correcting for LR artifacts, Muon retains a genuine 7x advantage
  over SGD in deep linear nets. This was measured at optimal LR for both.
  QUESTION: Does this 7x advantage persist, grow, or shrink across
  activation functions? If the advantage changes dramatically with
  nonlinearity, the mechanism may be geometry-specific, not universal.

PROTOCOL:
  For each activation in {linear, ReLU, tanh, GELU}:
    For each optimizer in {SGD, Muon}:
      Sweep LR (10 candidates each), then train 500 steps.
    Compute advantage = SGD_best_loss / Muon_best_loss.

KEY TESTS:
  T1: Does Muon beat SGD (at optimal LR) for ALL activations?
  T2: Is the advantage consistent (within 0.5-20x across activations)?
  T3: Does tanh (vanishing gradients) show larger Muon advantage than ReLU?
      (Hypothesis: SV equalization helps more when gradients vanish.)

Setup: 4-layer, 32x32, 500 steps, 10 seeds, LR swept per activation.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
BATCH_SIZE = 64

MUON_LRS = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
SGD_LRS = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005]

ACTIVATIONS = ['linear', 'relu', 'tanh', 'gelu']


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
    elif act_name == 'relu':
        return np.maximum(0, x)
    elif act_name == 'tanh':
        return np.tanh(x)
    elif act_name == 'gelu':
        return gelu(x)


def apply_act_deriv(pre, act_name):
    if act_name == 'linear':
        return np.ones_like(pre)
    elif act_name == 'relu':
        return (pre > 0).astype(float)
    elif act_name == 'tanh':
        return 1 - np.tanh(pre)**2
    elif act_name == 'gelu':
        return gelu_deriv(pre)


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


def train(weights_init, X, Y, lr, optimizer, act):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y, act)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y, act)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y, act)


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3
    return X, Y


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H6b: RESIDUAL 7x DIRECTIONAL ADVANTAGE ACROSS ACTIVATIONS")
    print("=" * 100)
    print(f"Activations: {ACTIVATIONS}")
    print(f"Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    results = {}
    for act in ACTIVATIONS:
        print(f"\n--- Activation: {act.upper()} ---")

        # LR sweep (3 seeds, 200 steps)
        best_lrs = {}
        for opt, candidates in [('muon', MUON_LRS), ('sgd', SGD_LRS)]:
            best_lr, best_loss = candidates[-1], float('inf')
            for lr in candidates:
                losses = []
                for s in seeds[:3]:
                    X, Y = make_data(s)
                    w = init_weights(s + 5000)
                    fl = train(w, X, Y, lr, opt, act)
                    losses.append(fl)
                ml = np.mean([l for l in losses if np.isfinite(l)]) if any(np.isfinite(l) for l in losses) else float('inf')
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr
            best_lrs[opt] = best_lr
            print(f"  {opt:>5} best_lr={best_lr:.4f}")

        # Full training
        for opt in ['muon', 'sgd']:
            losses = []
            for s in seeds:
                X, Y = make_data(s)
                w = init_weights(s + 5000)
                fl = train(w, X, Y, best_lrs[opt], opt, act)
                losses.append(fl)
            finite = [l for l in losses if np.isfinite(l)]
            mean_l = np.mean(finite) if finite else float('inf')
            results[(act, opt)] = {'lr': best_lrs[opt], 'mean_loss': mean_l}

    # Results table
    print(f"\n\n{'=' * 100}")
    print("RESULTS: Muon vs SGD at Optimal LR per Activation")
    print(f"{'=' * 100}")

    print(f"\n  {'Activation':>10}  {'Muon loss':>12} {'(lr)':>8}  {'SGD loss':>12} {'(lr)':>8}  {'Advantage':>12}")
    print("  " + "-" * 70)

    advantages = {}
    for act in ACTIVATIONS:
        ml = results[(act, 'muon')]['mean_loss']
        sl = results[(act, 'sgd')]['mean_loss']
        adv = sl / max(ml, 1e-30) if np.isfinite(sl) and np.isfinite(ml) else float('nan')
        advantages[act] = adv
        print(f"  {act:>10}  {ml:>12.4e} {results[(act,'muon')]['lr']:>8.4f}  "
              f"{sl:>12.4e} {results[(act,'sgd')]['lr']:>8.4f}  {adv:>12.1f}x")

    # Hypothesis tests
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    valid_advs = [v for v in advantages.values() if np.isfinite(v)]
    t1 = all(v > 1.0 for v in valid_advs)
    t2 = all(0.5 < v < 20.0 for v in valid_advs) if valid_advs else False
    t3 = advantages.get('tanh', 0) > advantages.get('relu', float('inf'))

    print(f"\n  T1: Muon beats SGD for ALL activations? --> {'PASS' if t1 else 'FAIL'}")
    print(f"  T2: Advantage consistent (0.5-20x range)? --> {'PASS' if t2 else 'FAIL'}")
    print(f"  T3: tanh advantage > ReLU advantage? --> {'PASS' if t3 else 'FAIL'}")
    print(f"       tanh={advantages.get('tanh', 'N/A'):.1f}x, ReLU={advantages.get('relu', 'N/A'):.1f}x")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
