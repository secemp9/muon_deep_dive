#!/usr/bin/env python3
"""
H17: Direction-vs-Conditioning Split Predicts Architecture Benefit
====================================================================

MOTIVATION (from H15 surprise):
  Muon's gauge effect operates through DIRECTION QUALITY, not conditioning.
  In hybrid nets, matrix layers under Muon have WORSE kappa (0.8x) but 3x
  better loss. The conditioning story (330x kappa reduction) is architecture-
  specific.

QUESTION: Can we predict WHICH architectures benefit most from Muon based
  on a simple diagnostic? The hypothesis: architectures where gradient
  ANISOTROPY is high (large sigma_1/sigma_min of gradients) benefit more,
  because Muon's SV equalization has more room to help.

PROTOCOL:
  For each architecture in {deep_linear, relu_net, tanh_net, bottleneck_net}:
    1. Measure gradient anisotropy at init (mean sigma_1/sigma_min of G).
    2. Train with SGD and Muon (optimal LR each).
    3. Compute Muon advantage = SGD_loss / Muon_loss.
  Correlate anisotropy with advantage across architectures.

  Also compute: "direction improvement" = cos(Muon_update, gradient) / cos(SGD_update, gradient).
  This measures how much Muon changes the update direction from raw gradient.

KEY TESTS:
  T1: Gradient anisotropy positively correlates with Muon advantage (r > 0.5).
  T2: Architectures with highest anisotropy have highest Muon advantage.
  T3: The correlation is predictive (not just post-hoc).

Setup: 4-layer, 32x32, 500 steps, 5 seeds per architecture.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
NUM_LAYERS = 4
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

LR_MUON = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003]
LR_SGD = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003]


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(arch, seed):
    rng = np.random.RandomState(seed)
    if arch == 'deep_linear':
        return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]
    elif arch == 'relu_net':
        return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]
    elif arch == 'tanh_net':
        return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(NUM_LAYERS)]
    elif arch == 'bottleneck':
        # Bottleneck: layers 1,3 are wide->narrow, 2,4 are narrow->wide
        # Using rectangular-ish SVD structure
        weights = []
        for l in range(NUM_LAYERS):
            W = rng.randn(DIM, DIM) * 0.1
            # Make half the SVs very small (simulated bottleneck)
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            s[DIM//2:] *= 0.01  # suppress bottom half
            W = U @ np.diag(s) @ Vt
            W += np.eye(DIM) * 0.5
            weights.append(W)
        return weights


def apply_act(x, arch, layer_idx, num_layers):
    if arch == 'deep_linear' or layer_idx == num_layers - 1:
        return x
    elif arch == 'relu_net':
        return np.maximum(0, x)
    elif arch == 'tanh_net':
        return np.tanh(x)
    elif arch == 'bottleneck':
        return np.maximum(0, x)


def act_deriv(pre, arch, layer_idx, num_layers):
    if arch == 'deep_linear' or layer_idx == num_layers - 1:
        return np.ones_like(pre)
    elif arch == 'relu_net' or arch == 'bottleneck':
        return (pre > 0).astype(float)
    elif arch == 'tanh_net':
        return 1 - np.tanh(pre)**2


def forward(weights, X, arch):
    out = X.copy()
    for idx, W in enumerate(weights):
        out = W @ out
        out = apply_act(out, arch, idx, len(weights))
    return out


def compute_loss(weights, X, Y, arch):
    pred = forward(weights, X, arch)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def compute_gradients(weights, X, Y, arch):
    L = len(weights)
    N = X.shape[1]
    acts_post = [X.copy()]
    pre_acts = []
    out = X.copy()
    for idx, W in enumerate(weights):
        pre = W @ out
        pre_acts.append(pre)
        out = apply_act(pre, arch, idx, L)
        acts_post.append(out)
    delta = (acts_post[-1] - Y) / N
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts_post[l].T
        if l > 0:
            delta = weights[l].T @ delta
            delta = delta * act_deriv(pre_acts[l-1], arch, l-1, L)
    return grads


def gradient_anisotropy(weights, X, Y, arch):
    """Mean sigma_1 / sigma_min of gradient across layers."""
    grads = compute_gradients(weights, X, Y, arch)
    anisotropies = []
    for G in grads:
        s = np.linalg.svd(G, compute_uv=False)
        if s[-1] > 1e-12:
            anisotropies.append(s[0] / s[-1])
        else:
            anisotropies.append(s[0] / 1e-12)
    return np.mean(anisotropies)


def train(weights_init, X, Y, lr, optimizer, arch):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y, arch)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y, arch)
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return compute_loss(weights, X, Y, arch)


def make_data(seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3
    return X, Y


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]
    archs = ['deep_linear', 'relu_net', 'tanh_net', 'bottleneck']

    print("=" * 100)
    print("H17: DIRECTION vs CONDITIONING SPLIT -- Architecture Benefit Predictor")
    print("=" * 100)
    print(f"Architectures: {archs}")
    print(f"Network: {NUM_LAYERS}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    results = {}
    for arch in archs:
        print(f"\n--- {arch.upper()} ---")

        # Measure gradient anisotropy at init
        anisotropies = []
        for s in seeds:
            X, Y = make_data(s)
            w = init_weights(arch, s + 5000)
            aniso = gradient_anisotropy(w, X, Y, arch)
            anisotropies.append(aniso)
        mean_aniso = np.mean(anisotropies)
        print(f"  Gradient anisotropy: {mean_aniso:.2f}")

        # LR sweep
        best = {}
        for opt, candidates in [('muon', LR_MUON), ('sgd', LR_SGD)]:
            best_lr, best_loss = candidates[-1], float('inf')
            for lr in candidates:
                losses = []
                for s in seeds[:3]:
                    X, Y = make_data(s)
                    w = init_weights(arch, s + 5000)
                    fl = train(w, X, Y, lr, opt, arch)
                    losses.append(fl)
                ml = np.mean([l for l in losses if np.isfinite(l)]) if any(np.isfinite(l) for l in losses) else float('inf')
                if ml < best_loss:
                    best_loss = ml
                    best_lr = lr
            best[opt] = best_lr

        # Full training
        for opt in ['muon', 'sgd']:
            losses = []
            for s in seeds:
                X, Y = make_data(s)
                w = init_weights(arch, s + 5000)
                fl = train(w, X, Y, best[opt], opt, arch)
                losses.append(fl)
            finite = [l for l in losses if np.isfinite(l)]
            results[(arch, opt)] = np.mean(finite) if finite else float('inf')

        advantage = results[(arch, 'sgd')] / max(results[(arch, 'muon')], 1e-30)
        results[(arch, 'anisotropy')] = mean_aniso
        results[(arch, 'advantage')] = advantage
        print(f"  Muon advantage: {advantage:.1f}x")

    # Correlation analysis
    print(f"\n\n{'=' * 100}")
    print("CORRELATION: Gradient Anisotropy vs Muon Advantage")
    print(f"{'=' * 100}")

    print(f"\n  {'Architecture':>15}  {'Anisotropy':>12}  {'Advantage':>12}")
    print("  " + "-" * 42)

    anisos = []
    advs = []
    for arch in archs:
        a = results[(arch, 'anisotropy')]
        adv = results[(arch, 'advantage')]
        anisos.append(a)
        advs.append(adv)
        print(f"  {arch:>15}  {a:>12.2f}  {adv:>12.1f}x")

    corr = np.corrcoef(anisos, advs)[0, 1]
    print(f"\n  Correlation r = {corr:.3f}")

    # Hypothesis tests
    print(f"\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    t1 = corr > 0.5
    print(f"\n  T1: Positive correlation r > 0.5?")
    print(f"      r = {corr:.3f}")
    print(f"      --> {'PASS' if t1 else 'FAIL'}")

    max_aniso_arch = archs[np.argmax(anisos)]
    max_adv_arch = archs[np.argmax(advs)]
    t2 = max_aniso_arch == max_adv_arch
    print(f"\n  T2: Highest anisotropy = highest advantage?")
    print(f"      Max anisotropy: {max_aniso_arch}")
    print(f"      Max advantage:  {max_adv_arch}")
    print(f"      --> {'PASS' if t2 else 'FAIL'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
