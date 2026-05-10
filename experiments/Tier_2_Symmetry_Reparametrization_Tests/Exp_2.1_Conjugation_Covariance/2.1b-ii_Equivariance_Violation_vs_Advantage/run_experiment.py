#!/usr/bin/env python3
"""
2.1b-ii: Equivariance Violation Correlates with Per-Layer Muon Advantage?
==========================================================================

MOTIVATION (from 2.1b surprise):
  Equivariance breaks linearly for data-driven losses. This means some
  layers have more equivariance violation than others.

QUESTION: Do layers with LARGER equivariance violation benefit MORE or
  LESS from Muon? Two competing hypotheses:
  (A) More violation = more gauge drift = MORE benefit from gauge-fixing.
  (B) More violation = equivariance already broken = gauge-fixing LESS relevant.

PROTOCOL:
  Train a deep net with SGD and Muon independently.
  For each layer, measure:
    - Equivariance violation (from conjugation test).
    - Per-layer Muon advantage = SGD_kappa / Muon_kappa at convergence.
    - Per-layer loss contribution reduction.
  Correlate violation with advantage across layers.

KEY TESTS:
  T1: Correlation between violation and kappa-advantage is positive
      (more violation = more benefit from Muon).
  T2: Correlation between violation and loss-advantage is positive.
  T3: The correlation is strong (|r| > 0.5).

Setup: 8-layer, 32x32, 300 steps, 5 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
DEPTH = 8
NUM_STEPS = 300
LR_MUON = 0.01
LR_SGD = 0.005
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def random_orthogonal(n, rng):
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(DEPTH)]


def compute_loss_and_grads(weights, X, Y):
    L = len(weights)
    N = X.shape[1]
    acts = [X.copy()]
    for W in weights:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    loss = 0.5 * np.mean(np.sum((acts[-1] - Y)**2, axis=0))
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = delta @ acts[l].T
        if l > 0:
            delta = weights[l].T @ delta
    return loss, grads


def train(weights_init, X, Y, optimizer, lr):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(NUM_STEPS):
        loss, grads = compute_loss_and_grads(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            break
        for i in range(len(weights)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            weights[i] = weights[i] - lr * mom[i]
    return weights


def measure_equivariance_violation(weights_init, X, Y, target_layer, rng):
    R = random_orthogonal(DIM, rng)
    S = random_orthogonal(DIM, rng)

    weights_A = train([W.copy() for W in weights_init], X, Y, 'muon', LR_MUON)

    weights_conj = [W.copy() for W in weights_init]
    weights_conj[target_layer] = R @ weights_conj[target_layer] @ S.T
    weights_B = train(weights_conj, X, Y, 'muon', LR_MUON)

    expected = R @ weights_A[target_layer] @ S.T
    actual = weights_B[target_layer]
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(weights_A[target_layer]), 1e-30)


def main():
    print("=" * 100)
    print("2.1b-ii: EQUIVARIANCE VIOLATION vs PER-LAYER MUON ADVANTAGE")
    print("=" * 100)
    print(f"Network: {DEPTH}-layer, {DIM}x{DIM}, {NUM_STEPS} steps, {NUM_SEEDS} seeds")
    print()

    violations = {l: [] for l in range(DEPTH)}
    kappa_advantages = {l: [] for l in range(DEPTH)}

    for seed_idx in range(NUM_SEEDS):
        seed = 42 + seed_idx * 137
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, BATCH_SIZE) * 0.3
        Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        weights_init = init_weights(seed + 5000)

        # Train with both optimizers
        final_sgd = train([W.copy() for W in weights_init], X, Y, 'sgd', LR_SGD)
        final_muon = train([W.copy() for W in weights_init], X, Y, 'muon', LR_MUON)

        for l in range(DEPTH):
            # Kappa advantage
            svs_sgd = np.linalg.svd(final_sgd[l], compute_uv=False)
            svs_muon = np.linalg.svd(final_muon[l], compute_uv=False)
            kappa_sgd = svs_sgd[0] / max(svs_sgd[-1], 1e-12)
            kappa_muon = svs_muon[0] / max(svs_muon[-1], 1e-12)
            kappa_advantages[l].append(kappa_sgd / max(kappa_muon, 1e-12))

            # Equivariance violation
            rng_v = np.random.RandomState(seed + l * 1000)
            viol = measure_equivariance_violation(weights_init, X, Y, l, rng_v)
            violations[l].append(viol)

    # Results
    print(f"\n{'=' * 100}")
    print("PER-LAYER RESULTS")
    print(f"{'=' * 100}")

    print(f"\n  {'Layer':>6}  {'Violation':>12}  {'kappa ratio':>14}  {'Violation std':>14}")
    print("  " + "-" * 50)

    mean_viols = []
    mean_kappas = []
    for l in range(DEPTH):
        mv = np.mean(violations[l])
        mk = np.mean(kappa_advantages[l])
        mean_viols.append(mv)
        mean_kappas.append(mk)
        print(f"  {l:>6}  {mv:>12.4e}  {mk:>14.2f}x  {np.std(violations[l]):>14.4e}")

    corr = np.corrcoef(mean_viols, mean_kappas)[0, 1]
    print(f"\n  Correlation(violation, kappa_advantage) = {corr:.3f}")

    # Hypothesis tests
    print(f"\n\n{'=' * 100}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 100}")

    t1 = corr > 0
    t3 = abs(corr) > 0.5

    print(f"\n  T1: Positive correlation (more violation -> more benefit)?")
    print(f"      r = {corr:.3f}")
    print(f"      --> {'PASS' if t1 else 'FAIL (negative correlation)'}")

    print(f"\n  T3: Strong correlation (|r| > 0.5)?")
    print(f"      |r| = {abs(corr):.3f}")
    print(f"      --> {'PASS' if t3 else 'FAIL'}")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
