#!/usr/bin/env python3
"""
2.1b-i: Equivariance Error vs Depth -- Layer Distance from Data
=================================================================

MOTIVATION (from 2.1b surprise):
  Equivariance holds iff the loss is conjugation-invariant. For inter-layer
  gauge (where both adjacent layers transform), equivariance is EXACT.
  For per-layer transforms with data-driven loss, it breaks linearly
  (0.195 error over 50 steps).

QUESTION: In a deep net, does equivariance error depend on how far the
  layer is from the data (input/output)? Middle layers have gradients that
  pass through more nonlinear transformations, potentially amplifying the
  non-equivariant component.

PROTOCOL:
  Train an L-layer deep linear net (L in {4, 8, 16}).
  At each layer l, measure equivariance error:
    Path A: train from W_l^0, get W_l^T.
    Path B: train from R W_l^0 S^T, get W_l'^T.
    Error = ||W_l'^T - R W_l^T S^T|| / ||W_l^T||.
  Plot error vs layer index l for each depth L.

KEY TESTS:
  T1: Error is smallest at layers adjacent to data (l=0 or l=L-1).
  T2: Error peaks at middle layers (parabolic profile).
  T3: Error at the conjugated layer is near-zero (the gauge-partner layer).

Setup: 32x32, 100 steps per run, 10 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
DEPTHS = [4, 8, 16]
NUM_STEPS = 100
LR = 0.01
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 10
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


def init_weights(depth, seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(depth)]


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


def train_muon(weights_init, X, Y, n_steps):
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    for step in range(n_steps):
        loss, grads = compute_loss_and_grads(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            break
        for i in range(len(weights)):
            mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            weights[i] = weights[i] - LR * mom[i]
    return weights


def main():
    print("=" * 100)
    print("2.1b-i: EQUIVARIANCE ERROR vs DEPTH -- Layer Distance from Data")
    print("=" * 100)
    print(f"Depths: {DEPTHS}, DIM={DIM}, Steps={NUM_STEPS}, Seeds={NUM_SEEDS}")
    print()

    for depth in DEPTHS:
        print(f"\n{'=' * 80}")
        print(f"  DEPTH = {depth}")
        print(f"{'=' * 80}")

        # For each layer, conjugate THAT layer and measure drift
        errors_by_layer = {l: [] for l in range(depth)}

        for seed in range(NUM_SEEDS):
            rng = np.random.RandomState(42 + seed * 137)
            X = rng.randn(DIM, BATCH_SIZE) * 0.3
            Y = rng.randn(DIM, BATCH_SIZE) * 0.3

            R = random_orthogonal(DIM, rng)
            S = random_orthogonal(DIM, rng)

            weights_init = init_weights(depth, seed + 5000)

            # Path A: original
            weights_A = train_muon(weights_init, X, Y, NUM_STEPS)

            # For each layer, conjugate ONLY that layer
            for target_layer in range(depth):
                weights_conj = [W.copy() for W in weights_init]
                weights_conj[target_layer] = R @ weights_conj[target_layer] @ S.T

                weights_B = train_muon(weights_conj, X, Y, NUM_STEPS)

                # Check if target layer is equivariant
                expected = R @ weights_A[target_layer] @ S.T
                actual = weights_B[target_layer]
                err = np.linalg.norm(actual - expected) / max(np.linalg.norm(weights_A[target_layer]), 1e-30)
                errors_by_layer[target_layer].append(err)

        # Print results
        print(f"\n  {'Layer':>6}  {'Mean Error':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
        print("  " + "-" * 50)
        for l in range(depth):
            errs = np.array(errors_by_layer[l])
            dist_from_edge = min(l, depth - 1 - l)
            print(f"  {l:>6}  {np.mean(errs):>12.4e}  {np.std(errs):>10.4e}  "
                  f"{np.min(errs):>10.4e}  {np.max(errs):>10.4e}  "
                  f"(dist_from_edge={dist_from_edge})")

        # Test: does error correlate with distance from edge?
        mean_errors = [np.mean(errors_by_layer[l]) for l in range(depth)]
        dists = [min(l, depth - 1 - l) for l in range(depth)]
        corr = np.corrcoef(dists, mean_errors)[0, 1] if len(set(dists)) > 1 else 0
        print(f"\n  Correlation(dist_from_edge, error) = {corr:.3f}")

    print(f"\n\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
