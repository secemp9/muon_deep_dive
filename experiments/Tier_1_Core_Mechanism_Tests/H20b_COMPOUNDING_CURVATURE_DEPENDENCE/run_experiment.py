#!/usr/bin/env python3
"""
H20b: Compounding Requires Loss Landscape Curvature
=====================================================

FROM H15a + H20a: The tiny cosine advantage compounds over 500 steps into
a 19x loss improvement. But does this compounding happen on ALL loss surfaces
or only on curved ones?

HYPOTHESIS:
  Cosine compounding requires NON-ZERO Hessian curvature to amplify small
  directional differences. On a pure quadratic (constant Hessian), a 0.004
  cosine advantage gives exactly (1+0.004)^500 ~ 7.3x improvement (weak
  compounding). On a nonconvex surface with curvature that VARIES, the
  advantage compounds faster because better directions lead to regions with
  MORE favorable curvature (positive feedback loop).

  Prediction: loss_ratio at T=500 is:
    - Quadratic:  ~ 5-10x (mild compounding)
    - Deep linear (mild nonconvex): ~ 15-30x (moderate compounding)
    - ReLU MLP (strongly nonconvex): ~ 50-200x (strong compounding)

PROTOCOL:
  Three loss surfaces, all with same dimensionality (32 params):
  (a) Pure quadratic: L(theta) = theta^T A theta, A random PSD
  (b) Deep linear 2-layer 4x4: L = ||W2 W1 x - y||^2
  (c) ReLU MLP 2-layer 4x4: L = ||ReLU(W1 x) W2 - y||^2

  For each: train Muon and NormSGD at optimal LR for 500 steps.
  Measure loss ratio at {100, 200, 300, 400, 500}.
  Compare compounding rate (slope of log(loss_ratio) vs T).
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 4
NUM_STEPS = 500
NUM_SEEDS = 5
BATCH_SIZE = 64
NS_ITERS = 5
MOMENTUM = 0.9
CHECKPOINTS = [100, 200, 300, 400, 500]


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =========================================================================
# SURFACE (a): PURE QUADRATIC
# =========================================================================
class QuadraticSurface:
    def __init__(self, rng):
        M = rng.randn(DIM*2, DIM*2)
        self.A = M.T @ M + 0.1 * np.eye(DIM*2)  # PSD
        self.n_params = DIM * 2
        self.n_layers = 2  # fake "layers" for Muon
        self.layer_size = DIM

    def loss(self, theta):
        return 0.5 * theta @ self.A @ theta

    def grad_matrices(self, theta):
        g = self.A @ theta
        return [g[:DIM].reshape(1, DIM), g[DIM:].reshape(1, DIM)]

    def grad_vec(self, theta):
        return self.A @ theta


# =========================================================================
# SURFACE (b): DEEP LINEAR
# =========================================================================
class DeepLinearSurface:
    def __init__(self, rng):
        self.X = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.n_params = 2 * DIM * DIM
        self.n_layers = 2
        self.layer_size = DIM * DIM

    def _unpack(self, theta):
        return [theta[:DIM*DIM].reshape(DIM, DIM), theta[DIM*DIM:].reshape(DIM, DIM)]

    def loss(self, theta):
        W1, W2 = self._unpack(theta)
        pred = W2 @ W1 @ self.X
        return 0.5 * np.mean(np.sum((pred - self.Y)**2, axis=0))

    def grad_matrices(self, theta):
        W1, W2 = self._unpack(theta)
        N = self.X.shape[1]
        a1 = W1 @ self.X
        pred = W2 @ a1
        delta = (pred - self.Y) / N
        G2 = delta @ a1.T
        G1 = (W2.T @ delta) @ self.X.T
        return [G1, G2]

    def grad_vec(self, theta):
        gm = self.grad_matrices(theta)
        return np.concatenate([g.ravel() for g in gm])


# =========================================================================
# SURFACE (c): RELU MLP
# =========================================================================
class ReLUMLPSurface:
    def __init__(self, rng):
        self.X = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        self.n_params = 2 * DIM * DIM
        self.n_layers = 2
        self.layer_size = DIM * DIM

    def _unpack(self, theta):
        return [theta[:DIM*DIM].reshape(DIM, DIM), theta[DIM*DIM:].reshape(DIM, DIM)]

    def loss(self, theta):
        W1, W2 = self._unpack(theta)
        h = np.maximum(0, W1 @ self.X)
        pred = W2 @ h
        return 0.5 * np.mean(np.sum((pred - self.Y)**2, axis=0))

    def grad_matrices(self, theta):
        W1, W2 = self._unpack(theta)
        N = self.X.shape[1]
        pre = W1 @ self.X
        h = np.maximum(0, pre)
        pred = W2 @ h
        delta = (pred - self.Y) / N
        G2 = delta @ h.T
        delta_h = W2.T @ delta
        delta_h *= (pre > 0).astype(float)
        G1 = delta_h @ self.X.T
        return [G1, G2]

    def grad_vec(self, theta):
        gm = self.grad_matrices(theta)
        return np.concatenate([g.ravel() for g in gm])


def train_optimizer(surface, theta0, lr, opt_name):
    """Train and return losses at checkpoints."""
    theta = theta0.copy()
    n = surface.n_layers
    mom = [np.zeros(surface.layer_size if hasattr(surface, 'layer_size') else DIM)
           for _ in range(n)]

    checkpoint_losses = {}
    for step in range(NUM_STEPS):
        loss = surface.loss(theta)
        if not np.isfinite(loss) or loss > 1e10:
            for cp in CHECKPOINTS:
                if cp not in checkpoint_losses:
                    checkpoint_losses[cp] = float('inf')
            break

        grads = surface.grad_matrices(theta)

        # Build update
        updates = []
        for i, G in enumerate(grads):
            if G.ndim == 1:
                G = G.reshape(1, -1)
            if opt_name == 'muon':
                ortho_g = newton_schulz(G)
                mom[i] = MOMENTUM * mom[i].reshape(G.shape) + ortho_g
            else:  # normsgd
                nrm = np.linalg.norm(G, 'fro')
                norm_g = G / max(nrm, 1e-15)
                mom[i] = MOMENTUM * mom[i].reshape(G.shape) + norm_g
            updates.append(mom[i].ravel())

        delta = np.concatenate(updates)
        theta = theta - lr * delta

        if step + 1 in CHECKPOINTS:
            checkpoint_losses[step + 1] = surface.loss(theta)

    return checkpoint_losses


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H20b: COMPOUNDING REQUIRES LOSS LANDSCAPE CURVATURE")
    print("=" * 100)
    print(f"Surfaces: Quadratic, DeepLinear, ReLU MLP (all {DIM}x{DIM})")
    print(f"Steps: {NUM_STEPS}, Checkpoints: {CHECKPOINTS}")
    print()

    surface_names = ['Quadratic', 'DeepLinear', 'ReLU_MLP']

    # Results: surface -> checkpoint -> [ratios across seeds]
    all_ratios = {s: {cp: [] for cp in CHECKPOINTS} for s in surface_names}

    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        surfaces = {
            'Quadratic': QuadraticSurface(rng),
            'DeepLinear': DeepLinearSurface(np.random.RandomState(seed + 1000)),
            'ReLU_MLP': ReLUMLPSurface(np.random.RandomState(seed + 2000)),
        }

        for sname, surf in surfaces.items():
            # Init
            rng_init = np.random.RandomState(seed + 3000)
            theta0 = rng_init.randn(surf.n_params) * 0.3

            # Quick LR sweep for each
            lr_grid_muon = np.logspace(-4, -1, 12)
            lr_grid_norm = np.logspace(-3, 0, 12)

            best_muon_lr, best_muon_loss = lr_grid_muon[0], float('inf')
            for lr in lr_grid_muon:
                cl = train_optimizer(surf, theta0, lr, 'muon')
                fl = cl.get(200, float('inf'))
                if np.isfinite(fl) and fl < best_muon_loss:
                    best_muon_loss = fl
                    best_muon_lr = lr

            best_norm_lr, best_norm_loss = lr_grid_norm[0], float('inf')
            for lr in lr_grid_norm:
                cl = train_optimizer(surf, theta0, lr, 'normsgd')
                fl = cl.get(200, float('inf'))
                if np.isfinite(fl) and fl < best_norm_loss:
                    best_norm_loss = fl
                    best_norm_lr = lr

            # Full run
            losses_muon = train_optimizer(surf, theta0, best_muon_lr, 'muon')
            losses_norm = train_optimizer(surf, theta0, best_norm_lr, 'normsgd')

            for cp in CHECKPOINTS:
                lm = losses_muon.get(cp, float('inf'))
                ln = losses_norm.get(cp, float('inf'))
                ratio = ln / max(lm, 1e-30)
                all_ratios[sname][cp].append(ratio)

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n{'='*100}")
    print("RESULTS: COMPOUNDING RATE BY SURFACE TYPE")
    print(f"{'='*100}")

    print(f"\n  {'Surface':<15}", end="")
    for cp in CHECKPOINTS:
        print(f"  {'T='+str(cp):>10}", end="")
    print(f"  {'Rate (slope)':>12}")
    print("  " + "-" * (15 + 12 * len(CHECKPOINTS) + 14))

    compounding_rates = {}
    for sname in surface_names:
        print(f"  {sname:<15}", end="")
        log_ratios = []
        for cp in CHECKPOINTS:
            mr = np.mean(all_ratios[sname][cp])
            print(f"  {mr:>10.1f}x", end="")
            log_ratios.append(np.log(max(mr, 1e-10)))

        # Fit log(ratio) vs T
        slope, _ = np.polyfit(CHECKPOINTS, log_ratios, 1)
        compounding_rates[sname] = slope
        print(f"  {slope:>12.6f}")

    # Tests
    print(f"\n  Compounding rates:")
    for sname in surface_names:
        print(f"    {sname}: {compounding_rates[sname]:.6f} per step")

    quad_rate = compounding_rates['Quadratic']
    linear_rate = compounding_rates['DeepLinear']
    relu_rate = compounding_rates['ReLU_MLP']

    t1 = relu_rate > linear_rate > quad_rate
    print(f"\n  T1: ReLU > DeepLinear > Quadratic compounding?  --> {'PASS' if t1 else 'FAIL'}")
    print(f"       Quad={quad_rate:.6f}, Linear={linear_rate:.6f}, ReLU={relu_rate:.6f}")

    t2 = relu_rate > 2 * quad_rate if quad_rate > 0 else relu_rate > 0
    print(f"  T2: ReLU compounds >2x faster than Quadratic?   --> {'PASS' if t2 else 'FAIL'}")

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
