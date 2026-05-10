#!/usr/bin/env python3
"""
H20a: Cosine Advantage Compounds Geometrically Over Training
==============================================================

FROM H15a: Muon vs NormSGD Newton alignment is slim (+0.004 cosine per step),
but Muon beats NormSGD by 19x on loss. The per-step edge is tiny but COMPOUNDS.

HYPOTHESIS:
  The +0.004 cosine advantage per step is not additive but MULTIPLICATIVE in
  its effect on loss. Each step, Muon lands slightly closer to the Newton
  direction, which means the NEXT step starts from a slightly better point
  with a slightly better gradient. This creates geometric compounding:
  loss_ratio(T) ~ exp(alpha * T) where alpha is proportional to the
  per-step cosine advantage.

  At T=500 steps: exp(0.004 * 500) = exp(2) ~ 7.4x, which is order-of-
  magnitude consistent with the observed 19x (the rest comes from the
  non-constant nature of the advantage).

PROTOCOL:
  2-layer deep linear 4x4, 500 steps. Track at checkpoints {50,100,200,500}:
    1. Cumulative cosine advantage: sum of (cos_muon - cos_normsgd) up to step t
    2. Loss ratio: loss_normsgd(t) / loss_muon(t)
    3. Trajectory divergence: ||theta_muon(t) - theta_normsgd(t)||_F

  Fit: log(loss_ratio) vs cumulative_cosine_advantage. If slope > 0 and
  the relationship is roughly linear, compounding is confirmed.

  Also measure: does the per-step cosine advantage GROW over training?
  (Would explain why observed 19x > predicted 7.4x)

KEY MEASUREMENTS:
  - Per-step cosine advantage time series
  - Cumulative sum of cosine advantages at {50, 100, 200, 500}
  - Loss ratio at each checkpoint
  - Fit of log(loss_ratio) ~ c * cumulative_cosine_sum
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 4
N_LAYERS = 2
N_PARAMS = N_LAYERS * DIM * DIM  # 32
NUM_STEPS = 500
NUM_SEEDS = 5
BATCH_SIZE = 64
FD_EPS = 1e-5
NS_ITERS = 5

CHECKPOINTS = [50, 100, 200, 300, 400, 500]


def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta):
    return [theta[i*DIM*DIM:(i+1)*DIM*DIM].reshape(DIM, DIM) for i in range(N_LAYERS)]


def forward(Ws, X):
    out = X.copy()
    for W in Ws:
        out = W @ out
    return out


def loss_fn(theta, X, Y):
    Ws = unpack(theta)
    pred = forward(Ws, X)
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


def grad_fn(theta, X, Y):
    Ws = unpack(theta)
    N = X.shape[1]
    acts = [X.copy()]
    for W in Ws:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / N
    grads = []
    for l in range(N_LAYERS - 1, -1, -1):
        grads.insert(0, delta @ acts[l].T)
        if l > 0:
            delta = Ws[l].T @ delta
    return pack(grads), grads


def hessian_fd(theta, X, Y):
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        tp = theta.copy(); tp[i] += FD_EPS
        tm = theta.copy(); tm[i] -= FD_EPS
        gp, _ = grad_fn(tp, X, Y)
        gm, _ = grad_fn(tm, X, Y)
        H[:, i] = (gp - gm) / (2 * FD_EPS)
    return 0.5 * (H + H.T)


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return np.dot(a, b) / (na * nb)


def muon_step_vec(grads_list):
    return pack([-newton_schulz(G) for G in grads_list])


def normsgd_step_vec(grads_list):
    dirs = []
    for G in grads_list:
        nrm = np.linalg.norm(G, 'fro')
        dirs.append(-G / max(nrm, 1e-15))
    return pack(dirs)


def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H20a: COSINE ADVANTAGE COMPOUNDS GEOMETRICALLY")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM} ({N_PARAMS} params)")
    print(f"Steps: {NUM_STEPS}, Checkpoints: {CHECKPOINTS}")
    print()

    # First find optimal LRs
    lr_grid_muon = np.logspace(-4, -1, 15)
    lr_grid_norm = np.logspace(-3, 0, 15)

    # Quick sweep on first seed
    rng = np.random.RandomState(seeds[0])
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = rng.randn(DIM, BATCH_SIZE) * 0.3

    best_lr = {}
    for opt_name, lr_grid, step_fn in [('muon', lr_grid_muon, muon_step_vec),
                                        ('normsgd', lr_grid_norm, normsgd_step_vec)]:
        best, best_l = lr_grid[0], float('inf')
        for lr in lr_grid:
            theta = pack([np.eye(DIM) + rng.randn(DIM, DIM)*0.1 for _ in range(N_LAYERS)])
            rng2 = np.random.RandomState(seeds[0])
            theta = pack([np.eye(DIM) + rng2.randn(DIM, DIM)*0.1 for _ in range(N_LAYERS)])
            ok = True
            for t in range(200):
                g, gl = grad_fn(theta, X, Y)
                d = step_fn(gl)
                theta = theta + lr * d
                l = loss_fn(theta, X, Y)
                if not np.isfinite(l) or l > 1e6:
                    ok = False
                    break
            if ok and l < best_l:
                best_l = l
                best = lr
        best_lr[opt_name] = best
        print(f"  {opt_name}: optimal LR = {best:.6f}")

    # Main measurement loop
    all_cos_advs = []
    all_cumul_at_checkpoints = {cp: [] for cp in CHECKPOINTS}
    all_loss_ratios_at_checkpoints = {cp: [] for cp in CHECKPOINTS}

    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, BATCH_SIZE) * 0.3
        Y = rng.randn(DIM, BATCH_SIZE) * 0.3
        weights_init = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]

        # Train both optimizers from same init
        theta_muon = pack([W.copy() for W in weights_init])
        theta_norm = pack([W.copy() for W in weights_init])

        cos_adv_series = []
        cumul_cos = 0.0

        for step in range(NUM_STEPS):
            # Compute Newton direction at Muon's point (for reference)
            compute_hessian = (step in CHECKPOINTS or step % 50 == 0)

            # Muon step
            g_m, gl_m = grad_fn(theta_muon, X, Y)
            d_muon = muon_step_vec(gl_m)

            # NormSGD step
            g_n, gl_n = grad_fn(theta_norm, X, Y)
            d_norm = normsgd_step_vec(gl_n)

            # Newton direction at Muon's point
            if compute_hessian:
                H = hessian_fd(theta_muon, X, Y)
                H_pinv = np.linalg.pinv(H, rcond=1e-6)
                d_newton = -H_pinv @ g_m

                cos_muon_newton = cosine(d_muon, d_newton)
                cos_norm_newton = cosine(d_norm, d_newton)
                cos_adv = cos_muon_newton - cos_norm_newton
            else:
                cos_adv = 0.0  # Don't compute Hessian every step

            cos_adv_series.append(cos_adv)
            cumul_cos += cos_adv

            # Take steps
            theta_muon = theta_muon + best_lr['muon'] * d_muon
            theta_norm = theta_norm + best_lr['normsgd'] * d_norm

            # Record at checkpoints
            if step + 1 in CHECKPOINTS:
                loss_m = loss_fn(theta_muon, X, Y)
                loss_n = loss_fn(theta_norm, X, Y)
                ratio = loss_n / max(loss_m, 1e-30)
                all_cumul_at_checkpoints[step + 1].append(cumul_cos)
                all_loss_ratios_at_checkpoints[step + 1].append(ratio)

        all_cos_advs.append(cos_adv_series)
        print(f"  Seed {si+1}: final loss_ratio={all_loss_ratios_at_checkpoints[NUM_STEPS][-1]:.2f}x, "
              f"cumul_cos_adv={cumul_cos:.4f}")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'='*100}")
    print("RESULTS: COMPOUNDING OF COSINE ADVANTAGE")
    print(f"{'='*100}")

    print(f"\n  {'Checkpoint':>12}  {'Mean Cumul Cos':>16}  {'Mean Loss Ratio':>16}  {'log(ratio)':>12}")
    print("  " + "-" * 60)

    cps = []
    log_ratios = []
    cumuls = []
    for cp in CHECKPOINTS:
        mc = np.mean(all_cumul_at_checkpoints[cp])
        mr = np.mean(all_loss_ratios_at_checkpoints[cp])
        lr_val = np.log(max(mr, 1e-10))
        print(f"  {cp:>12}  {mc:>16.4f}  {mr:>16.2f}x  {lr_val:>12.4f}")
        cps.append(cp)
        log_ratios.append(lr_val)
        cumuls.append(mc)

    # Fit: log(loss_ratio) vs cumulative cosine
    if len(cumuls) >= 3:
        slope, intercept = np.polyfit(cumuls, log_ratios, 1)
        print(f"\n  Fit: log(loss_ratio) = {slope:.2f} * cumul_cos + {intercept:.4f}")
        print(f"  Interpretation: each unit of cumulative cosine advantage")
        print(f"  multiplies the loss ratio by exp({slope:.2f}) = {np.exp(slope):.2f}x")

    # Also fit log(loss_ratio) vs step count
    slope_t, intercept_t = np.polyfit(cps, log_ratios, 1)
    print(f"\n  Fit: log(loss_ratio) = {slope_t:.6f} * T + {intercept_t:.4f}")
    print(f"  Exponential rate: {slope_t:.6f} per step => exp(rate*500) = {np.exp(slope_t*500):.2f}x")

    # Does per-step advantage grow over training?
    early_advs = [np.mean([s[i] for s in all_cos_advs if i < len(s)])
                  for i in range(0, min(50, NUM_STEPS), 10) if i % 50 == 0]
    late_advs = [np.mean([s[i] for s in all_cos_advs if i < len(s)])
                 for i in range(max(0, NUM_STEPS-100), NUM_STEPS, 50) if i % 50 == 0]

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
