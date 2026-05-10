#!/usr/bin/env python3
"""
H20a: COSINE COMPOUNDING -- Does +0.004/step Become 19x?
==========================================================

MOTIVATION:
  H15a: Muon's per-step Newton alignment is only +0.004 better than NormSGD.
  H3:   Muon beats NormSGD by 19x on loss.
  How does a tiny per-step edge produce a huge loss gap?

HYPOTHESIS:
  The cosine advantage COMPOUNDS geometrically. Each step, Muon is slightly
  closer to Newton => starts next step from a slightly better point => slightly
  better gradient => the advantage accumulates multiplicatively:
    loss_ratio(T) ~ exp(alpha * cumulative_cosine_advantage)

PROTOCOL:
  4-layer deep linear 32x32, 500 steps. At every step, train both Muon and
  NormSGD from the SAME init. At every 50 steps, compute full Hessian and:
    1. Newton direction at each optimizer's current point
    2. cos(step, Newton) for Muon and NormSGD
    3. Cumulative cosine advantage: sum(cos_Muon(t) - cos_NormSGD(t))
    4. Loss ratio: loss_NormSGD(t) / loss_Muon(t)

  Fit two models:
    (A) log(loss_ratio) vs cumulative_cos_advantage  -- if linear => geometric compounding
    (B) log(loss_ratio) vs t                          -- if linear => simple exponential

  KEY TEST: does cumulative cosine advantage predict loss ratio BETTER than
  just step count? Compare R^2 of both fits.

Setup: 4-layer, 32x32, 500 steps, 5 seeds.
"""

import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 500
NUM_SEEDS = 5
BATCH_SIZE = 64
MOMENTUM = 0.9
NS_ITERS = 5

# Checkpoints where we compute Hessian-based Newton direction
# We measure every 50 steps for the cosine computation
MEASURE_EVERY = 50
CHECKPOINTS = list(range(MEASURE_EVERY, NUM_STEPS + 1, MEASURE_EVERY))


# =============================================================================
# NETWORK
# =============================================================================

def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, 'fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def init_weights(seed):
    rng = np.random.RandomState(seed)
    return [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]


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


def pack(weights):
    return np.concatenate([W.ravel() for W in weights])


def unpack(vec):
    ws = []
    offset = 0
    for _ in range(N_LAYERS):
        ws.append(vec[offset:offset + DIM * DIM].reshape(DIM, DIM))
        offset += DIM * DIM
    return ws


def make_data(seed):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


def loss_fn_flat(theta, X, Y):
    ws = unpack(theta)
    return compute_loss(ws, X, Y)


def grad_fn_flat(theta, X, Y):
    ws = unpack(theta)
    grads = compute_gradients(ws, X, Y)
    return pack(grads), grads


def hessian_fd(theta, X, Y, eps=1e-5):
    """Finite-difference Hessian. Only used on subsampled params for 32x32."""
    n = len(theta)
    H = np.zeros((n, n))
    g0, _ = grad_fn_flat(theta, X, Y)
    for i in range(n):
        tp = theta.copy()
        tp[i] += eps
        gp, _ = grad_fn_flat(tp, X, Y)
        H[:, i] = (gp - g0) / eps
    return 0.5 * (H + H.T)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return np.dot(a, b) / (na * nb)


# =============================================================================
# APPROXIMATE NEWTON DIRECTION
# =============================================================================
# For 32x32, 4 layers: 4096 params. Full Hessian is 4096x4096 = too expensive.
# Instead, use the conjugate gradient Newton direction:
#   d_N = -H^{-1} g  approximated via CG with Hessian-vector products.

def hessian_vec_product(theta, X, Y, v, eps=1e-5):
    """Hessian-vector product via finite differences: H @ v ~ (grad(theta+eps*v) - grad(theta-eps*v)) / (2*eps)."""
    vn = np.linalg.norm(v)
    if vn < 1e-15:
        return np.zeros_like(v)
    tp = theta + eps * v
    tm = theta - eps * v
    gp, _ = grad_fn_flat(tp, X, Y)
    gm, _ = grad_fn_flat(tm, X, Y)
    return (gp - gm) / (2 * eps)


def cg_newton(theta, X, Y, g, max_iters=50, tol=1e-6):
    """Conjugate gradient to approximately solve H @ d = -g."""
    b = -g
    d = np.zeros_like(g)
    r = b.copy()
    p = r.copy()
    rsold = np.dot(r, r)
    if rsold < tol ** 2:
        return d
    for _ in range(max_iters):
        Ap = hessian_vec_product(theta, X, Y, p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-15:
            break
        alpha = rsold / pAp
        d = d + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if rsnew < tol ** 2:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return d


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_muon_trajectory(weights_init, X, Y, lr):
    """Train with Muon, return loss at every step."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    losses = []
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        losses.append(loss)
        if not np.isfinite(loss) or loss > 1e10:
            losses.extend([float('inf')] * (NUM_STEPS - step - 1))
            break
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            mom[l] = MOMENTUM * mom[l] + newton_schulz(grads[l])
            weights[l] = weights[l] - lr * mom[l]
    return losses, weights


def train_normsgd_trajectory(weights_init, X, Y, lr):
    """Train with NormSGD, return loss at every step."""
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]
    losses = []
    for step in range(NUM_STEPS):
        loss = compute_loss(weights, X, Y)
        losses.append(loss)
        if not np.isfinite(loss) or loss > 1e10:
            losses.extend([float('inf')] * (NUM_STEPS - step - 1))
            break
        grads = compute_gradients(weights, X, Y)
        for l in range(N_LAYERS):
            nrm = np.linalg.norm(grads[l], 'fro')
            normed = grads[l] / max(nrm, 1e-15)
            mom[l] = MOMENTUM * mom[l] + normed
            weights[l] = weights[l] - lr * mom[l]
    return losses, weights


# =============================================================================
# FULL TRAJECTORY WITH COSINE MEASUREMENT
# =============================================================================

def run_trajectory_with_cosine(weights_init, X, Y, lr_muon, lr_norm):
    """
    Run Muon and NormSGD in parallel from same init.
    At every MEASURE_EVERY steps, compute cosine(step, Newton) for both.
    """
    w_muon = [W.copy() for W in weights_init]
    w_norm = [W.copy() for W in weights_init]
    mom_muon = [np.zeros_like(W) for W in weights_init]
    mom_norm = [np.zeros_like(W) for W in weights_init]

    cos_advantages = []  # per-checkpoint cosine advantage
    loss_muon_at_cp = {}
    loss_norm_at_cp = {}
    cumul_cos = 0.0
    cumul_at_cp = {}

    for step in range(NUM_STEPS):
        l_muon = compute_loss(w_muon, X, Y)
        l_norm = compute_loss(w_norm, X, Y)

        if not np.isfinite(l_muon) or l_muon > 1e10:
            l_muon = float('inf')
        if not np.isfinite(l_norm) or l_norm > 1e10:
            l_norm = float('inf')

        # Measure cosine at checkpoints
        if (step + 1) in CHECKPOINTS or step == 0:
            theta_muon = pack(w_muon)
            g_muon_flat, grads_muon = grad_fn_flat(theta_muon, X, Y)

            theta_norm = pack(w_norm)
            g_norm_flat, grads_norm = grad_fn_flat(theta_norm, X, Y)

            # Newton direction at Muon's point (using CG)
            d_newton_muon = cg_newton(theta_muon, X, Y, g_muon_flat, max_iters=30)
            # Newton direction at NormSGD's point
            d_newton_norm = cg_newton(theta_norm, X, Y, g_norm_flat, max_iters=30)

            # Muon step direction (at Muon's point)
            muon_step = pack([newton_schulz(G) for G in grads_muon])
            # NormSGD step direction (at NormSGD's point)
            norm_step = pack([G / max(np.linalg.norm(G, 'fro'), 1e-15) for G in grads_norm])

            # Cosine with Newton
            cos_muon_newton = cosine(-muon_step, d_newton_muon)
            cos_norm_newton = cosine(-norm_step, d_newton_norm)
            cos_adv = cos_muon_newton - cos_norm_newton
            cos_advantages.append(cos_adv)
            cumul_cos += cos_adv

        # Record at checkpoints
        if (step + 1) in CHECKPOINTS:
            loss_muon_at_cp[step + 1] = l_muon
            loss_norm_at_cp[step + 1] = l_norm
            cumul_at_cp[step + 1] = cumul_cos

        # Take Muon step
        if np.isfinite(l_muon) and l_muon < 1e10:
            grads_m = compute_gradients(w_muon, X, Y)
            for l in range(N_LAYERS):
                mom_muon[l] = MOMENTUM * mom_muon[l] + newton_schulz(grads_m[l])
                w_muon[l] = w_muon[l] - lr_muon * mom_muon[l]

        # Take NormSGD step
        if np.isfinite(l_norm) and l_norm < 1e10:
            grads_n = compute_gradients(w_norm, X, Y)
            for l in range(N_LAYERS):
                nrm = np.linalg.norm(grads_n[l], 'fro')
                normed = grads_n[l] / max(nrm, 1e-15)
                mom_norm[l] = MOMENTUM * mom_norm[l] + normed
                w_norm[l] = w_norm[l] - lr_norm * mom_norm[l]

    return cos_advantages, cumul_at_cp, loss_muon_at_cp, loss_norm_at_cp


# =============================================================================
# LR SWEEP
# =============================================================================

def sweep_lr(train_fn, weights_init, X, Y, candidates):
    best_lr, best_loss = candidates[0], float('inf')
    for lr in candidates:
        losses, _ = train_fn([W.copy() for W in weights_init], X, Y, lr)
        final = losses[-1] if losses else float('inf')
        if np.isfinite(final) and final < best_loss:
            best_loss = final
            best_lr = lr
    return best_lr


# =============================================================================
# MAIN
# =============================================================================

def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 100)
    print("H20a: COSINE COMPOUNDING -- Does +0.004/step Compound to 19x?")
    print("=" * 100)
    print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}")
    print(f"Steps: {NUM_STEPS}, Checkpoints: {CHECKPOINTS}")
    print(f"Seeds: {NUM_SEEDS}")
    print()

    lr_muon_candidates = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
    lr_norm_candidates = [0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]

    # LR sweep on first seed
    print("LR sweep on first seed...")
    X0, Y0 = make_data(seeds[0])
    w0 = init_weights(seeds[0] + 5000)

    best_lr_muon = sweep_lr(train_muon_trajectory, w0, X0, Y0, lr_muon_candidates)
    best_lr_norm = sweep_lr(train_normsgd_trajectory, w0, X0, Y0, lr_norm_candidates)
    print(f"  Muon LR: {best_lr_muon}")
    print(f"  NormSGD LR: {best_lr_norm}")
    print()

    # Main measurement across seeds
    all_cumul = {cp: [] for cp in CHECKPOINTS}
    all_loss_ratio = {cp: [] for cp in CHECKPOINTS}
    all_cos_advs = []

    for si, seed in enumerate(seeds):
        print(f"  Seed {si + 1}/{NUM_SEEDS} (seed={seed})...")
        X, Y = make_data(seed)
        w_init = init_weights(seed + 5000)

        cos_advs, cumul_at_cp, loss_m_cp, loss_n_cp = run_trajectory_with_cosine(
            w_init, X, Y, best_lr_muon, best_lr_norm
        )
        all_cos_advs.append(cos_advs)

        for cp in CHECKPOINTS:
            if cp in cumul_at_cp and cp in loss_m_cp and cp in loss_n_cp:
                all_cumul[cp].append(cumul_at_cp[cp])
                lm = loss_m_cp[cp]
                ln = loss_n_cp[cp]
                if np.isfinite(lm) and np.isfinite(ln) and lm > 1e-30:
                    ratio = ln / lm
                else:
                    ratio = float('nan')
                all_loss_ratio[cp].append(ratio)

        # Print per-seed summary
        final_cp = CHECKPOINTS[-1]
        if all_loss_ratio[final_cp]:
            print(f"    Final loss_ratio={all_loss_ratio[final_cp][-1]:.2f}x, "
                  f"cumul_cos={all_cumul[final_cp][-1]:.4f}")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print(f"\n\n{'=' * 100}")
    print("RESULTS: COSINE COMPOUNDING ANALYSIS")
    print(f"{'=' * 100}")

    print(f"\n  {'Step':>6}  {'Mean Cumul Cos':>16}  {'Mean Loss Ratio':>16}  {'log(ratio)':>12}")
    print("  " + "-" * 55)

    cps_valid = []
    log_ratios = []
    cumuls = []
    steps_valid = []

    for cp in CHECKPOINTS:
        ratios = [r for r in all_loss_ratio[cp] if np.isfinite(r) and r > 0]
        cums = [c for c in all_cumul[cp] if np.isfinite(c)]
        if ratios and cums:
            mr = np.mean(ratios)
            mc = np.mean(cums)
            lr_val = np.log(max(mr, 1e-30))
            print(f"  {cp:>6}  {mc:>16.4f}  {mr:>16.4f}x  {lr_val:>12.4f}")
            cps_valid.append(cp)
            log_ratios.append(lr_val)
            cumuls.append(mc)
            steps_valid.append(cp)

    # ---- FIT A: log(loss_ratio) vs cumulative cosine advantage ----
    print(f"\n  === FIT A: log(loss_ratio) vs cumulative_cosine_advantage ===")
    if len(cumuls) >= 3:
        slope_cos, intercept_cos = np.polyfit(cumuls, log_ratios, 1)
        # R^2
        pred_cos = np.array(cumuls) * slope_cos + intercept_cos
        ss_res_cos = np.sum((np.array(log_ratios) - pred_cos) ** 2)
        ss_tot = np.sum((np.array(log_ratios) - np.mean(log_ratios)) ** 2)
        r2_cos = 1 - ss_res_cos / max(ss_tot, 1e-30)
        print(f"  Fit: log(ratio) = {slope_cos:.4f} * cumul_cos + {intercept_cos:.4f}")
        print(f"  R^2 = {r2_cos:.4f}")
        print(f"  Interpretation: each unit of cumul cos advantage multiplies "
              f"ratio by exp({slope_cos:.4f}) = {np.exp(slope_cos):.2f}x")
    else:
        r2_cos = float('nan')
        print("  Not enough data points for fit")

    # ---- FIT B: log(loss_ratio) vs step count ----
    print(f"\n  === FIT B: log(loss_ratio) vs step_count ===")
    if len(steps_valid) >= 3:
        slope_t, intercept_t = np.polyfit(steps_valid, log_ratios, 1)
        pred_t = np.array(steps_valid) * slope_t + intercept_t
        ss_res_t = np.sum((np.array(log_ratios) - pred_t) ** 2)
        r2_t = 1 - ss_res_t / max(ss_tot, 1e-30)
        print(f"  Fit: log(ratio) = {slope_t:.6f} * T + {intercept_t:.4f}")
        print(f"  R^2 = {r2_t:.4f}")
        print(f"  Rate: {slope_t:.6f}/step => exp(rate*500) = {np.exp(slope_t * 500):.2f}x predicted at T=500")
    else:
        r2_t = float('nan')
        print("  Not enough data points for fit")

    # ---- KEY TEST: which fit is better? ----
    print(f"\n  === KEY TEST: Does cumul cosine predict loss ratio BETTER than step count? ===")
    if np.isfinite(r2_cos) and np.isfinite(r2_t):
        if r2_cos > r2_t:
            print(f"  --> YES: R^2(cosine)={r2_cos:.4f} > R^2(step)={r2_t:.4f}")
            print(f"      Geometric compounding via cosine advantage CONFIRMED")
        else:
            print(f"  --> NO: R^2(cosine)={r2_cos:.4f} <= R^2(step)={r2_t:.4f}")
            print(f"      Simple exponential divergence (step count is sufficient)")
    else:
        print("  --> INCONCLUSIVE (insufficient data)")

    # ---- Per-step cosine advantage trend ----
    print(f"\n  === Per-step cosine advantage over time ===")
    if all_cos_advs:
        n_cp = len(all_cos_advs[0])
        for ci in range(min(n_cp, len(CHECKPOINTS))):
            vals = [s[ci] for s in all_cos_advs if ci < len(s)]
            cp_step = CHECKPOINTS[ci] if ci < len(CHECKPOINTS) else ci * MEASURE_EVERY
            print(f"    Step ~{cp_step:>4}: mean cos advantage = {np.mean(vals):+.6f}  (std={np.std(vals):.6f})")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
