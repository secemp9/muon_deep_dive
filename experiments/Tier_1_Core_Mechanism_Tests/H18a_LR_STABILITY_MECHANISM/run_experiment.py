#!/usr/bin/env python3
"""
H18a: WHY Is Muon's Optimal LR Stable Across Depths?
=====================================================

FROM H6a: SGD's optimal LR drops ~100x going from depth 2 to depth 16.
Muon's drops only ~2x. This is the flagship practical claim. WHY?

HYPOTHESIS:
  ||ortho(G)||_op = 1 always (Axiom 0.5). So Muon's per-step weight change
  is ||eta * ortho(G)||_op = eta, bounded and depth-independent. SGD's
  per-step change is ||eta * G||_op = eta * ||G||_op, where ||G||_op grows
  with the product of per-layer singular values (depth-dependent).

KEY DISTINCTION:
  Max stable LR (divergence boundary) vs OPTIMAL LR (best final loss).
  H6a measured the OPTIMAL LR. We measure BOTH, plus the mechanism.

  For SGD: the optimal LR is determined by balancing convergence speed
  (wants large LR) against stability (limited by ||step||_op * curvature).
  Since ||step||_op = eta * ||G||_op grows with depth, the optimal eta
  must shrink with depth.

  For Muon: ||step||_op = eta * ||ortho(G)||_op = eta * 1. The step
  magnitude is depth-independent, so the optimal eta can stay constant
  -- it only needs to adapt to curvature changes, which are much weaker.

PROTOCOL:
  Three-phase measurement:
  Phase 1: Gradient/ortho norms and Hessian properties
  Phase 2: Full LR sweep for OPTIMAL LR (what H6a measured)
  Phase 3: Max stable LR via binary search (theoretical boundary)
  Phase 4: Mechanism verification (product law)

KEY TESTS:
  T1: ||ortho(G)||_op = 1.0 for all depths
  T2: ||G||_op grows with depth
  T3: SGD's OPTIMAL LR drops > 20x from depth 2 to 16
  T4: Muon's OPTIMAL LR drops < 5x from depth 2 to 16
  T5: SGD maxLR * ||G||_op is approximately constant (mechanism proof)
"""

import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURATION
# =============================================================================
DIM = 32
DEPTHS = [2, 4, 8, 16]
NS_ITERS = 5
BATCH_SIZE = 64
MOMENTUM = 0.9
NUM_SEEDS = 5
TRAIN_STEPS = 300      # for LR sweep
DIVERGENCE_STEPS = 100  # for stability binary search
DIVERGENCE_THRESHOLD = 1e6

# Dense LR grids for optimal LR sweep
SGD_LR_GRID = np.logspace(-5, 0, 30)
MUON_LR_GRID = np.logspace(-4, 0, 30)


# =============================================================================
# CORE: Newton-Schulz orthogonalization
# =============================================================================
def newton_schulz(M, n_iters=NS_ITERS):
    """Compute the orthogonal polar factor via Newton-Schulz iteration."""
    norm = np.linalg.norm(M, ord='fro')
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


# =============================================================================
# NETWORK OPERATIONS
# =============================================================================
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
    return 0.5 * np.mean(np.sum((pred - Y)**2, axis=0))


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


def make_data(seed):
    rng = np.random.RandomState(seed)
    W_target = rng.randn(DIM, DIM) * 0.5
    X = rng.randn(DIM, BATCH_SIZE) * 0.3
    Y = W_target @ X
    return X, Y


# =============================================================================
# TRAINING
# =============================================================================
def train(weights_init, X, Y, lr, optimizer, num_steps=TRAIN_STEPS):
    """Train and return final loss, plus step-1 update operator norms."""
    ws = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in ws]
    step1_op_norms = None

    for step in range(num_steps):
        loss = compute_loss(ws, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf'), None
        grads = compute_gradients(ws, X, Y)
        deltas = []
        for i in range(len(ws)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            delta = lr * mom[i]
            deltas.append(delta)
            ws[i] = ws[i] - delta

        if step == 0:
            step1_op_norms = [np.linalg.svd(d, compute_uv=False)[0] for d in deltas]

    return compute_loss(ws, X, Y), step1_op_norms


# =============================================================================
# MAX STABLE LR: binary search
# =============================================================================
def is_stable(weights_init, X, Y, lr, optimizer, steps=DIVERGENCE_STEPS):
    ws = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in ws]
    for step in range(steps):
        grads = compute_gradients(ws, X, Y)
        for i in range(len(ws)):
            if optimizer == 'muon':
                mom[i] = MOMENTUM * mom[i] + newton_schulz(grads[i])
            else:
                mom[i] = MOMENTUM * mom[i] + grads[i]
            ws[i] = ws[i] - lr * mom[i]
        loss = compute_loss(ws, X, Y)
        if not np.isfinite(loss) or loss > DIVERGENCE_THRESHOLD:
            return False
    return True


def find_max_stable_lr(weights_init, X, Y, optimizer, lr_low=1e-6, lr_high=10.0):
    while is_stable(weights_init, X, Y, lr_high, optimizer):
        lr_high *= 2
        if lr_high > 1e4:
            return lr_high
    while not is_stable(weights_init, X, Y, lr_low, optimizer):
        lr_low /= 2
        if lr_low < 1e-10:
            return 0.0
    for _ in range(25):
        lr_mid = np.sqrt(lr_low * lr_high)
        if lr_high / lr_low < 1.05:
            break
        if is_stable(weights_init, X, Y, lr_mid, optimizer):
            lr_low = lr_mid
        else:
            lr_high = lr_mid
    return np.sqrt(lr_low * lr_high)


# =============================================================================
# HESSIAN lambda_max
# =============================================================================
def flatten_weights(weights):
    return np.concatenate([W.ravel() for W in weights])


def unflatten_weights(flat, shapes):
    weights = []
    idx = 0
    for s in shapes:
        size = s[0] * s[1]
        weights.append(flat[idx:idx + size].reshape(s))
        idx += size
    return weights


def hessian_vector_product(weights, X, Y, v_flat, eps=1e-5):
    shapes = [W.shape for W in weights]
    w_flat = flatten_weights(weights)
    wp = unflatten_weights(w_flat + eps * v_flat, shapes)
    gp = flatten_weights(compute_gradients(wp, X, Y))
    wm = unflatten_weights(w_flat - eps * v_flat, shapes)
    gm = flatten_weights(compute_gradients(wm, X, Y))
    return (gp - gm) / (2 * eps)


def power_iteration_lambda_max(weights, X, Y, n_iters=20):
    dim = sum(W.size for W in weights)
    rng = np.random.RandomState(0)
    v = rng.randn(dim)
    v = v / np.linalg.norm(v)
    lam = 0
    for _ in range(n_iters):
        Hv = hessian_vector_product(weights, X, Y, v)
        lam = np.dot(v, Hv)
        nrm = np.linalg.norm(Hv)
        if nrm < 1e-15:
            break
        v = Hv / nrm
    return abs(lam)


# =============================================================================
# MAIN
# =============================================================================
def main():
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    print("=" * 115)
    print("H18a: WHY IS MUON'S OPTIMAL LR STABLE ACROSS DEPTHS?")
    print("=" * 115)
    print()
    print("HYPOTHESIS: ||ortho(G)||_op = 1 => Muon step magnitude = eta (constant).")
    print("SGD step magnitude = eta * ||G||_op (grows with depth).")
    print("So SGD's OPTIMAL LR must shrink with depth, while Muon's can stay constant.")
    print()
    print(f"Config: DIM={DIM}, depths={DEPTHS}, seeds={NUM_SEEDS}, train_steps={TRAIN_STEPS}")
    print()

    # =========================================================================
    # PHASE 1: Gradient properties at initialization
    # =========================================================================
    print(f"{'='*115}")
    print("PHASE 1: GRADIENT NORMS AND HESSIAN PROPERTIES AT INITIALIZATION")
    print(f"{'='*115}")

    grad_norms = {}
    ortho_norms = {}
    lambda_maxes = {}

    for depth in DEPTHS:
        grad_norms[depth] = []
        ortho_norms[depth] = []
        lambda_maxes[depth] = []

        for seed in seeds:
            X, Y = make_data(seed)
            w0 = init_weights(depth, seed + 5000)
            grads = compute_gradients(w0, X, Y)
            g_ops = [np.linalg.svd(g, compute_uv=False)[0] for g in grads]
            og_ops = [np.linalg.svd(newton_schulz(g), compute_uv=False)[0] for g in grads]
            grad_norms[depth].append(max(g_ops))
            ortho_norms[depth].append(max(og_ops))
            lam = power_iteration_lambda_max(w0, X, Y)
            lambda_maxes[depth].append(lam)

        print(f"  Depth {depth:>2}: ||G||_op = {np.mean(grad_norms[depth]):>10.4f} +/- {np.std(grad_norms[depth]):.4f}   "
              f"||oG||_op = {np.mean(ortho_norms[depth]):.6f} +/- {np.std(ortho_norms[depth]):.6f}   "
              f"lambda_max(H) = {np.mean(lambda_maxes[depth]):>10.2f} +/- {np.std(lambda_maxes[depth]):.2f}")

    # =========================================================================
    # PHASE 2: Optimal LR sweep (what H6a actually measured)
    # =========================================================================
    print(f"\n{'='*115}")
    print("PHASE 2: OPTIMAL LR SWEEP (300 steps, best final loss)")
    print(f"{'='*115}")

    sgd_opt_lrs = {}
    muon_opt_lrs = {}
    sgd_opt_losses = {}
    muon_opt_losses = {}
    sgd_step1_ops = {}
    muon_step1_ops = {}

    for depth in DEPTHS:
        print(f"\n  Depth L={depth}:")
        sgd_opt_lrs[depth] = []
        muon_opt_lrs[depth] = []
        sgd_opt_losses[depth] = []
        muon_opt_losses[depth] = []
        sgd_step1_ops[depth] = []
        muon_step1_ops[depth] = []

        for opt, lr_grid, opt_lrs, opt_losses, step1_ops in [
            ('sgd', SGD_LR_GRID, sgd_opt_lrs[depth], sgd_opt_losses[depth], sgd_step1_ops[depth]),
            ('muon', MUON_LR_GRID, muon_opt_lrs[depth], muon_opt_losses[depth], muon_step1_ops[depth]),
        ]:
            per_seed_best = []
            for seed in seeds:
                X, Y = make_data(seed)
                w0 = init_weights(depth, seed + 5000)
                best_lr = lr_grid[0]
                best_loss = float('inf')
                best_ops = None
                for lr in lr_grid:
                    fl, ops = train(w0, X, Y, lr, opt)
                    if np.isfinite(fl) and fl < best_loss:
                        best_loss = fl
                        best_lr = lr
                        best_ops = ops
                per_seed_best.append((best_lr, best_loss, best_ops))

            for lr, loss, ops in per_seed_best:
                opt_lrs.append(lr)
                opt_losses.append(loss)
                if ops is not None:
                    step1_ops.append(max(ops))

            mean_lr = np.mean(opt_lrs)
            median_lr = np.median(opt_lrs)
            mean_loss = np.mean(opt_losses)
            mean_op = np.mean(step1_ops) if step1_ops else float('nan')
            print(f"    {opt:>5}: optimal_LR = {median_lr:.6f} (median), {mean_lr:.6f} (mean)   "
                  f"loss = {mean_loss:.6e}   step1_||dW||_op = {mean_op:.4e}")

    # =========================================================================
    # PHASE 3: Max stable LR via binary search
    # =========================================================================
    print(f"\n{'='*115}")
    print("PHASE 3: MAX STABLE LR (divergence boundary, binary search)")
    print(f"{'='*115}")

    sgd_max_lrs = {}
    muon_max_lrs = {}

    for depth in DEPTHS:
        sgd_max_lrs[depth] = []
        muon_max_lrs[depth] = []

        for seed in seeds:
            X, Y = make_data(seed)
            w0 = init_weights(depth, seed + 5000)
            sgd_max_lrs[depth].append(find_max_stable_lr(w0, X, Y, 'sgd'))
            muon_max_lrs[depth].append(find_max_stable_lr(w0, X, Y, 'muon'))

        print(f"  Depth {depth:>2}: SGD max LR = {np.mean(sgd_max_lrs[depth]):>10.6f}   "
              f"Muon max LR = {np.mean(muon_max_lrs[depth]):>10.4f}")

    # =========================================================================
    # COMPLETE SUMMARY TABLE
    # =========================================================================
    print(f"\n\n{'='*130}")
    print("COMPLETE SUMMARY TABLE")
    print(f"{'='*130}")
    header = (f"{'Depth':>5} | {'||G||_op':>10} | {'||oG||_op':>9} | {'lam_max':>9} | "
              f"{'SGD optLR':>10} | {'Muon optLR':>10} | {'SGD maxLR':>10} | {'Muon maxLR':>10} | "
              f"{'SGDopt*||G||':>12} | {'SGDmax*||G||':>12} | "
              f"{'SGD drop':>8} | {'Muon drop':>9}")
    print(header)
    print("-" * 130)

    sgd_opt_d2 = np.median(sgd_opt_lrs[2])
    muon_opt_d2 = np.median(muon_opt_lrs[2])

    for depth in DEPTHS:
        g_op = np.mean(grad_norms[depth])
        og_op = np.mean(ortho_norms[depth])
        lam = np.mean(lambda_maxes[depth])
        sgd_opt = np.median(sgd_opt_lrs[depth])
        muon_opt = np.median(muon_opt_lrs[depth])
        sgd_max = np.mean(sgd_max_lrs[depth])
        muon_max = np.mean(muon_max_lrs[depth])
        sgd_opt_prod = sgd_opt * g_op
        sgd_max_prod = sgd_max * g_op
        sgd_drop = sgd_opt_d2 / sgd_opt
        muon_drop = muon_opt_d2 / muon_opt

        print(f"{depth:>5} | {g_op:>10.4f} | {og_op:>9.6f} | {lam:>9.2f} | "
              f"{sgd_opt:>10.6f} | {muon_opt:>10.6f} | {sgd_max:>10.6f} | {muon_max:>10.4f} | "
              f"{sgd_opt_prod:>12.4f} | {sgd_max_prod:>12.4f} | "
              f"{sgd_drop:>8.1f}x | {muon_drop:>9.1f}x")

    # =========================================================================
    # DEPTH SCALING: LOG-LOG FITS
    # =========================================================================
    print(f"\n\n{'='*115}")
    print("DEPTH SCALING: LOG-LOG FITS")
    print(f"{'='*115}")

    def log_log_fit(x, y):
        lx = np.log(np.array(x, dtype=float))
        ly = np.log(np.abs(np.array(y, dtype=float)) + 1e-15)
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = np.sum((ly - pred)**2)
        ss_tot = np.sum((ly - np.mean(ly))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0
        return slope, r2

    metrics_to_fit = [
        ('||G||_op', [np.mean(grad_norms[d]) for d in DEPTHS]),
        ('lambda_max(H)', [np.mean(lambda_maxes[d]) for d in DEPTHS]),
        ('SGD OPTIMAL LR', [np.median(sgd_opt_lrs[d]) for d in DEPTHS]),
        ('Muon OPTIMAL LR', [np.median(muon_opt_lrs[d]) for d in DEPTHS]),
        ('SGD MAX STABLE LR', [np.mean(sgd_max_lrs[d]) for d in DEPTHS]),
        ('Muon MAX STABLE LR', [np.mean(muon_max_lrs[d]) for d in DEPTHS]),
        ('SGD optLR * ||G||_op', [np.median(sgd_opt_lrs[d]) * np.mean(grad_norms[d]) for d in DEPTHS]),
        ('SGD maxLR * ||G||_op', [np.mean(sgd_max_lrs[d]) * np.mean(grad_norms[d]) for d in DEPTHS]),
    ]

    for name, vals in metrics_to_fit:
        slope, r2 = log_log_fit(DEPTHS, vals)
        ratio = vals[0] / vals[-1] if vals[-1] != 0 else float('inf')
        print(f"  {name:>25s}: slope={slope:>7.3f}  R^2={r2:.3f}  "
              f"d2/d16={ratio:>9.1f}x  ~O(L^{slope:.2f})")

    # =========================================================================
    # KEY TESTS
    # =========================================================================
    print(f"\n\n{'='*115}")
    print("KEY TESTS")
    print(f"{'='*115}")

    # T1: ||ortho(G)||_op = 1.0
    all_ortho = [n for d in DEPTHS for n in ortho_norms[d]]
    t1_max_dev = max(abs(n - 1.0) for n in all_ortho)
    t1_pass = t1_max_dev < 0.01
    print(f"\n  T1: ||ortho(G)||_op = 1.0 for all depths")
    print(f"      Max deviation from 1.0: {t1_max_dev:.6f}")
    print(f"      --> {'PASS' if t1_pass else 'FAIL'}")

    # T2: ||G||_op grows with depth
    g_d2 = np.mean(grad_norms[2])
    g_d16 = np.mean(grad_norms[16])
    t2_ratio = g_d16 / g_d2
    t2_pass = t2_ratio > 5.0
    print(f"\n  T2: ||G||_op grows with depth")
    print(f"      depth 2: {g_d2:.4f},  depth 16: {g_d16:.4f},  ratio: {t2_ratio:.1f}x")
    print(f"      --> {'PASS' if t2_pass else 'FAIL'}")

    # T3: SGD's OPTIMAL LR drops > 20x
    sgd_opt_2 = np.median(sgd_opt_lrs[2])
    sgd_opt_16 = np.median(sgd_opt_lrs[16])
    t3_ratio = sgd_opt_2 / sgd_opt_16
    t3_pass = t3_ratio > 20.0
    print(f"\n  T3: SGD's optimal LR drops > 20x from depth 2 to 16")
    print(f"      depth 2: {sgd_opt_2:.6f},  depth 16: {sgd_opt_16:.6f},  ratio: {t3_ratio:.1f}x")
    print(f"      --> {'PASS' if t3_pass else 'FAIL'}")

    # T4: Muon's OPTIMAL LR drops < 5x
    muon_opt_2 = np.median(muon_opt_lrs[2])
    muon_opt_16 = np.median(muon_opt_lrs[16])
    t4_ratio = muon_opt_2 / muon_opt_16
    t4_pass = t4_ratio < 5.0
    print(f"\n  T4: Muon's optimal LR drops < 5x from depth 2 to 16")
    print(f"      depth 2: {muon_opt_2:.6f},  depth 16: {muon_opt_16:.6f},  ratio: {t4_ratio:.1f}x")
    print(f"      --> {'PASS' if t4_pass else 'FAIL'}")

    # T5: SGD max_LR * ||G||_op ~ constant
    max_products = [np.mean(sgd_max_lrs[d]) * np.mean(grad_norms[d]) for d in DEPTHS]
    t5_cv = np.std(max_products) / np.mean(max_products)
    t5_pass = t5_cv < 0.5
    print(f"\n  T5: SGD maxLR * ||G||_op is approximately constant (mechanism proof)")
    print(f"      Products: {[f'{p:.4f}' for p in max_products]}")
    print(f"      CV = {t5_cv:.3f}  (need < 0.5)")
    print(f"      --> {'PASS' if t5_pass else 'FAIL'}")

    # =========================================================================
    # MECHANISM SUMMARY
    # =========================================================================
    sgd_opt_range = sgd_opt_2 / sgd_opt_16
    muon_opt_range = muon_opt_2 / muon_opt_16
    advantage = sgd_opt_range / muon_opt_range if muon_opt_range > 0 else float('inf')

    print(f"\n\n{'='*115}")
    print("MECHANISM SUMMARY")
    print(f"{'='*115}")
    print()
    print(f"  SGD optimal LR drops  {sgd_opt_range:>6.1f}x  (depth 2 -> 16)")
    print(f"  Muon optimal LR drops {muon_opt_range:>6.1f}x  (depth 2 -> 16)")
    print(f"  Muon advantage:       {advantage:>6.1f}x  more stable")
    print(f"  (H6a reported: SGD ~100x, Muon ~2x, advantage ~50x)")
    print()
    print(f"  WHY?")
    print(f"  ||G||_op grows {t2_ratio:.0f}x from depth 2 to 16.")
    print(f"  SGD step = eta * G, so ||step||_op = eta * ||G||_op = eta * {g_d16:.0f} at depth 16.")
    print(f"  Muon step = eta * ortho(G), so ||step||_op = eta * 1.0 at ALL depths.")
    print(f"  The optimal LR must shrink when the step magnitude grows (to avoid overshooting).")
    print(f"  SGD's step magnitude grows {t2_ratio:.0f}x => its optimal LR shrinks {sgd_opt_range:.0f}x.")
    print(f"  Muon's step magnitude is constant => its optimal LR barely changes ({muon_opt_range:.1f}x).")
    print()
    print(f"  Verification: SGD_maxLR * ||G||_op = constant across depths (CV={t5_cv:.3f})")
    print(f"  This directly proves that ||G||_op is the scaling factor for SGD's LR sensitivity.")

    # =========================================================================
    # OVERALL VERDICT
    # =========================================================================
    all_pass = t1_pass and t2_pass and t3_pass and t4_pass and t5_pass
    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass])

    print(f"\n\n{'='*115}")
    print(f"OVERALL VERDICT: {n_pass}/5 tests passed")
    print(f"{'='*115}")
    if all_pass:
        print()
        print("ALL TESTS PASSED.")
        print()
        print("ANSWER TO 'WHY is Muon's optimal LR stable across depths?':")
        print()
        print("  Because ortho(G) has ||.||_op = 1 at every depth (by construction),")
        print("  Muon's step magnitude is eta, independent of depth.")
        print("  SGD's step magnitude is eta * ||G||_op, which grows as O(L^{:.1f}).".format(
            log_log_fit(DEPTHS, [np.mean(grad_norms[d]) for d in DEPTHS])[0]))
        print("  The optimal LR must compensate for step magnitude, so SGD's optimal")
        print("  LR drops as O(L^{:.1f}) while Muon's stays nearly constant.".format(
            log_log_fit(DEPTHS, [np.median(sgd_opt_lrs[d]) for d in DEPTHS])[0]))
    else:
        failed = []
        if not t1_pass: failed.append("T1 (ortho norm)")
        if not t2_pass: failed.append("T2 (grad norm growth)")
        if not t3_pass: failed.append("T3 (SGD optimal LR drops)")
        if not t4_pass: failed.append("T4 (Muon optimal LR stable)")
        if not t5_pass: failed.append("T5 (product constant)")
        print(f"\n  FAILED: {', '.join(failed)}")
        if not t4_pass and t1_pass and t2_pass:
            print()
            print("  Note: T4 failure means Muon's optimal LR also varies with depth")
            print("  more than expected. This could be because:")
            print("  a) The Frobenius norm of ortho(G) grows with depth (more layers)")
            print("  b) The Hessian curvature affects Muon too, just less than SGD")
            print("  c) The optimal LR is determined by more than just step magnitude")

    print(f"\n{'='*115}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*115}")

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
