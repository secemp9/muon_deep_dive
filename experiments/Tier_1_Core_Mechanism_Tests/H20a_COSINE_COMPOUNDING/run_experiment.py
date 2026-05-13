#!/usr/bin/env python3
"""
H20a: Sampled Cosine-Compounding Check
======================================

Motivation:
  Prior toy experiments suggested that Muon can show a noticeably better loss
  trajectory than NormSGD while only showing a small Newton-alignment edge on
  sampled updates. This file tests a narrow, exploratory association question:
  does cumulative sampled Newton-alignment advantage track Muon's sampled loss
  advantage better than sampled update count alone?

Hypothesis under test:
  If sampled Newton alignment is a useful explanatory proxy in this toy setup,
  then log(loss_NormSGD / loss_Muon) should increase with cumulative sampled
  cosine advantage, and that cosine-based fit should outperform a step-only fit.

Protocol:
  - 4-layer deep linear 32x32 network trained for 500 updates.
  - Same data and initialization for Muon and NormSGD within each seed.
  - Learning rates chosen by a small sweep on the first seed.
  - Every 50 updates, measure:
      * the cosine between each optimizer's applied momentum update and an
        approximate Newton direction computed with CG + finite-difference HVPs;
      * the post-update loss ratio loss_NormSGD / loss_Muon.
  - Compare two linear fits:
      * log(loss ratio) vs cumulative sampled cosine advantage
      * log(loss ratio) vs sampled update count

This is an association test, not a proof of causality.
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 500
NUM_SEEDS = 5
BATCH_SIZE = 64
MOMENTUM = 0.9
NS_ITERS = 5

MEASURE_EVERY = 50
SAMPLE_STEPS = list(range(MEASURE_EVERY, NUM_STEPS + 1, MEASURE_EVERY))
CHECKPOINTS = SAMPLE_STEPS

LR_MUON_CANDIDATES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
LR_NORM_CANDIDATES = [0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]


# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else float('nan')


def finite_or_inf(value):
    value = float(value)
    return value if np.isfinite(value) and value < 1e10 else float('inf')


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
    """Finite-difference Hessian; kept as an optional diagnostic helper."""
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
# For 32x32, 4 layers: 4096 params. Building the full Hessian is too expensive
# for the default run. We therefore approximate the Newton direction via
# conjugate gradient with finite-difference Hessian-vector products.


def hessian_vec_product(theta, X, Y, v, eps=1e-5):
    """Finite-difference Hessian-vector product."""
    vn = np.linalg.norm(v)
    if vn < 1e-15:
        return np.zeros_like(v)
    tp = theta + eps * v
    tm = theta - eps * v
    gp, _ = grad_fn_flat(tp, X, Y)
    gm, _ = grad_fn_flat(tm, X, Y)
    return (gp - gm) / (2 * eps)


def cg_newton(theta, X, Y, g, max_iters=50, tol=1e-6, return_info=False):
    """Conjugate gradient approximation to the Newton direction H d = -g."""
    b = -g
    d = np.zeros_like(g)
    r = b.copy()
    p = r.copy()
    rsold = np.dot(r, r)
    residual_norm = np.sqrt(max(rsold, 0.0))
    iterations = 0

    if rsold < tol ** 2:
        info = {'iterations': 0, 'residual_norm': safe_float(residual_norm)}
        return (d, info) if return_info else d

    for iteration in range(1, max_iters + 1):
        Ap = hessian_vec_product(theta, X, Y, p)
        pAp = np.dot(p, Ap)
        iterations = iteration
        if abs(pAp) < 1e-15:
            residual_norm = np.sqrt(max(np.dot(r, r), 0.0))
            break
        alpha = rsold / pAp
        d = d + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        residual_norm = np.sqrt(max(rsnew, 0.0))
        if rsnew < tol ** 2:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    info = {'iterations': int(iterations), 'residual_norm': safe_float(residual_norm)}
    return (d, info) if return_info else d


# =============================================================================
# OPTIMIZER STEP HELPERS
# =============================================================================


def muon_step_directions(grads, momentum_buffers):
    step_dirs = []
    for l in range(N_LAYERS):
        momentum_buffers[l] = MOMENTUM * momentum_buffers[l] + newton_schulz(grads[l])
        step_dirs.append(momentum_buffers[l].copy())
    return step_dirs


def normsgd_step_directions(grads, momentum_buffers):
    step_dirs = []
    for l in range(N_LAYERS):
        nrm = np.linalg.norm(grads[l], 'fro')
        normed = grads[l] / max(nrm, 1e-15)
        momentum_buffers[l] = MOMENTUM * momentum_buffers[l] + normed
        step_dirs.append(momentum_buffers[l].copy())
    return step_dirs


def apply_step(weights, step_dirs, lr):
    for l in range(N_LAYERS):
        weights[l] = weights[l] - lr * step_dirs[l]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_muon_trajectory(weights_init, X, Y, lr):
    """Train with Muon, returning recorded pre-update losses and final weights."""
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
        step_dirs = muon_step_directions(grads, mom)
        apply_step(weights, step_dirs, lr)
    return losses, weights


def train_normsgd_trajectory(weights_init, X, Y, lr):
    """Train with NormSGD, returning recorded pre-update losses and final weights."""
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
        step_dirs = normsgd_step_directions(grads, mom)
        apply_step(weights, step_dirs, lr)
    return losses, weights


# =============================================================================
# TRAJECTORY WITH SAMPLED COSINE MEASUREMENTS
# =============================================================================


def run_trajectory_with_cosine(weights_init, X, Y, lr_muon, lr_norm):
    """
    Run Muon and NormSGD in parallel from the same init.

    Measurement semantics:
      - sample steps are the completed update counts in SAMPLE_STEPS;
      - cosine is measured on the sampled update direction about to be applied,
        using the approximate Newton direction at that pre-update state;
      - losses are recorded immediately after that sampled update is applied.
    """
    w_muon = [W.copy() for W in weights_init]
    w_norm = [W.copy() for W in weights_init]
    mom_muon = [np.zeros_like(W) for W in weights_init]
    mom_norm = [np.zeros_like(W) for W in weights_init]

    sampled_muon_cosine = []
    sampled_norm_cosine = []
    sampled_cos_advantage = []
    sampled_cumulative_cos = []
    sampled_loss_muon = []
    sampled_loss_norm = []
    sampled_loss_ratio = []
    sampled_log_loss_ratio = []
    muon_cg_iterations = []
    norm_cg_iterations = []
    muon_cg_residual_norm = []
    norm_cg_residual_norm = []

    cumulative_cos = 0.0

    for step_idx in range(NUM_STEPS):
        sample_step = step_idx + 1
        should_sample = sample_step in SAMPLE_STEPS

        pre_loss_muon = compute_loss(w_muon, X, Y)
        pre_loss_norm = compute_loss(w_norm, X, Y)

        muon_valid = np.isfinite(pre_loss_muon) and pre_loss_muon < 1e10
        norm_valid = np.isfinite(pre_loss_norm) and pre_loss_norm < 1e10

        grads_muon = compute_gradients(w_muon, X, Y) if muon_valid else None
        grads_norm = compute_gradients(w_norm, X, Y) if norm_valid else None
        muon_step_dirs = muon_step_directions(grads_muon, mom_muon) if muon_valid else None
        norm_step_dirs = normsgd_step_directions(grads_norm, mom_norm) if norm_valid else None

        cos_muon_newton = float('nan')
        cos_norm_newton = float('nan')
        muon_info = {'iterations': 0, 'residual_norm': float('nan')}
        norm_info = {'iterations': 0, 'residual_norm': float('nan')}

        if should_sample and muon_valid:
            theta_muon = pack(w_muon)
            g_muon_flat = pack(grads_muon)
            d_newton_muon, muon_info = cg_newton(
                theta_muon, X, Y, g_muon_flat, max_iters=30, return_info=True
            )
            muon_step_flat = pack(muon_step_dirs)
            cos_muon_newton = cosine(-muon_step_flat, d_newton_muon)

        if should_sample and norm_valid:
            theta_norm = pack(w_norm)
            g_norm_flat = pack(grads_norm)
            d_newton_norm, norm_info = cg_newton(
                theta_norm, X, Y, g_norm_flat, max_iters=30, return_info=True
            )
            norm_step_flat = pack(norm_step_dirs)
            cos_norm_newton = cosine(-norm_step_flat, d_newton_norm)

        if muon_valid:
            apply_step(w_muon, muon_step_dirs, lr_muon)
        if norm_valid:
            apply_step(w_norm, norm_step_dirs, lr_norm)

        post_loss_muon = finite_or_inf(compute_loss(w_muon, X, Y))
        post_loss_norm = finite_or_inf(compute_loss(w_norm, X, Y))

        if should_sample:
            if np.isfinite(cos_muon_newton) and np.isfinite(cos_norm_newton) and np.isfinite(cumulative_cos):
                cos_advantage = cos_muon_newton - cos_norm_newton
                cumulative_cos += cos_advantage
                current_cumulative_cos = cumulative_cos
            else:
                cos_advantage = float('nan')
                cumulative_cos = float('nan')
                current_cumulative_cos = float('nan')

            if np.isfinite(post_loss_muon) and np.isfinite(post_loss_norm) and post_loss_muon > 1e-30:
                loss_ratio = post_loss_norm / post_loss_muon
                log_loss_ratio = np.log(max(loss_ratio, 1e-30))
            else:
                loss_ratio = float('nan')
                log_loss_ratio = float('nan')

            sampled_muon_cosine.append(safe_float(cos_muon_newton))
            sampled_norm_cosine.append(safe_float(cos_norm_newton))
            sampled_cos_advantage.append(safe_float(cos_advantage))
            sampled_cumulative_cos.append(safe_float(current_cumulative_cos))
            sampled_loss_muon.append(safe_float(post_loss_muon))
            sampled_loss_norm.append(safe_float(post_loss_norm))
            sampled_loss_ratio.append(safe_float(loss_ratio))
            sampled_log_loss_ratio.append(safe_float(log_loss_ratio))
            muon_cg_iterations.append(int(muon_info['iterations']))
            norm_cg_iterations.append(int(norm_info['iterations']))
            muon_cg_residual_norm.append(safe_float(muon_info['residual_norm']))
            norm_cg_residual_norm.append(safe_float(norm_info['residual_norm']))

    final_loss_ratio = sampled_loss_ratio[-1] if sampled_loss_ratio else float('nan')
    final_cumulative_cos = sampled_cumulative_cos[-1] if sampled_cumulative_cos else float('nan')
    valid_sample_count = int(np.sum(np.isfinite(np.asarray(sampled_loss_ratio, dtype=float))))

    return {
        'sample_steps': SAMPLE_STEPS.copy(),
        'sample_schedule_note': (
            'Cosine is measured on sampled update directions at the pre-update '
            'state; losses are recorded immediately after those sampled updates.'
        ),
        'muon_newton_cosine': sampled_muon_cosine,
        'normsgd_newton_cosine': sampled_norm_cosine,
        'sampled_cosine_advantage': sampled_cos_advantage,
        'sampled_cumulative_cosine_advantage': sampled_cumulative_cos,
        'sampled_loss_muon': sampled_loss_muon,
        'sampled_loss_normsgd': sampled_loss_norm,
        'sampled_loss_ratio': sampled_loss_ratio,
        'sampled_log_loss_ratio': sampled_log_loss_ratio,
        'muon_cg_iterations': muon_cg_iterations,
        'normsgd_cg_iterations': norm_cg_iterations,
        'muon_cg_residual_norm': muon_cg_residual_norm,
        'normsgd_cg_residual_norm': norm_cg_residual_norm,
        'final_loss_ratio': safe_float(final_loss_ratio),
        'final_cumulative_cosine_advantage': safe_float(final_cumulative_cos),
        'valid_sample_count': valid_sample_count,
    }


# =============================================================================
# LR SWEEP
# =============================================================================


def sweep_lr(train_fn, weights_init, X, Y, candidates):
    best_lr, best_loss = candidates[0], float('inf')
    records = []
    for lr in candidates:
        losses, _ = train_fn([W.copy() for W in weights_init], X, Y, lr)
        final = losses[-1] if losses else float('inf')
        final = float(final)
        records.append({'lr': float(lr), 'final_loss': safe_float(final)})
        if np.isfinite(final) and final < best_loss:
            best_loss = final
            best_lr = lr
    return float(best_lr), records


# =============================================================================
# RESULT AGGREGATION
# =============================================================================


def summarize_values(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan'), float('nan'), 0
    return safe_float(np.mean(arr)), safe_float(np.std(arr)), int(arr.size)


def fit_linear_relation(x_values, y_values, x_name, y_name):
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x = x_arr[mask]
    y = y_arr[mask]

    result = {
        'valid': False,
        'x_name': x_name,
        'y_name': y_name,
        'n_points': int(x.size),
        'slope': float('nan'),
        'intercept': float('nan'),
        'r2': float('nan'),
        'x': x.tolist(),
        'y': y.tolist(),
        'predicted_y': [],
        'residuals': [],
    }

    if x.size < 3 or np.allclose(np.std(x), 0.0):
        return result

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    result.update({
        'valid': True,
        'slope': safe_float(slope),
        'intercept': safe_float(intercept),
        'r2': safe_float(r2),
        'predicted_y': predicted.tolist(),
        'residuals': (y - predicted).tolist(),
    })
    return result


def evaluate_verdict(cos_fit, step_fit):
    if not cos_fit['valid'] or not step_fit['valid']:
        return {
            'status': 'inconclusive',
            'supports_compounding': False,
            'cosine_slope_positive': None,
            'cosine_fit_beats_step_fit': None,
            'delta_r2': float('nan'),
            'message': 'Inconclusive: insufficient valid fit data.',
        }

    cosine_slope_positive = bool(cos_fit['slope'] > 0)
    cosine_fit_beats_step_fit = bool(cos_fit['r2'] > step_fit['r2'])
    delta_r2 = cos_fit['r2'] - step_fit['r2']

    if cosine_slope_positive and cosine_fit_beats_step_fit:
        status = 'supported'
        supports_compounding = True
        message = (
            'Supported under the current proxy criterion: the cosine-based fit '
            'has a positive slope and a higher R^2 than the step-only fit.'
        )
    elif not cosine_slope_positive:
        status = 'not_supported'
        supports_compounding = False
        message = (
            'Not supported by the current results: the fitted relationship '
            'between cumulative sampled cosine advantage and log(loss ratio) '
            'is non-positive.'
        )
    else:
        status = 'not_supported'
        supports_compounding = False
        message = (
            'Not supported by the current proxy criterion: the cosine-based fit '
            'does not outperform the step-only comparator.'
        )

    return {
        'status': status,
        'supports_compounding': supports_compounding,
        'cosine_slope_positive': cosine_slope_positive,
        'cosine_fit_beats_step_fit': cosine_fit_beats_step_fit,
        'delta_r2': safe_float(delta_r2),
        'message': message,
    }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


def run_experiment(verbose=True):
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    if verbose:
        print('=' * 100)
        print('H20a: SAMPLED COSINE-COMPOUNDING CHECK')
        print('=' * 100)
        print(f'Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}')
        print(f'Updates: {NUM_STEPS}, sampled updates: {SAMPLE_STEPS}')
        print(f'Seeds: {NUM_SEEDS}')
        print('Measurement note: cosine uses the applied sampled momentum update;')
        print('                 sampled losses are recorded immediately after that update.')
        print()

    if verbose:
        print('LR sweep on first seed...')
    X0, Y0 = make_data(seeds[0])
    w0 = init_weights(seeds[0] + 5000)
    best_lr_muon, muon_lr_records = sweep_lr(train_muon_trajectory, w0, X0, Y0, LR_MUON_CANDIDATES)
    best_lr_norm, norm_lr_records = sweep_lr(train_normsgd_trajectory, w0, X0, Y0, LR_NORM_CANDIDATES)

    if verbose:
        print(f'  Muon LR: {best_lr_muon}')
        print(f'  NormSGD LR: {best_lr_norm}')
        print()

    per_seed = []
    sample_rows = []

    for si, seed in enumerate(seeds):
        if verbose:
            print(f'  Seed {si + 1}/{NUM_SEEDS} (seed={seed})...')
        X, Y = make_data(seed)
        w_init = init_weights(seed + 5000)
        seed_result = run_trajectory_with_cosine(w_init, X, Y, best_lr_muon, best_lr_norm)
        seed_result['seed'] = int(seed)
        per_seed.append(seed_result)

        for idx, sample_step in enumerate(seed_result['sample_steps']):
            row = {
                'seed': int(seed),
                'sample_step': int(sample_step),
                'muon_newton_cosine': seed_result['muon_newton_cosine'][idx],
                'normsgd_newton_cosine': seed_result['normsgd_newton_cosine'][idx],
                'sampled_cosine_advantage': seed_result['sampled_cosine_advantage'][idx],
                'cumulative_sampled_cosine_advantage': seed_result['sampled_cumulative_cosine_advantage'][idx],
                'loss_muon': seed_result['sampled_loss_muon'][idx],
                'loss_normsgd': seed_result['sampled_loss_normsgd'][idx],
                'loss_ratio': seed_result['sampled_loss_ratio'][idx],
                'log_loss_ratio': seed_result['sampled_log_loss_ratio'][idx],
                'muon_cg_iterations': seed_result['muon_cg_iterations'][idx],
                'normsgd_cg_iterations': seed_result['normsgd_cg_iterations'][idx],
                'muon_cg_residual_norm': seed_result['muon_cg_residual_norm'][idx],
                'normsgd_cg_residual_norm': seed_result['normsgd_cg_residual_norm'][idx],
            }
            sample_rows.append(row)

        if verbose:
            print(
                f"    Final sampled loss_ratio={seed_result['final_loss_ratio']:.2f}x, "
                f"cumul_cos={seed_result['final_cumulative_cosine_advantage']:.4f}"
            )

    aggregate_by_sample = []
    for sample_step in SAMPLE_STEPS:
        rows = [row for row in sample_rows if row['sample_step'] == sample_step]
        mean_cos_adv, std_cos_adv, n_cos_adv = summarize_values([row['sampled_cosine_advantage'] for row in rows])
        mean_cumul, std_cumul, n_cumul = summarize_values(
            [row['cumulative_sampled_cosine_advantage'] for row in rows]
        )
        mean_ratio, std_ratio, n_ratio = summarize_values([row['loss_ratio'] for row in rows])
        mean_log_ratio, std_log_ratio, n_log_ratio = summarize_values([row['log_loss_ratio'] for row in rows])
        mean_muon_cos, std_muon_cos, _ = summarize_values([row['muon_newton_cosine'] for row in rows])
        mean_norm_cos, std_norm_cos, _ = summarize_values([row['normsgd_newton_cosine'] for row in rows])
        aggregate_by_sample.append({
            'sample_step': int(sample_step),
            'n_rows': int(len(rows)),
            'n_valid_cosine_advantage': int(n_cos_adv),
            'n_valid_cumulative_cosine': int(n_cumul),
            'n_valid_loss_ratio': int(n_ratio),
            'n_valid_log_loss_ratio': int(n_log_ratio),
            'mean_muon_newton_cosine': mean_muon_cos,
            'std_muon_newton_cosine': std_muon_cos,
            'mean_normsgd_newton_cosine': mean_norm_cos,
            'std_normsgd_newton_cosine': std_norm_cos,
            'mean_sampled_cosine_advantage': mean_cos_adv,
            'std_sampled_cosine_advantage': std_cos_adv,
            'mean_cumulative_sampled_cosine_advantage': mean_cumul,
            'std_cumulative_sampled_cosine_advantage': std_cumul,
            'mean_loss_ratio': mean_ratio,
            'std_loss_ratio': std_ratio,
            'mean_log_loss_ratio': mean_log_ratio,
            'std_log_loss_ratio': std_log_ratio,
        })

    fit_cos = fit_linear_relation(
        [row['cumulative_sampled_cosine_advantage'] for row in sample_rows],
        [row['log_loss_ratio'] for row in sample_rows],
        x_name='cumulative_sampled_cosine_advantage',
        y_name='log_loss_ratio',
    )
    fit_step = fit_linear_relation(
        [row['sample_step'] for row in sample_rows],
        [row['log_loss_ratio'] for row in sample_rows],
        x_name='sample_step',
        y_name='log_loss_ratio',
    )
    verdict = evaluate_verdict(fit_cos, fit_step)

    results = {
        'experiment_id': 'H20a_COSINE_COMPOUNDING',
        'question': (
            'Does cumulative sampled Newton-alignment advantage track Muon\'s '
            'sampled loss advantage over NormSGD better than sampled step count alone?'
        ),
        'script_dir': SCRIPT_DIR,
        'config': {
            'dim': DIM,
            'n_layers': N_LAYERS,
            'num_steps': NUM_STEPS,
            'num_seeds': NUM_SEEDS,
            'batch_size': BATCH_SIZE,
            'momentum': MOMENTUM,
            'ns_iters': NS_ITERS,
            'measure_every': MEASURE_EVERY,
            'sample_steps': SAMPLE_STEPS.copy(),
            'sample_count': len(SAMPLE_STEPS),
            'sample_schedule_note': (
                'Cosine is measured on sampled update directions at the pre-update '
                'state; losses are recorded immediately after those sampled updates.'
            ),
            'lr_muon_candidates': [float(x) for x in LR_MUON_CANDIDATES],
            'lr_normsgd_candidates': [float(x) for x in LR_NORM_CANDIDATES],
        },
        'seeds': [int(seed) for seed in seeds],
        'lr_sweeps': {
            'seed': int(seeds[0]),
            'muon': muon_lr_records,
            'normsgd': norm_lr_records,
            'best_lr_muon': float(best_lr_muon),
            'best_lr_normsgd': float(best_lr_norm),
        },
        'per_seed': per_seed,
        'sample_rows': sample_rows,
        'aggregate_by_sample': aggregate_by_sample,
        'fits': {
            'log_loss_ratio_vs_cumulative_cosine': fit_cos,
            'log_loss_ratio_vs_step': fit_step,
        },
        'verdict': verdict,
    }

    if verbose:
        print_results_summary(results)
    return results


# =============================================================================
# CONSOLE REPORTING
# =============================================================================


def print_results_summary(results):
    aggregate_rows = results['aggregate_by_sample']
    fit_cos = results['fits']['log_loss_ratio_vs_cumulative_cosine']
    fit_step = results['fits']['log_loss_ratio_vs_step']
    verdict = results['verdict']

    print(f"\n\n{'=' * 100}")
    print('RESULTS: SAMPLED COSINE-COMPOUNDING ANALYSIS')
    print(f"{'=' * 100}")
    print()
    print('  Aggregate sampled summary (means across available seeds)')
    print(f"  {'Step':>6}  {'Mean Cos Adv':>14}  {'Mean Cumul Cos':>16}  {'Mean Loss Ratio':>16}  {'Mean log(ratio)':>16}")
    print('  ' + '-' * 82)
    for row in aggregate_rows:
        print(
            f"  {row['sample_step']:>6}  "
            f"{row['mean_sampled_cosine_advantage']:>14.6f}  "
            f"{row['mean_cumulative_sampled_cosine_advantage']:>16.4f}  "
            f"{row['mean_loss_ratio']:>16.4f}x  "
            f"{row['mean_log_loss_ratio']:>16.4f}"
        )

    print('\n  === FIT A: log(loss_ratio) vs cumulative_sampled_cosine_advantage ===')
    if fit_cos['valid']:
        print(
            f"  Fit: log(ratio) = {fit_cos['slope']:.4f} * cumul_cos + {fit_cos['intercept']:.4f}"
        )
        print(f"  R^2 = {fit_cos['r2']:.4f}  (n={fit_cos['n_points']})")
        print(
            f"  Multiplicative interpretation: exp(slope) = {np.exp(fit_cos['slope']):.2f}x per unit cumul_cos"
        )
    else:
        print('  Not enough valid points for fit')

    print('\n  === FIT B: log(loss_ratio) vs sampled_update_count ===')
    if fit_step['valid']:
        print(f"  Fit: log(ratio) = {fit_step['slope']:.6f} * T + {fit_step['intercept']:.4f}")
        print(f"  R^2 = {fit_step['r2']:.4f}  (n={fit_step['n_points']})")
        print(
            f"  500-step extrapolation from fit: exp(rate*500) = {np.exp(fit_step['slope'] * 500):.2f}x"
        )
    else:
        print('  Not enough valid points for fit')

    print('\n  === VERDICT ===')
    print(f"  Cosine slope positive? {verdict['cosine_slope_positive']}")
    print(f"  Cosine fit beats step fit? {verdict['cosine_fit_beats_step_fit']}")
    print(f"  Delta R^2 (cos - step): {verdict['delta_r2']:.4f}")
    print(f"  Status: {verdict['status']}")
    print(f"  {verdict['message']}")

    print('\n  === Sampled cosine advantage by labeled sample step ===')
    for row in aggregate_rows:
        print(
            f"    Step {row['sample_step']:>4}: mean cos advantage = "
            f"{row['mean_sampled_cosine_advantage']:+.6f}  "
            f"(std={row['std_sampled_cosine_advantage']:.6f})"
        )

    print(f"\n{'=' * 100}")
    print('EXPERIMENT COMPLETE')
    print(f"{'=' * 100}")


# =============================================================================
# ENTRYPOINT
# =============================================================================


def main():
    return run_experiment(verbose=True)


if __name__ == '__main__':
    main()
