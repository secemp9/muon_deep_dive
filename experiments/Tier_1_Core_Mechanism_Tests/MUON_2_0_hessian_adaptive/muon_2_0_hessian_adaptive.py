#!/usr/bin/env python3
"""
Experiment 3.1: Hessian-Adaptive Muon 2.0

Toy setting:
- 3-layer deep linear network with 4x4 layers (48 parameters total)
- ill-conditioned target singular values [100, 10, 1, 0.1]
- 500 optimization steps, 5 seeds

Scientific question:
- Can a small k-step Lanczos Ritz subspace close some of the gap between
  standard Muon and an exact-Hessian democratic update, while staying much
  cheaper than the full oracle construction?

Important cost caveat:
- The counted "matmul cost" in this file is only a proxy over operations routed
  through counted_matmul.
- It includes optimizer-path forward/gradient work, Newton-Schulz matmuls, and
  counted finite-difference HVP paths.
- It excludes exact full-Hessian construction overhead in FullDemocratic,
  eigendecompositions/SVD/QR, and other non-counted linear algebra.
- Treat recovery/Pareto outputs as honest proxy-level summaries for this toy
  experiment, not as wall-clock or full-FLOP truth.

Notebook counterpart:
- experiments/Tier_1_Core_Mechanism_Tests/MUON_2_0_hessian_adaptive/run_experiment.ipynb
"""

import time
import numpy as np

EXPERIMENT_NAME = "Experiment 3.1: Hessian-Adaptive Muon 2.0"
COUNTERPART_NOTEBOOK_PATH = (
    "experiments/Tier_1_Core_Mechanism_Tests/"
    "MUON_2_0_hessian_adaptive/run_experiment.ipynb"
)

# ── fixed network / problem structure ──────────────────────────────────────
DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM
EPS_FD = 1e-5

# ── default experiment schedule ────────────────────────────────────────────
N_STEPS = 500
HESSIAN_RECOMPUTE_EVERY = 50
N_SEEDS = 5
DEFAULT_LR_CANDIDATES = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

METHOD_NAMES = [
    "SGD",
    "Muon",
    "FullDemocratic",
    "Muon2_L10",
    "Muon2_L5",
    "Muon+Hrescale",
]
DISPLAY_NAMES = {
    "SGD": "SGD (baseline)",
    "Muon": "Muon (polar, k=5 NS)",
    "FullDemocratic": "Full Democratic SGD (oracle)",
    "Muon2_L10": "Muon 2.0 (Lanczos-10 Ritz)",
    "Muon2_L5": "Muon 2.0 (Lanczos-5 Ritz)",
    "Muon+Hrescale": "Muon + Hessian rescale",
}
METHOD_PROXY_NOTES = {
    "SGD": "Counted proxy includes per-step gradient work only; monitoring losses are excluded.",
    "Muon": "Counted proxy includes SGD gradient work plus Newton-Schulz matmuls; SVD is not used in the optimizer path.",
    "FullDemocratic": "Oracle reference: counted proxy excludes exact full-Hessian construction and eigendecomposition overhead.",
    "Muon2_L10": "Counted proxy includes SGD gradient work plus 10 counted HVPs every Hessian refresh (Lanczos Ritz subspace).",
    "Muon2_L5": "Counted proxy includes SGD gradient work plus 5 counted HVPs every Hessian refresh (Lanczos Ritz subspace).",
    "Muon+Hrescale": "Counted proxy includes SGD gradient work plus per-layer counted HVPs for curvature rescaling; SVD overhead is excluded.",
}
COST_PROXY_CAVEAT = (
    "Counted matmul totals are a proxy, not a full compute model: they include only "
    "operations routed through counted_matmul in the optimizer path, and exclude exact "
    "full-Hessian construction in FullDemocratic plus decompositions such as eig/SVD/QR."
)
TRAJECTORY_SEMANTICS = (
    "loss_curve[0] is the initial loss before updates; loss_curve[t] for t>=1 is the "
    "post-update loss after t optimization steps."
)

# ── counted matmul proxy ───────────────────────────────────────────────────
_matmul_count = 0


def counted_matmul(A, B):
    """Matrix multiply with global counting."""
    global _matmul_count
    _matmul_count += 1
    return A @ B


def reset_matmul_count():
    global _matmul_count
    _matmul_count = 0


def get_matmul_count():
    return _matmul_count


# ── helpers ────────────────────────────────────────────────────────────────
def get_default_config():
    """Return a fresh default config dictionary for the experiment runner."""
    return {
        "dim": DIM,
        "n_layers": N_LAYERS,
        "n_params": N_PARAMS,
        "eps_fd": EPS_FD,
        "n_steps": N_STEPS,
        "hessian_recompute_every": HESSIAN_RECOMPUTE_EVERY,
        "n_seeds": N_SEEDS,
        "seed_base": 42,
        "seed_stride": 7,
        "target_singular_values": [100.0, 10.0, 1.0, 0.1],
        "init_scale": 0.3,
        "lr_candidates": list(DEFAULT_LR_CANDIDATES),
        "methods": list(METHOD_NAMES),
        "divergence_ceiling": 1e8,
    }


def resolve_config(config=None):
    """Merge user config with defaults and validate supported overrides."""
    cfg = get_default_config()
    if config is not None:
        for key, value in config.items():
            if key in ("dim", "n_layers", "n_params", "eps_fd") and value != cfg[key]:
                raise ValueError(
                    f"{key} is fixed in this experiment implementation; expected {cfg[key]!r}, got {value!r}."
                )
            cfg[key] = value

    cfg["n_steps"] = int(cfg["n_steps"])
    cfg["hessian_recompute_every"] = int(cfg["hessian_recompute_every"])
    cfg["n_seeds"] = int(cfg["n_seeds"])
    cfg["seed_base"] = int(cfg["seed_base"])
    cfg["seed_stride"] = int(cfg["seed_stride"])
    cfg["init_scale"] = float(cfg["init_scale"])
    cfg["divergence_ceiling"] = float(cfg["divergence_ceiling"])
    cfg["lr_candidates"] = [float(x) for x in cfg["lr_candidates"]]
    cfg["target_singular_values"] = [float(x) for x in cfg["target_singular_values"]]
    cfg["methods"] = list(cfg["methods"])

    if cfg["n_steps"] <= 0:
        raise ValueError("n_steps must be positive")
    if cfg["hessian_recompute_every"] <= 0:
        raise ValueError("hessian_recompute_every must be positive")
    if cfg["n_seeds"] <= 0:
        raise ValueError("n_seeds must be positive")
    if not cfg["lr_candidates"]:
        raise ValueError("lr_candidates must be non-empty")
    if len(cfg["target_singular_values"]) != DIM:
        raise ValueError(f"target_singular_values must have length {DIM}")
    unknown = [m for m in cfg["methods"] if m not in METHOD_NAMES]
    if unknown:
        raise ValueError(f"Unknown methods requested: {unknown}")
    return cfg


def seeds_from_config(config):
    return [config["seed_base"] + i * config["seed_stride"] for i in range(config["n_seeds"])]


def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta):
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM * DIM].reshape(DIM, DIM))
        idx += DIM * DIM
    return Ws


def flat_to_matrices(flat_vec):
    mats = []
    for k in range(N_LAYERS):
        start = k * DIM * DIM
        stop = (k + 1) * DIM * DIM
        mats.append(flat_vec[start:stop].reshape(DIM, DIM))
    return mats


def forward(Ws, count=True):
    mm = counted_matmul if count else (lambda a, b: a @ b)
    out = Ws[0]
    for W in Ws[1:]:
        out = mm(W, out)
    return out


def loss_fn(theta, target, count=True):
    Ws = unpack(theta)
    diff = forward(Ws, count=count) - target
    return 0.5 * np.sum(diff ** 2)


def grad_fn(theta, target, count=True):
    """Gradient of the deep-linear squared loss with respect to flat theta."""
    Ws = unpack(theta)
    mm = counted_matmul if count else (lambda a, b: a @ b)
    prod = forward(Ws, count=count)
    residual = prod - target

    grads = []
    for k in range(N_LAYERS):
        L = np.eye(DIM)
        for j in range(k + 1, N_LAYERS):
            L = mm(Ws[j], L)
        Rp = np.eye(DIM)
        for j in range(0, k):
            Rp = mm(Ws[j], Rp)
        dWk = mm(L.T, mm(residual, Rp.T))
        grads.append(dWk.ravel())
    return np.concatenate(grads)


def grad_matrices(theta, target, count=True):
    return flat_to_matrices(grad_fn(theta, target, count=count))


# ── Hessian and HVPs ───────────────────────────────────────────────────────
def hessian_fn(theta, target):
    """Exact full Hessian via finite differences.

    This is intentionally uncounted overhead and serves as an oracle reference.
    """
    n = len(theta)
    H = np.zeros((n, n))
    for i in range(n):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += EPS_FD
        theta_m[i] -= EPS_FD
        g_p = grad_fn(theta_p, target, count=False)
        g_m = grad_fn(theta_m, target, count=False)
        H[:, i] = (g_p - g_m) / (2 * EPS_FD)
    return 0.5 * (H + H.T)


def hvp(theta, target, v, count=True):
    """Finite-difference Hessian-vector product.

    When count=True, the two gradient evaluations contributing to this HVP route
    through counted_matmul and therefore contribute to the proxy cost.
    """
    theta_p = theta + EPS_FD * v
    theta_m = theta - EPS_FD * v
    g_p = grad_fn(theta_p, target, count=count)
    g_m = grad_fn(theta_m, target, count=count)
    return (g_p - g_m) / (2 * EPS_FD)


def lanczos(theta, target, k, count=True):
    """Run k steps of Lanczos and return the k-dimensional Ritz sketch.

    Returns
    -------
    ritz_vals : ndarray shape (k,)
        Ritz values from the tridiagonal Lanczos matrix.
    ritz_vecs : ndarray shape (n, k)
        Ritz vectors mapped back to the full parameter space.

    Notes
    -----
    This is a k-step Lanczos Ritz subspace, not an explicit "top + bottom"
    eigenpair selection. In this small experiment the full k-dimensional sketch
    is used for equalization.
    """
    n = len(theta)
    rng = np.random.RandomState(int(np.abs(theta[:4]).sum() * 1000) % (2 ** 31))
    q = rng.randn(n)
    q = q / np.linalg.norm(q)

    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)
    Q[:, 0] = q

    for j in range(k):
        v = hvp(theta, target, Q[:, j], count=count)
        alpha[j] = Q[:, j] @ v
        if j == 0:
            v = v - alpha[j] * Q[:, j]
        else:
            v = v - alpha[j] * Q[:, j] - beta[j] * Q[:, j - 1]

        for jj in range(j + 1):
            v = v - (Q[:, jj] @ v) * Q[:, jj]

        beta_next = np.linalg.norm(v)
        if beta_next < 1e-12:
            break
        if j + 1 < k:
            beta[j + 1] = beta_next
            Q[:, j + 1] = v / beta_next

    T = np.diag(alpha)
    for j in range(k - 1):
        T[j, j + 1] = beta[j + 1]
        T[j + 1, j] = beta[j + 1]

    ritz_vals, ritz_vecs_small = np.linalg.eigh(T)
    ritz_vecs = Q @ ritz_vecs_small
    for i in range(ritz_vecs.shape[1]):
        norm = np.linalg.norm(ritz_vecs[:, i])
        if norm > 1e-12:
            ritz_vecs[:, i] /= norm
    return ritz_vals, ritz_vecs


# ── optimizer directions ───────────────────────────────────────────────────
def polar_factor_ns(M, k=5, count=True):
    """Polar factor via k Newton-Schulz iterations."""
    mm = counted_matmul if count else (lambda a, b: a @ b)
    a = np.linalg.norm(M, "fro")
    if a < 1e-12:
        return M.copy()
    X = M / a
    for _ in range(k):
        X = 1.5 * X - 0.5 * mm(X, mm(X.T, X))
    return X


def polar_factor_svd(M, count=True):
    """Exact polar factor via SVD, kept for diagnostics/reference."""
    U, _, Vt = np.linalg.svd(M, full_matrices=True)
    mm = counted_matmul if count else (lambda a, b: a @ b)
    return mm(U, Vt)


def muon_direction_from_grad_mats(gmats, count=True):
    polars = [polar_factor_ns(gm, k=5, count=count).ravel() for gm in gmats]
    return np.concatenate(polars)


def muon_direction(theta, target, gmats=None, count=True):
    if gmats is None:
        gmats = grad_matrices(theta, target, count=count)
    return muon_direction_from_grad_mats(gmats, count=count)


def democratic_direction(grad_vec, eigvecs):
    projs = eigvecs.T @ grad_vec
    signs = np.sign(projs)
    magnitudes = np.abs(projs)
    mean_mag = np.mean(magnitudes) if np.max(magnitudes) > 1e-30 else 0.0
    eq_projs = signs * mean_mag
    return eigvecs @ eq_projs


def muon2_lanczos_direction(grad_vec, ritz_vecs):
    """Equalize within the Lanczos Ritz subspace and keep SGD in the complement."""
    projs = ritz_vecs.T @ grad_vec
    signs = np.sign(projs)
    magnitudes = np.abs(projs)
    mean_mag = np.mean(magnitudes) if np.max(magnitudes) > 1e-30 else 0.0
    eq_projs = signs * mean_mag
    equalized_part = ritz_vecs @ eq_projs
    grad_in_subspace = ritz_vecs @ projs
    complement = grad_vec - grad_in_subspace
    return equalized_part + complement


def muon_hessian_rescale_direction(theta, target, gmats=None, count=True):
    """Muon direction with per-singular-direction Hessian curvature rescaling.

    SVD and other dense linear algebra are intentionally not part of the counted
    proxy. The counted portion comes from the HVP calls when count=True.
    """
    if gmats is None:
        gmats = grad_matrices(theta, target, count=count)
    directions = []

    for layer_idx, gm in enumerate(gmats):
        U, _, Vt = np.linalg.svd(gm, full_matrices=True)
        curvatures = np.zeros(DIM)
        offset = layer_idx * DIM * DIM
        for i in range(DIM):
            direction_mat = np.outer(U[:, i], Vt[i, :])
            v_full = np.zeros(N_PARAMS)
            v_full[offset:offset + DIM * DIM] = direction_mat.ravel()
            Hv = hvp(theta, target, v_full, count=count)
            curvatures[i] = np.abs(v_full @ Hv) + 1e-8

        rescaled = np.zeros((DIM, DIM))
        for i in range(DIM):
            weight = 1.0 / np.sqrt(curvatures[i])
            rescaled += weight * np.outer(U[:, i], Vt[i, :])
        directions.append(rescaled.ravel())

    return np.concatenate(directions)


def match_reference_norm(direction, reference):
    ref_norm = np.linalg.norm(reference)
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 1e-12:
        return direction * (ref_norm / dir_norm)
    return direction


# ── execution helpers ──────────────────────────────────────────────────────
def make_seed_problem(seed, config=None):
    cfg = resolve_config(config)
    rng_init = np.random.RandomState(seed)
    U_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    V_t, _ = np.linalg.qr(rng_init.randn(DIM, DIM))
    sigma_t = np.array(cfg["target_singular_values"])
    target = U_t @ np.diag(sigma_t) @ V_t
    theta0 = cfg["init_scale"] * rng_init.randn(N_PARAMS)
    return {
        "seed": int(seed),
        "theta0": theta0,
        "target": target,
        "target_singular_values": sigma_t.tolist(),
        "target_condition_number": float(np.linalg.cond(target)),
    }


def _refresh_method_state(method, theta, target, step, recompute_every, count, state):
    if step % recompute_every != 0:
        return state
    if method == "FullDemocratic":
        H = hessian_fn(theta, target)
        _, state["H_eigvecs"] = np.linalg.eigh(H)
    elif method == "Muon2_L10":
        state["ritz_vals"], state["ritz_vecs"] = lanczos(theta, target, 10, count=count)
    elif method == "Muon2_L5":
        state["ritz_vals"], state["ritz_vecs"] = lanczos(theta, target, 5, count=count)
    return state


def _step_direction(method, theta, target, grad_vec, grad_mats_current, state, count):
    if method == "SGD":
        return grad_vec
    if method == "Muon":
        return muon_direction(theta, target, gmats=grad_mats_current, count=count)
    if method == "FullDemocratic":
        return match_reference_norm(democratic_direction(grad_vec, state["H_eigvecs"]), grad_vec)
    if method in ("Muon2_L10", "Muon2_L5"):
        return match_reference_norm(muon2_lanczos_direction(grad_vec, state["ritz_vecs"]), grad_vec)
    if method == "Muon+Hrescale":
        d = muon_hessian_rescale_direction(theta, target, gmats=grad_mats_current, count=count)
        return match_reference_norm(d, grad_vec)
    raise ValueError(f"Unknown method: {method}")


def _run_optimizer_path(method, lr, theta0, target, config=None, count=False, record_trajectory=False):
    cfg = resolve_config(config)
    if method not in cfg["methods"] and method not in METHOD_NAMES:
        raise ValueError(f"Unknown method: {method}")

    if count:
        reset_matmul_count()

    theta = theta0.copy()
    state = {"H_eigvecs": None, "ritz_vals": None, "ritz_vecs": None}
    current_loss = float(loss_fn(theta, target, count=False))
    losses = [current_loss] if record_trajectory else None

    for step in range(cfg["n_steps"]):
        grad_vec = grad_fn(theta, target, count=count)
        grad_mats_current = flat_to_matrices(grad_vec)
        state = _refresh_method_state(
            method,
            theta,
            target,
            step,
            cfg["hessian_recompute_every"],
            count,
            state,
        )
        direction = _step_direction(method, theta, target, grad_vec, grad_mats_current, state, count)
        theta -= lr * direction

        current_loss = float(loss_fn(theta, target, count=False))
        if not np.isfinite(current_loss) or current_loss > cfg["divergence_ceiling"]:
            current_loss = cfg["divergence_ceiling"]
            if record_trajectory:
                losses.append(current_loss)
                while len(losses) < cfg["n_steps"] + 1:
                    losses.append(current_loss)
            break
        if record_trajectory:
            losses.append(current_loss)

    if record_trajectory and len(losses) < cfg["n_steps"] + 1:
        while len(losses) < cfg["n_steps"] + 1:
            losses.append(current_loss)

    matmul_proxy = int(get_matmul_count()) if count else None
    return {
        "final_loss": float(current_loss),
        "loss_curve": np.array(losses, dtype=float) if record_trajectory else None,
        "matmul_proxy": matmul_proxy,
    }


def run_single(method, lr, theta0, target, config=None):
    """Run one uncounted training path and return the terminal post-update loss."""
    return _run_optimizer_path(method, lr, theta0, target, config=config, count=False, record_trajectory=False)["final_loss"]


def run_full_counted(method, lr, theta0, target, config=None):
    """Run one counted training path and return full trajectory + proxy cost."""
    out = _run_optimizer_path(method, lr, theta0, target, config=config, count=True, record_trajectory=True)
    return out["loss_curve"], out["final_loss"], out["matmul_proxy"]


def sweep_best_lrs(theta0, target, methods=None, config=None):
    cfg = resolve_config(config)
    methods = cfg["methods"] if methods is None else list(methods)
    best_lrs = {}
    sweep_records = {m: [] for m in methods}

    for method in methods:
        best_loss = np.inf
        best_lr = cfg["lr_candidates"][0]
        for lr in cfg["lr_candidates"]:
            final_loss = run_single(method, lr, theta0, target, config=cfg)
            sweep_records[method].append({"lr": float(lr), "final_loss": float(final_loss)})
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
        best_lrs[method] = float(best_lr)
    return best_lrs, sweep_records


def _summary_rows_from_aggregate(aggregate, methods):
    rows = []
    for method in methods:
        stats = aggregate["method_stats"][method]
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "mean_final_loss": stats["mean_loss"],
                "std_final_loss": stats["std_loss"],
                "mean_matmul_proxy": stats["mean_matmul_proxy"],
                "cost_ratio_vs_sgd": stats["cost_ratio"],
                "mean_recovery_pct": stats["mean_recovery"],
                "std_recovery_pct": stats["std_recovery"],
                "pareto_score": stats["pareto_score"],
                "proxy_note": METHOD_PROXY_NOTES[method],
            }
        )
    return rows


def _threshold_rows_from_aggregate(aggregate):
    rows = []
    for test_id in ("T1", "T2", "T3", "T4"):
        test = aggregate["threshold_tests"][test_id]
        rows.append(
            {
                "test_id": test_id,
                "question": test["question"],
                "metric": test["metric"],
                "reference": test["reference"],
                "pass": bool(test["pass"]),
            }
        )
    return rows


def aggregate_results(seed_results, config):
    methods = config["methods"]
    n_steps = config["n_steps"]
    method_stats = {}

    for method in methods:
        final_losses = np.array([seed_record["per_method"][method]["final_loss"] for seed_record in seed_results], dtype=float)
        matmuls = np.array([seed_record["per_method"][method]["matmul_proxy"] for seed_record in seed_results], dtype=float)
        curves = np.array([seed_record["per_method"][method]["loss_curve"] for seed_record in seed_results], dtype=float)
        best_lrs = np.array([seed_record["best_lrs"][method] for seed_record in seed_results], dtype=float)

        method_stats[method] = {
            "display_name": DISPLAY_NAMES[method],
            "mean_loss": float(np.mean(final_losses)),
            "std_loss": float(np.std(final_losses)),
            "median_loss": float(np.median(final_losses)),
            "mean_matmul_proxy": float(np.mean(matmuls)),
            "std_matmul_proxy": float(np.std(matmuls)),
            "final_losses": final_losses.tolist(),
            "matmul_proxies": matmuls.tolist(),
            "best_lrs": best_lrs.tolist(),
            "loss_curve_mean": np.mean(curves, axis=0).tolist(),
            "loss_curve_std": np.std(curves, axis=0).tolist(),
            "loss_curves": [curve.tolist() for curve in curves],
        }

    recoveries = {m: [] for m in methods}
    for seed_record in seed_results:
        sgd_loss = seed_record["per_method"]["SGD"]["final_loss"]
        dem_loss = seed_record["per_method"]["FullDemocratic"]["final_loss"]
        gap = sgd_loss - dem_loss
        for method in methods:
            method_loss = seed_record["per_method"][method]["final_loss"]
            rec = 0.0 if abs(gap) <= 1e-12 else (sgd_loss - method_loss) / gap * 100.0
            recoveries[method].append(float(rec))

    sgd_proxy = method_stats["SGD"]["mean_matmul_proxy"]
    for method in methods:
        method_stats[method]["recoveries"] = recoveries[method]
        method_stats[method]["mean_recovery"] = float(np.mean(recoveries[method]))
        method_stats[method]["std_recovery"] = float(np.std(recoveries[method]))
        cost_ratio = method_stats[method]["mean_matmul_proxy"] / sgd_proxy if sgd_proxy > 0 else np.nan
        method_stats[method]["cost_ratio"] = float(cost_ratio)
        method_stats[method]["pareto_score"] = (
            float(method_stats[method]["mean_recovery"] / cost_ratio)
            if np.isfinite(cost_ratio) and cost_ratio > 0
            else 0.0
        )

    sorted_methods = sorted(methods, key=lambda m: method_stats[m]["cost_ratio"])
    pareto_frontier = []
    best_recovery_so_far = -np.inf
    for method in sorted_methods:
        recovery = method_stats[method]["mean_recovery"]
        if recovery > best_recovery_so_far:
            pareto_frontier.append(method)
            best_recovery_so_far = recovery

    rec_l10 = method_stats["Muon2_L10"]["mean_recovery"]
    rec_l5 = method_stats["Muon2_L5"]["mean_recovery"]
    rec_muon = method_stats["Muon"]["mean_recovery"]
    rec_hrescale = method_stats["Muon+Hrescale"]["mean_recovery"]
    best_pareto = max(methods, key=lambda m: method_stats[m]["pareto_score"])

    threshold_tests = {
        "T1": {
            "question": "Does Muon 2.0 (Lanczos-10 Ritz) recover >80% of Full Democratic's advantage?",
            "metric": float(rec_l10),
            "reference": "> 80% recovery",
            "pass": bool(rec_l10 > 80.0),
            "details": {"per_seed_recoveries": method_stats["Muon2_L10"]["recoveries"]},
        },
        "T2": {
            "question": "Does Muon 2.0 (Lanczos-10 Ritz) beat standard Muon on mean recovery?",
            "metric": float(rec_l10 - rec_muon),
            "reference": "> 0 percentage-point recovery gap vs Muon",
            "pass": bool(rec_l10 > rec_muon),
            "details": {"Muon2_L10": float(rec_l10), "Muon": float(rec_muon)},
        },
        "T3": {
            "question": "Does Muon 2.0 (Lanczos-5 Ritz) beat standard Muon on mean recovery?",
            "metric": float(rec_l5 - rec_muon),
            "reference": "> 0 percentage-point recovery gap vs Muon",
            "pass": bool(rec_l5 > rec_muon),
            "details": {"Muon2_L5": float(rec_l5), "Muon": float(rec_muon)},
        },
        "T4": {
            "question": "Does Muon + Hessian rescale beat standard Muon on mean recovery?",
            "metric": float(rec_hrescale - rec_muon),
            "reference": "> 0 percentage-point recovery gap vs Muon",
            "pass": bool(rec_hrescale > rec_muon),
            "details": {"Muon+Hrescale": float(rec_hrescale), "Muon": float(rec_muon)},
        },
        "T5": {
            "question": "Which method has the largest proxy Pareto score under the counted-matmul proxy?",
            "metric": float(method_stats[best_pareto]["pareto_score"]),
            "reference": f"Best method = {best_pareto}",
            "pass": True,
            "details": {
                "best_method": best_pareto,
                "pareto_frontier": pareto_frontier,
                "ranking": [
                    {
                        "method": m,
                        "display_name": DISPLAY_NAMES[m],
                        "pareto_score": float(method_stats[m]["pareto_score"]),
                        "cost_ratio": float(method_stats[m]["cost_ratio"]),
                        "mean_recovery": float(method_stats[m]["mean_recovery"]),
                    }
                    for m in sorted(methods, key=lambda x: method_stats[x]["pareto_score"], reverse=True)
                ],
            },
        },
    }

    n_pass = sum(int(threshold_tests[t]["pass"]) for t in ("T1", "T2", "T3", "T4"))
    if threshold_tests["T1"]["pass"] and threshold_tests["T2"]["pass"]:
        verdict_category = "strong"
        verdict_summary = (
            "Lanczos-10 clears the 80% recovery threshold and beats standard Muon on this toy problem."
        )
    elif threshold_tests["T2"]["pass"]:
        verdict_category = "positive"
        verdict_summary = (
            "Lanczos-10 beats standard Muon, but does not clear the 80% recovery threshold in this toy setting."
        )
    elif threshold_tests["T1"]["pass"]:
        verdict_category = "mixed"
        verdict_summary = (
            "Lanczos-10 reaches high recovery but does not beat Muon; Muon may already capture most of the gain here."
        )
    else:
        verdict_category = "negative"
        verdict_summary = (
            "The Lanczos sketch does not clearly improve on Muon in this toy setting under the current setup."
        )

    aggregate = {
        "trajectory_steps": list(range(n_steps + 1)),
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "proxy_cost_caveat": COST_PROXY_CAVEAT,
        "proxy_cost_notes": dict(METHOD_PROXY_NOTES),
        "method_stats": method_stats,
        "pareto_frontier": pareto_frontier,
        "threshold_tests": threshold_tests,
        "verdict": {
            "tests_passed": int(n_pass),
            "tests_considered": 4,
            "category": verdict_category,
            "summary": verdict_summary,
            "caveat": (
                "Interpret only within this 48-parameter deep-linear toy problem and under the current counted-matmul proxy."
            ),
            "best_proxy_pareto_method": best_pareto,
        },
    }
    aggregate["summary_rows"] = _summary_rows_from_aggregate(aggregate, methods)
    aggregate["threshold_rows"] = _threshold_rows_from_aggregate(aggregate)
    return aggregate


def _build_raw_result_views(seed_results, aggregate, methods):
    """Build convenient per-seed raw result views for notebook/report consumers."""
    seed_ids = [str(seed_record["seed"]) for seed_record in seed_results]

    per_seed_best_lrs = {
        method: {
            str(seed_record["seed"]): float(seed_record["best_lrs"][method])
            for seed_record in seed_results
        }
        for method in methods
    }
    per_seed_final_losses = {
        method: {
            str(seed_record["seed"]): float(seed_record["per_method"][method]["final_loss"])
            for seed_record in seed_results
        }
        for method in methods
    }
    per_seed_loss_curves = {
        method: {
            str(seed_record["seed"]): list(seed_record["per_method"][method]["loss_curve"])
            for seed_record in seed_results
        }
        for method in methods
    }
    per_seed_matmul_proxies = {
        method: {
            str(seed_record["seed"]): int(seed_record["per_method"][method]["matmul_proxy"])
            for seed_record in seed_results
        }
        for method in methods
    }
    per_seed_recoveries = {
        method: {
            seed_id: float(recovery)
            for seed_id, recovery in zip(seed_ids, aggregate["method_stats"][method]["recoveries"])
        }
        for method in methods
    }
    per_seed_lr_sweeps = {
        str(seed_record["seed"]): seed_record["lr_sweep"]
        for seed_record in seed_results
    }
    pareto_metrics = {
        method: {
            "pareto_score": float(aggregate["method_stats"][method]["pareto_score"]),
            "cost_ratio_vs_sgd": float(aggregate["method_stats"][method]["cost_ratio"]),
            "mean_recovery_pct": float(aggregate["method_stats"][method]["mean_recovery"]),
            "mean_matmul_proxy": float(aggregate["method_stats"][method]["mean_matmul_proxy"]),
        }
        for method in methods
    }

    return {
        "per_seed_best_lrs": per_seed_best_lrs,
        "per_seed_lr_sweeps": per_seed_lr_sweeps,
        "per_seed_final_losses": per_seed_final_losses,
        "per_seed_loss_curves": per_seed_loss_curves,
        "per_seed_matmul_proxies": per_seed_matmul_proxies,
        "per_seed_recoveries": per_seed_recoveries,
        "pareto_metrics": pareto_metrics,
    }


def run_experiment(config=None, verbose=True):
    """Run the full experiment and return structured raw + aggregate results."""
    cfg = resolve_config(config)
    seeds = seeds_from_config(cfg)
    seed_results = []

    if verbose:
        print("=" * 90)
        print(EXPERIMENT_NAME)
        print("=" * 90)
        print(f"Network: {N_LAYERS}-layer deep linear {DIM}x{DIM}  ({N_PARAMS} params)")
        print(f"Steps: {cfg['n_steps']}, Hessian/Lanczos refresh every {cfg['hessian_recompute_every']} steps")
        print(f"Seeds: {cfg['n_seeds']} -> {seeds}")
        print(f"Target singular values: {cfg['target_singular_values']}")
        print("Methods:", ", ".join(DISPLAY_NAMES[m] for m in cfg["methods"]))
        print("Cost proxy caveat:")
        print(f"  {COST_PROXY_CAVEAT}")
        print()

    t_start = time.time()

    for seed in seeds:
        problem = make_seed_problem(seed, config=cfg)
        theta0 = problem["theta0"]
        target = problem["target"]
        best_lrs, lr_sweep = sweep_best_lrs(theta0, target, methods=cfg["methods"], config=cfg)

        seed_record = {
            "seed": int(seed),
            "target_condition_number": float(problem["target_condition_number"]),
            "target_singular_values": list(problem["target_singular_values"]),
            "best_lrs": dict(best_lrs),
            "lr_sweep": lr_sweep,
            "per_method": {},
        }

        if verbose:
            print(f"--- Seed {seed} (target kappa={problem['target_condition_number']:.0f}) ---")
            lr_str = ", ".join(
                f"{DISPLAY_NAMES[m].split('(')[0].strip()[:12]}={best_lrs[m]}" for m in cfg["methods"]
            )
            print(f"  Best LRs: {lr_str}")

        for method in cfg["methods"]:
            loss_curve, final_loss, matmul_proxy = run_full_counted(method, best_lrs[method], theta0, target, config=cfg)
            seed_record["per_method"][method] = {
                "best_lr": float(best_lrs[method]),
                "final_loss": float(final_loss),
                "matmul_proxy": int(matmul_proxy),
                "loss_curve": loss_curve.tolist(),
            }

        if verbose:
            finals = seed_record["per_method"]
            final_str = "  ".join(f"{m[:10]}={finals[m]['final_loss']:.4f}" for m in cfg["methods"])
            print(f"  Finals: {final_str}")
            print()

        seed_results.append(seed_record)

    elapsed = time.time() - t_start
    aggregate = aggregate_results(seed_results, cfg)
    raw_views = _build_raw_result_views(seed_results, aggregate, cfg["methods"])

    return {
        "experiment_name": EXPERIMENT_NAME,
        "script_path": __file__,
        "counterpart_notebook_path": COUNTERPART_NOTEBOOK_PATH,
        "config": cfg,
        "methods": list(cfg["methods"]),
        "display_names": {m: DISPLAY_NAMES[m] for m in cfg["methods"]},
        "seeds": seeds,
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "cost_proxy_caveat": COST_PROXY_CAVEAT,
        "runtime_seconds": float(elapsed),
        "seed_results": seed_results,
        "aggregate": aggregate,
        **raw_views,
    }


def get_summary_rows(results):
    return list(results["aggregate"]["summary_rows"])


def get_threshold_rows(results):
    return list(results["aggregate"]["threshold_rows"])


def print_report(results):
    """Pretty-print the structured results from run_experiment."""
    aggregate = results["aggregate"]
    methods = results["methods"]

    print(f"\nTotal runtime: {results['runtime_seconds']:.1f}s\n")
    print("=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)
    print(COST_PROXY_CAVEAT)
    print()

    header = f"{'Method':<32} {'Final Loss':>12} {'+-Std':>10} {'MatmulPx':>10} {'Recovery%':>12} {'ParetoPx':>10}"
    print(header)
    print("-" * len(header))
    for method in methods:
        stats = aggregate["method_stats"][method]
        print(
            f"{DISPLAY_NAMES[method]:<32} "
            f"{stats['mean_loss']:>12.6f} "
            f"{stats['std_loss']:>10.6f} "
            f"{stats['mean_matmul_proxy']:>10.0f} "
            f"{stats['mean_recovery']:>11.1f}% "
            f"{stats['pareto_score']:>10.1f}"
        )
    print()

    print("Per-seed recovery % (relative to the SGD-to-FullDemocratic final-loss gap):")
    print(f"{'Method':<32}", end="")
    for i, seed in enumerate(results["seeds"]):
        print(f" {('seed'+str(i))[:8]:>8}", end="")
    print(f" {'Mean':>8}")
    print("-" * (32 + 9 * (len(results["seeds"]) + 1)))
    for method in methods:
        stats = aggregate["method_stats"][method]
        print(f"{DISPLAY_NAMES[method]:<32}", end="")
        for rec in stats["recoveries"]:
            print(f" {rec:>7.1f}%", end="")
        print(f" {stats['mean_recovery']:>7.1f}%")
    print()

    print("Loss trajectory snapshot (seed 0; post-update losses):")
    print(TRAJECTORY_SEMANTICS)
    step_points = sorted(set(list(range(0, results["config"]["n_steps"] + 1, 50)) + [results["config"]["n_steps"]]))
    print(f"{'Step':>5}", end="")
    for method in methods:
        print(f" {DISPLAY_NAMES[method][:16]:>16}", end="")
    print()
    seed0 = results["seed_results"][0]
    for step in step_points:
        print(f"{step:>5}", end="")
        for method in methods:
            lc = seed0["per_method"][method]["loss_curve"]
            print(f" {lc[step]:>16.6f}", end="")
        print()
    print()

    print("Counted-matmul proxy notes:")
    print(f"{'Method':<32} {'MatmulPx':>10} {'Cost ratio':>12}  Note")
    print("-" * 120)
    for method in methods:
        stats = aggregate["method_stats"][method]
        print(
            f"{DISPLAY_NAMES[method]:<32} {stats['mean_matmul_proxy']:>10.0f} "
            f"{stats['cost_ratio']:>11.2f}x  {METHOD_PROXY_NOTES[method]}"
        )
    print()

    print("=" * 90)
    print("THRESHOLD TESTS")
    print("=" * 90)
    for test_id in ("T1", "T2", "T3", "T4", "T5"):
        test = aggregate["threshold_tests"][test_id]
        verdict = "PASS" if test["pass"] else "FAIL"
        print(f"{test_id}: {test['question']}")
        print(f"    Metric: {test['metric']:.3f}")
        print(f"    Reference: {test['reference']}")
        print(f"    --> {verdict}")
        if test_id == "T5":
            detail = test["details"]
            print(f"    Proxy Pareto frontier: {[DISPLAY_NAMES[m] for m in detail['pareto_frontier']]}")
        print()

    verdict = aggregate["verdict"]
    print("=" * 90)
    print("SUMMARY VERDICT")
    print("=" * 90)
    print(f"Tests passed: {verdict['tests_passed']}/{verdict['tests_considered']}")
    print(f"Category: {verdict['category']}")
    print(verdict["summary"])
    print(verdict["caveat"])
    print(f"Best proxy Pareto method: {DISPLAY_NAMES[verdict['best_proxy_pareto_method']]}")
    print("=" * 90)


def main():
    results = run_experiment(verbose=True)
    print_report(results)


if __name__ == "__main__":
    main()
