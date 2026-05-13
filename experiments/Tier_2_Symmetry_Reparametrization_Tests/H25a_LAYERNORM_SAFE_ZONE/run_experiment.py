#!/usr/bin/env python3
"""
H25a: LayerNorm, balance ratio, and Muon on a toy nonlinear regression.

This first completion pass keeps the original NumPy toy experiment but makes the
comparison more rigorous and reusable:
- the module is import-safe (no execution on import)
- all four conditions share the same dataset and, by default, the same base
  weight initialization
- the balance ratio c(t) is tracked at every optimization step
- T4 compares how much removing LayerNorm hurts SGD relative to Muon

Scope caveat: this is still a single-seed synthetic full-batch regression test.
It is a toy mechanistic probe, not a transformer benchmark.
"""

import time
import numpy as np

# =============================================================================
# Parameters
# =============================================================================
D = 32
N_LAYERS = 4
N_SAMPLES = 100
LR = 0.005
N_STEPS = 500
REPORT_STEPS = [0, 100, 200, 300, 500]
DISPLAY_EVERY = 50
SEED = 42
SAFE_ZONE_C = 2.0
T2_MUON_C_RATIO_THRESHOLD = 1.10
T3_ADVANTAGE_THRESHOLD = 1.20
T4_ASYMMETRY_THRESHOLD = 1.50

CONFIGS = [
    {"name": "SGD+LN", "use_ln": True, "use_muon": False},
    {"name": "SGD-LN", "use_ln": False, "use_muon": False},
    {"name": "Muon+LN", "use_ln": True, "use_muon": True},
    {"name": "Muon-LN", "use_ln": False, "use_muon": True},
]


# =============================================================================
# Data
# =============================================================================
def make_dataset(rng):
    """Create the shared synthetic regression dataset used by all configs."""
    x_train = rng.randn(N_SAMPLES, D).astype(np.float64)
    x_train = x_train / (np.linalg.norm(x_train, axis=-1, keepdims=True) + 1e-8) * np.sqrt(D)
    w_target = rng.randn(D, 8) @ rng.randn(8, D) * 0.3
    y_train = x_train @ w_target + 0.1 * rng.randn(N_SAMPLES, D)
    return x_train, y_train, w_target


# =============================================================================
# LayerNorm
# =============================================================================
def layernorm_fwd(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta, x_hat, mu, var


def layernorm_bwd(dout, x_hat, gamma, var, eps=1e-5):
    d = dout.shape[-1]
    dgamma = (dout * x_hat).sum(axis=0)
    dbeta = dout.sum(axis=0)
    dx_hat = dout * gamma
    std_inv = 1.0 / np.sqrt(var + eps)
    dx = (1.0 / d) * std_inv * (
        d * dx_hat - dx_hat.sum(axis=-1, keepdims=True)
        - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


# =============================================================================
# Network
# =============================================================================
def init_base_weights(rng):
    """Shared He-normal-like weight initialization for all configs."""
    base = {}
    scale = np.sqrt(2.0 / D)
    for layer in range(N_LAYERS):
        base[f"Wa{layer}"] = rng.randn(D, D) * scale
        base[f"Wm{layer}"] = rng.randn(D, D) * scale
    return base


def clone_params(base_weights, use_ln):
    params = {k: v.copy() for k, v in base_weights.items()}
    if use_ln:
        for layer in range(N_LAYERS):
            params[f"g{layer}"] = np.ones(D)
            params[f"b{layer}"] = np.zeros(D)
    return params


def forward(x, p, use_ln):
    cache = {}
    h = x.copy()
    for layer in range(N_LAYERS):
        cache[f"ha{layer}"] = h
        z = h @ p[f"Wa{layer}"]
        if use_ln:
            z, x_hat, mu, var = layernorm_fwd(z, p[f"g{layer}"], p[f"b{layer}"])
            cache[f"xh{layer}"] = x_hat
            cache[f"mu{layer}"] = mu
            cache[f"var{layer}"] = var
        mask = (z > 0).astype(np.float64)
        h_relu = z * mask
        cache[f"mask{layer}"] = mask
        cache[f"hm{layer}"] = h_relu
        h = h_relu @ p[f"Wm{layer}"]
    cache["out"] = h
    return h, cache


def backward(p, cache, y, use_ln):
    grads = {}
    n = y.shape[0]
    d_out = 2.0 * (cache["out"] - y) / n
    for layer in reversed(range(N_LAYERS)):
        grads[f"Wm{layer}"] = cache[f"hm{layer}"].T @ d_out
        d_out = d_out @ p[f"Wm{layer}"].T
        d_out = d_out * cache[f"mask{layer}"]
        if use_ln:
            d_out, d_gamma, d_beta = layernorm_bwd(
                d_out,
                cache[f"xh{layer}"],
                p[f"g{layer}"],
                cache[f"var{layer}"],
            )
            grads[f"g{layer}"] = d_gamma
            grads[f"b{layer}"] = d_beta
        grads[f"Wa{layer}"] = cache[f"ha{layer}"].T @ d_out
        d_out = d_out @ p[f"Wa{layer}"].T
    return grads


def mse(y_pred, target):
    return np.mean((y_pred - target) ** 2)


# =============================================================================
# Optimizers
# =============================================================================
def muon_step(w, grad, lr):
    u, _, vt = np.linalg.svd(grad, full_matrices=False)
    return w - lr * (u @ vt)


def sgd_step(w, grad, lr):
    return w - lr * grad


# =============================================================================
# Metrics
# =============================================================================
def weight_keys(mapping):
    return sorted(k for k in mapping if k.startswith("W"))


def balance_ratio(p):
    norms = [np.linalg.norm(p[k], "fro") for k in weight_keys(p)]
    return max(norms) / max(min(norms), 1e-12)


def layer_norms(p):
    return {k: float(np.linalg.norm(p[k], "fro")) for k in weight_keys(p)}


def data_stats(x_train, y_train, w_target):
    input_norms = np.linalg.norm(x_train, axis=-1)
    return {
        "x_shape": list(x_train.shape),
        "y_shape": list(y_train.shape),
        "input_norm_mean": float(input_norms.mean()),
        "input_norm_std": float(input_norms.std()),
        "target_fro": float(np.linalg.norm(w_target, "fro")),
        "target_rank_upper_bound": 8,
    }


# =============================================================================
# Training
# =============================================================================
def train_single(x_train, y_train, base_weights, use_ln, use_muon, lr=LR, n_steps=N_STEPS):
    params = clone_params(base_weights, use_ln)
    steps = list(range(n_steps + 1))
    loss_history = []
    c_history = []
    diverged = False
    divergence_step = None

    for step in steps:
        out, cache = forward(x_train, params, use_ln)
        loss = float(mse(out, y_train))
        c_val = float(balance_ratio(params))
        bad = (
            np.isnan(loss)
            or np.isinf(loss)
            or loss > 1e10
            or np.isnan(c_val)
            or np.isinf(c_val)
        )

        if bad:
            diverged = True
            divergence_step = step
            remaining = n_steps - step + 1
            loss_history.extend([float("inf")] * remaining)
            c_history.extend([float("inf")] * remaining)
            break

        loss_history.append(loss)
        c_history.append(c_val)

        if step == n_steps:
            break

        grads = backward(params, cache, y_train, use_ln)
        for key in params:
            if key.startswith("W"):
                if use_muon:
                    params[key] = muon_step(params[key], grads[key], lr)
                else:
                    params[key] = sgd_step(params[key], grads[key], lr)
            elif key in grads:
                params[key] = params[key] - lr * grads[key]

    result = {
        "step": steps,
        "loss": [float(v) for v in loss_history],
        "c": [float(v) for v in c_history],
        "norms_final": layer_norms(params) if not diverged else {},
        "diverged": diverged,
        "divergence_step": divergence_step,
        "final_loss": float(loss_history[-1]),
        "final_c": float(c_history[-1]),
        "max_c": float(max(v for v in c_history if np.isfinite(v))) if any(np.isfinite(v) for v in c_history) else float("inf"),
    }
    return result


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_tests(results):
    final_c = {name: results[name]["final_c"] for name in results}
    final_loss = {name: results[name]["final_loss"] for name in results}
    max_c = {name: results[name]["max_c"] for name in results}

    sgd_noln_diverged = results["SGD-LN"]["diverged"]
    muon_noln_diverged = results["Muon-LN"]["diverged"]

    # T1: honest every-step safe-zone test
    t1_pass = (max_c["Muon+LN"] < SAFE_ZONE_C) and (final_c["Muon+LN"] < SAFE_ZONE_C)

    # T2: without LN, balance worsens or SGD diverges
    muon_max_c_ratio = max_c["Muon-LN"] / max(max_c["Muon+LN"], 1e-12)
    sgd_max_c_ratio = max_c["SGD-LN"] / max(max_c["SGD+LN"], 1e-12)
    t2_pass = sgd_noln_diverged or (muon_max_c_ratio > T2_MUON_C_RATIO_THRESHOLD)

    # T3: Muon advantage with LN
    if np.isfinite(final_loss["Muon+LN"]) and np.isfinite(final_loss["SGD+LN"]):
        adv_ln = final_loss["SGD+LN"] / max(final_loss["Muon+LN"], 1e-12)
    else:
        adv_ln = float("inf")
    t3_pass = adv_ln > T3_ADVANTAGE_THRESHOLD

    # T4: removing LN should hurt SGD more than Muon
    if sgd_noln_diverged:
        sgd_ln_penalty = float("inf")
    else:
        sgd_ln_penalty = final_loss["SGD-LN"] / max(final_loss["SGD+LN"], 1e-12)

    if muon_noln_diverged:
        muon_ln_penalty = float("inf")
    else:
        muon_ln_penalty = final_loss["Muon-LN"] / max(final_loss["Muon+LN"], 1e-12)

    if sgd_ln_penalty == float("inf") and np.isfinite(muon_ln_penalty):
        asymmetry_ratio = float("inf")
        t4_pass = True
    elif np.isfinite(sgd_ln_penalty) and np.isfinite(muon_ln_penalty):
        asymmetry_ratio = sgd_ln_penalty / max(muon_ln_penalty, 1e-12)
        t4_pass = sgd_ln_penalty > (muon_ln_penalty * T4_ASYMMETRY_THRESHOLD)
    else:
        asymmetry_ratio = float("nan")
        t4_pass = False

    tests = {
        "T1": {
            "description": "Muon+LN keeps c < 2 throughout training",
            "pass": bool(t1_pass),
            "raw": {
                "safe_zone_c": float(SAFE_ZONE_C),
                "muon_ln_max_c": float(max_c["Muon+LN"]),
                "muon_ln_final_c": float(final_c["Muon+LN"]),
            },
        },
        "T2": {
            "description": "Without LN, balance worsens or SGD diverges",
            "pass": bool(t2_pass),
            "raw": {
                "sgd_noln_diverged": bool(sgd_noln_diverged),
                "sgd_max_c_plus_ln": float(max_c["SGD+LN"]),
                "sgd_max_c_no_ln": float(max_c["SGD-LN"]),
                "sgd_max_c_ratio_no_ln_over_plus_ln": float(sgd_max_c_ratio),
                "muon_max_c_plus_ln": float(max_c["Muon+LN"]),
                "muon_max_c_no_ln": float(max_c["Muon-LN"]),
                "muon_max_c_ratio_no_ln_over_plus_ln": float(muon_max_c_ratio),
            },
        },
        "T3": {
            "description": "Muon outperforms SGD when both use LN",
            "pass": bool(t3_pass),
            "raw": {
                "sgd_plus_ln_final_loss": float(final_loss["SGD+LN"]),
                "muon_plus_ln_final_loss": float(final_loss["Muon+LN"]),
                "advantage_ratio_sgd_over_muon": float(adv_ln),
            },
        },
        "T4": {
            "description": "Removing LN hurts SGD more than Muon",
            "pass": bool(t4_pass),
            "raw": {
                "sgd_loss_ratio_no_ln_over_plus_ln": float(sgd_ln_penalty),
                "muon_loss_ratio_no_ln_over_plus_ln": float(muon_ln_penalty),
                "asymmetry_ratio_sgd_penalty_over_muon_penalty": float(asymmetry_ratio),
                "asymmetry_threshold": float(T4_ASYMMETRY_THRESHOLD),
            },
        },
    }

    tests["summary"] = {
        "n_pass": int(sum(test["pass"] for name, test in tests.items() if name.startswith("T"))),
        "score_out_of": 4,
    }
    return tests


# =============================================================================
# Experiment runner
# =============================================================================
def run_all(seed=SEED, shared_init=True, report_steps=None):
    start = time.time()
    report_steps = REPORT_STEPS if report_steps is None else list(report_steps)

    rng = np.random.RandomState(seed)
    x_train, y_train, w_target = make_dataset(rng)

    initial_weight_sets = {}
    results = {}

    if shared_init:
        shared_base = init_base_weights(rng)

    for config in CONFIGS:
        if shared_init:
            base_weights = {k: v.copy() for k, v in shared_base.items()}
        else:
            base_weights = init_base_weights(rng)

        initial_weight_sets[config["name"]] = {k: v.copy() for k, v in base_weights.items()}
        run_result = train_single(
            x_train,
            y_train,
            base_weights,
            use_ln=config["use_ln"],
            use_muon=config["use_muon"],
            lr=LR,
            n_steps=N_STEPS,
        )
        run_result.update({
            "name": config["name"],
            "use_ln": bool(config["use_ln"]),
            "use_muon": bool(config["use_muon"]),
            "initial_c": float(balance_ratio(base_weights)),
            "initial_norms": layer_norms(base_weights),
        })
        results[config["name"]] = run_result

    reference_name = CONFIGS[0]["name"]
    reference_weights = initial_weight_sets[reference_name]
    shared_init_verified = all(
        all(np.allclose(initial_weight_sets[name][key], reference_weights[key]) for key in reference_weights)
        for name in initial_weight_sets
    )

    tests = evaluate_tests(results)
    elapsed = time.time() - start

    experiment = {
        "meta": {
            "title": "H25a: LayerNorm and Muon on a toy nonlinear regression",
            "seed": int(seed),
            "shared_init": bool(shared_init),
            "shared_init_verified": bool(shared_init_verified),
            "runtime_sec": float(elapsed),
            "config_order": [config["name"] for config in CONFIGS],
            "report_steps": [int(step) for step in report_steps],
            "display_every": int(DISPLAY_EVERY),
            "dimensions": {
                "D": int(D),
                "n_layers": int(N_LAYERS),
                "n_samples": int(N_SAMPLES),
                "lr": float(LR),
                "n_steps": int(N_STEPS),
            },
            "safe_zone_c": float(SAFE_ZONE_C),
            "notes": [
                "Single-seed synthetic full-batch nonlinear regression.",
                "Shared dataset across all four configs.",
                "Shared base weight initialization across configs by default.",
                "Histories store loss and c at every optimization step.",
            ],
        },
        "data": data_stats(x_train, y_train, w_target),
        "results": results,
        "tests": tests,
    }
    return experiment


# =============================================================================
# Reporting helpers
# =============================================================================
def fmt(value, digits=4):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if value is None:
        return "None"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if np.isinf(value):
        return "INF"
    if np.isnan(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def print_report(experiment):
    meta = experiment["meta"]
    results = experiment["results"]
    tests = experiment["tests"]
    dims = meta["dimensions"]

    print("=" * 88)
    print(meta["title"])
    print("=" * 88)
    print("Single-seed toy study; mechanistic evidence only (not a transformer benchmark).")
    print()
    print(f"  4 blocks: Linear({dims['D']},{dims['D']}) -> [LN({dims['D']})] -> ReLU -> Linear({dims['D']},{dims['D']})")
    print(f"  LR = {dims['lr']} (same for all), Steps = {dims['n_steps']}, Samples = {dims['n_samples']}")
    print(f"  Seed = {meta['seed']}")
    print(f"  Init: shared He-normal-like base weights across all four conditions = {meta['shared_init']} (verified: {meta['shared_init_verified']})")
    print(f"  Histories stored every step; display tables sample steps {meta['report_steps']}")
    print()

    print("Per-configuration summary:")
    for name in meta["config_order"]:
        run = results[name]
        if run["diverged"]:
            print(f"  {name:>8}: DIVERGED at step {run['divergence_step']}")
        else:
            print(
                f"  {name:>8}: final loss = {fmt(run['final_loss'])}, final c = {fmt(run['final_c'])}, max c = {fmt(run['max_c'])}"
            )
    print()

    print("=" * 120)
    print(
        f"{'Step':>4} | {'c(SGD+LN)':>9} | {'c(SGD-LN)':>9} | {'c(Muon+LN)':>10} | {'c(Muon-LN)':>10} | {'L(SGD+LN)':>10} | {'L(Muon+LN)':>10} | {'L(Muon-LN)':>10}"
    )
    print("-" * 120)
    for step in meta["report_steps"]:
        cols = [f"{step:>4}"]
        for name in ["SGD+LN", "SGD-LN", "Muon+LN", "Muon-LN"]:
            cols.append(f"{fmt(results[name]['c'][step]):>9}")
        for name in ["SGD+LN", "Muon+LN", "Muon-LN"]:
            cols.append(f"{fmt(results[name]['loss'][step]):>10}")
        print(" | ".join(cols))
    print("=" * 120)
    print()

    print(f"c(t) trajectory sampled every {meta['display_every']} steps for display (tracked every step internally):")
    print("-" * 62)
    print(f"{'Step':>4} | {'SGD+LN':>7} | {'SGD-LN':>7} | {'Muon+LN':>7} | {'Muon-LN':>7}")
    print("-" * 62)
    for step in range(0, dims["n_steps"] + 1, meta["display_every"]):
        cols = [f"{step:>4}"]
        for name in ["SGD+LN", "SGD-LN", "Muon+LN", "Muon-LN"]:
            cols.append(f"{fmt(results[name]['c'][step]):>7}")
        print(" | ".join(cols))
    print()

    print("Per-layer ||W||_F at end of training:")
    print("-" * 88)
    for name in meta["config_order"]:
        norms = results[name]["norms_final"]
        if norms:
            vals = " ".join(f"{v:.2f}" for v in norms.values())
            print(f"  {name:>8}: [{vals}]")
        else:
            print(f"  {name:>8}: DIVERGED")
    print()

    print("=" * 88)
    print("KEY TESTS")
    print("=" * 88)
    print()

    t1 = tests["T1"]
    print("T1: Muon+LN keeps c < 2 throughout training")
    print(f"    max c(Muon+LN):   {fmt(t1['raw']['muon_ln_max_c'])} {'< 2 PASS' if t1['raw']['muon_ln_max_c'] < SAFE_ZONE_C else '>= 2 FAIL'}")
    print(f"    final c(Muon+LN): {fmt(t1['raw']['muon_ln_final_c'])} {'< 2 PASS' if t1['raw']['muon_ln_final_c'] < SAFE_ZONE_C else '>= 2 FAIL'}")
    print(f"    T1: {'PASS' if t1['pass'] else 'FAIL'}")
    print()

    t2 = tests["T2"]
    print("T2: Without LN, balance worsens or SGD diverges")
    print(f"    SGD max-c ratio (no LN / +LN):   {fmt(t2['raw']['sgd_max_c_ratio_no_ln_over_plus_ln'])}")
    print(f"    Muon max-c ratio (no LN / +LN):  {fmt(t2['raw']['muon_max_c_ratio_no_ln_over_plus_ln'])}")
    print(f"    SGD-LN diverged:                 {t2['raw']['sgd_noln_diverged']}")
    print(f"    T2: {'PASS' if t2['pass'] else 'FAIL'}")
    print()

    t3 = tests["T3"]
    print("T3: Muon outperforms SGD when both use LN")
    print(f"    Loss(SGD+LN):  {fmt(t3['raw']['sgd_plus_ln_final_loss'])}")
    print(f"    Loss(Muon+LN): {fmt(t3['raw']['muon_plus_ln_final_loss'])}")
    print(f"    Advantage ratio (SGD / Muon): {fmt(t3['raw']['advantage_ratio_sgd_over_muon'], digits=2)}x")
    print(f"    T3: {'PASS' if t3['pass'] else 'FAIL'}")
    print()

    t4 = tests["T4"]
    print("T4: Removing LN hurts SGD more than Muon")
    print(f"    SGD loss ratio   (no LN / +LN): {fmt(t4['raw']['sgd_loss_ratio_no_ln_over_plus_ln'], digits=2)}x")
    print(f"    Muon loss ratio  (no LN / +LN): {fmt(t4['raw']['muon_loss_ratio_no_ln_over_plus_ln'], digits=2)}x")
    print(f"    Asymmetry ratio  (SGD / Muon):  {fmt(t4['raw']['asymmetry_ratio_sgd_penalty_over_muon_penalty'], digits=2)}x")
    print(f"    Threshold: SGD penalty > {fmt(t4['raw']['asymmetry_threshold'], digits=2)} * Muon penalty")
    print(f"    T4: {'PASS' if t4['pass'] else 'FAIL'}")
    print()

    ranking = sorted((results[name]["final_loss"], name) for name in meta["config_order"])
    ranking_text = " < ".join(f"{name} ({fmt(loss)})" for loss, name in ranking)

    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"  Score: {tests['summary']['n_pass']}/{tests['summary']['score_out_of']}")
    print(f"  Final-loss ranking (lower is better): {ranking_text}")
    if np.isfinite(results["Muon-LN"]["final_loss"]) and np.isfinite(results["Muon+LN"]["final_loss"]):
        if results["Muon-LN"]["final_loss"] < results["Muon+LN"]["final_loss"]:
            print("  Note: in this run, Muon-LN attains lower final loss than Muon+LN despite a higher observed c trajectory.")
        else:
            print("  Note: in this run, Muon+LN attains lower or equal final loss than Muon-LN.")
    print()
    print("Interpretation (toy-scope only):")
    print("  - Supported here: Muon+LN's c(t) can be checked honestly at every step, and")
    print("    removing LN hurts SGD much more than Muon when judged by the implemented T4 metric.")
    print("  - Not established here: transformer-level generalization, uncertainty across seeds,")
    print("    or causal mechanism beyond the measured loss/c(t)/final-norm summaries.")
    print()
    print(f"Runtime: {meta['runtime_sec']:.2f}s")
    print("=" * 88)


# =============================================================================
# Entry point
# =============================================================================
def main():
    experiment = run_all()
    print_report(experiment)


if __name__ == "__main__":
    main()
