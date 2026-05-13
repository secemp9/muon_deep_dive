#!/usr/bin/env python3
"""
H19b: Hessian-basis vs SV-basis partial equalization
====================================================

This file implements a small toy geometric probe on a 3-layer 4x4 deep linear
network. The probe compares several update directions at the same warm-start
point and asks how they distribute energy in the Hessian eigenbasis.

What this script does:
- warm-start a tiny deep linear network with 50 momentum-SGD steps
- compute a full 48x48 Hessian by finite differences at that point
- compare SGD, partial/full SV-basis equalization, partial/full Hessian-basis
  equalization, and Muon-style polar-factor updates
- report curvature-weighted alignment, Hessian-eigenbasis energy distribution,
  and a small outcome-linked metric: one-step loss change after norm-matching
  each update to ||g|| and taking the same step size

Important caveats:
- This is a toy single-point diagnostic, not an end-to-end performance study and
  not a completed mechanistic explanation of the broader Muon/SV paradox.
- The Hessian-energy sectors are split by algebraic eigenvalue order from
  ``np.linalg.eigh``. The lowest-third sector therefore means the eigenvectors
  with the most negative/smallest eigenvalues, not necessarily the flattest
  directions by |lambda|.
- In this square 4x4 setting, ``SV_full_k4`` is exactly the same update as
  ``Muon`` because both compute the per-layer polar factor ``U @ V^T``.
"""

from __future__ import annotations

import time
import numpy as np

DIM = 4
N_LAYERS = 3
N_PARAMS = N_LAYERS * DIM * DIM  # 48
WARMUP_STEPS = 50
NUM_SEEDS = 5
LR = 0.01
MOMENTUM = 0.9
FD_EPS = 1e-5

METHOD_ORDER = (
    "SGD",
    "SV_partial_k2",
    "SV_full_k4",
    "Hessian_partial_k10",
    "Hessian_full_k24",
    "Muon",
)

METHOD_DESCRIPTIONS = {
    "SGD": "Raw gradient.",
    "SV_partial_k2": "Per-layer SV equalization of the top 2 singular values, then per-layer rescaling.",
    "SV_full_k4": "Per-layer full SV equalization; in this 4x4 setup this is exactly U @ V^T.",
    "Hessian_partial_k10": "Equalize the first 10 and last 10 Hessian-eigenbasis coefficients in algebraic eig-order.",
    "Hessian_full_k24": "Equalize all 48 Hessian-eigenbasis coefficients because k=24 selects both ends.",
    "Muon": "Per-layer polar factor U @ V^T.",
}

ENERGY_SECTOR_LABELS = {
    "lowest_eig_third_energy": "Lowest algebraic eigenvalue third (most negative/smallest eigenvalues)",
    "middle_eig_third_energy": "Middle algebraic eigenvalue third",
    "highest_eig_third_energy": "Highest algebraic eigenvalue third (largest eigenvalues)",
}


def pack(Ws):
    return np.concatenate([W.ravel() for W in Ws])


def unpack(theta):
    Ws = []
    idx = 0
    for _ in range(N_LAYERS):
        Ws.append(theta[idx:idx + DIM * DIM].reshape(DIM, DIM))
        idx += DIM * DIM
    return Ws


def forward(Ws, X):
    out = X.copy()
    for W in Ws:
        out = W @ out
    return out


def loss_fn(theta, X, Y):
    Ws = unpack(theta)
    pred = forward(Ws, X)
    return 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))


def grad_fn(theta, X, Y):
    Ws = unpack(theta)
    n = X.shape[1]
    acts = [X.copy()]
    for W in Ws:
        acts.append(W @ acts[-1])
    delta = (acts[-1] - Y) / n
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
        tp = theta.copy()
        tp[i] += FD_EPS
        tm = theta.copy()
        tm[i] -= FD_EPS
        gp, _ = grad_fn(tp, X, Y)
        gm, _ = grad_fn(tm, X, Y)
        H[:, i] = (gp - gm) / (2 * FD_EPS)
    return 0.5 * (H + H.T)


def partial_sv_eq_step(grads_list, k_per_layer):
    """Apply partial SV equalization per layer and return a flat update vector."""
    step_layers = []
    for G in grads_list:
        U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
        d = len(sigma)
        kk = min(k_per_layer, d)
        if kk == 0:
            step_layers.append(G)
        elif kk >= d:
            step_layers.append(U @ Vt)
        else:
            sigma_new = sigma.copy()
            sigma_new[:kk] = np.mean(sigma[:kk])
            target_norm = np.sqrt(d)
            current_norm = np.linalg.norm(sigma_new)
            if current_norm > 1e-15:
                sigma_new *= target_norm / current_norm
            step_layers.append(U @ np.diag(sigma_new) @ Vt)
    return pack(step_layers)


def partial_hessian_eq_step(g_vec, eigvecs, k):
    """
    Equalize the first-k and last-k Hessian-eigenbasis coefficients in algebraic
    eigenvalue order, preserving their signs, then renormalize to ||g||.
    """
    n = len(g_vec)
    projs = eigvecs.T @ g_vec
    if 2 * k >= n:
        selected = list(range(n))
    else:
        selected = list(range(k)) + list(range(n - k, n))

    eq_projs = projs.copy()
    mean_mag = np.mean(np.abs(projs[selected]))
    for idx in selected:
        eq_projs[idx] = np.sign(projs[idx]) * mean_mag

    result = eigvecs @ eq_projs
    result_norm = np.linalg.norm(result)
    grad_norm = np.linalg.norm(g_vec)
    if result_norm > 1e-15:
        result *= grad_norm / result_norm
    return result


def full_muon_step(grads_list):
    """Full per-layer polar factor U @ V^T."""
    step_layers = []
    for G in grads_list:
        U, _, Vt = np.linalg.svd(G, full_matrices=False)
        step_layers.append(U @ Vt)
    return pack(step_layers)


def curvature_weighted_alignment(update_vec, eigvecs, eigvals):
    """
    Compute sum_i |lambda_i| * |<update, v_i>|^2 / ||update||^2.
    Higher means more energy in high-|lambda| directions.
    """
    projs = eigvecs.T @ update_vec
    norm = np.linalg.norm(update_vec)
    if norm < 1e-15:
        return 0.0
    projs_normalized = projs / norm
    return float(np.sum(np.abs(eigvals) * projs_normalized ** 2))


def energy_distribution(update_vec, eigvecs, n_params):
    """
    Return the energy fractions in the lowest/middle/highest thirds of the
    Hessian eigenvectors in algebraic eigenvalue order.
    """
    projs = eigvecs.T @ update_vec
    energy = projs ** 2
    total = np.sum(energy) + 1e-30
    third = n_params // 3
    lowest = np.sum(energy[:third]) / total
    middle = np.sum(energy[third:2 * third]) / total
    highest = np.sum(energy[2 * third:]) / total
    return float(lowest), float(middle), float(highest)


def normalize_to_norm(vec, target_norm):
    current_norm = np.linalg.norm(vec)
    if current_norm < 1e-15:
        return vec.copy()
    return vec * (target_norm / current_norm)


def one_step_loss_delta(theta, X, Y, update_vec, step_size, target_update_norm):
    matched_update = normalize_to_norm(update_vec, target_update_norm)
    before = loss_fn(theta, X, Y)
    after = loss_fn(theta - step_size * matched_update, X, Y)
    return float(after - before)


def build_updates(g_vec, grads_list, eigvecs):
    return {
        "SGD": g_vec,
        "SV_partial_k2": partial_sv_eq_step(grads_list, 2),
        "SV_full_k4": partial_sv_eq_step(grads_list, 4),
        "Hessian_partial_k10": partial_hessian_eq_step(g_vec, eigvecs, 10),
        "Hessian_full_k24": partial_hessian_eq_step(g_vec, eigvecs, 24),
        "Muon": full_muon_step(grads_list),
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def run_experiment(verbose=False):
    """Run the H19b toy probe and return structured raw + aggregate results."""
    start_time = time.perf_counter()
    seeds = [42 + i * 137 for i in range(NUM_SEEDS)]

    raw_rows = []
    seed_rows = []

    for seed_index, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(DIM, 64) * 0.3
        Y = rng.randn(DIM, 64) * 0.3
        weights = [np.eye(DIM) + rng.randn(DIM, DIM) * 0.1 for _ in range(N_LAYERS)]
        theta = pack(weights)

        initial_loss = loss_fn(theta, X, Y)

        momentum_vec = np.zeros_like(theta)
        for _ in range(WARMUP_STEPS):
            g_vec, _ = grad_fn(theta, X, Y)
            momentum_vec = MOMENTUM * momentum_vec + g_vec
            theta = theta - LR * momentum_vec

        probe_loss = loss_fn(theta, X, Y)
        H = hessian_fd(theta, X, Y)
        eigvals, eigvecs = np.linalg.eigh(H)
        g_vec, grads_list = grad_fn(theta, X, Y)
        grad_norm = float(np.linalg.norm(g_vec))
        hessian_symmetry_residual = float(np.linalg.norm(H - H.T))
        num_negative_eigvals = int(np.sum(eigvals < 0.0))
        trace = float(np.trace(H))
        extreme_eigenvalue_ratio = float(eigvals[-1] / (abs(eigvals[0]) + 1e-15))

        updates = build_updates(g_vec, grads_list, eigvecs)
        sv_full_minus_muon_max_abs = float(np.max(np.abs(updates["SV_full_k4"] - updates["Muon"])))
        sv_full_equals_muon = bool(np.allclose(updates["SV_full_k4"], updates["Muon"], atol=1e-12, rtol=1e-12))

        seed_rows.append({
            "seed": int(seed),
            "seed_index": int(seed_index),
            "initial_loss": float(initial_loss),
            "probe_loss": float(probe_loss),
            "grad_norm": grad_norm,
            "eig_min": float(eigvals[0]),
            "eig_max": float(eigvals[-1]),
            "num_negative_eigvals": num_negative_eigvals,
            "trace": trace,
            "extreme_eigenvalue_ratio": extreme_eigenvalue_ratio,
            "hessian_symmetry_residual": hessian_symmetry_residual,
            "sv_full_minus_muon_max_abs": sv_full_minus_muon_max_abs,
            "sv_full_equals_muon": sv_full_equals_muon,
        })

        if verbose:
            print(
                f"Seed {seed_index + 1}/{len(seeds)} seed={seed}: "
                f"probe_loss={probe_loss:.6f}, grad_norm={grad_norm:.6f}, "
                f"eig_min={eigvals[0]:.4e}, eig_max={eigvals[-1]:.4e}, "
                f"SV_full==Muon={sv_full_equals_muon}"
            )

        for method_name in METHOD_ORDER:
            update = updates[method_name]
            cwa = curvature_weighted_alignment(update, eigvecs, eigvals)
            lowest, middle, highest = energy_distribution(update, eigvecs, N_PARAMS)
            energy_sum = lowest + middle + highest
            norm_matched_delta = one_step_loss_delta(
                theta,
                X,
                Y,
                update,
                step_size=LR,
                target_update_norm=grad_norm,
            )

            raw_rows.append({
                "seed": int(seed),
                "seed_index": int(seed_index),
                "method": method_name,
                "curvature_weighted_alignment": float(cwa),
                "lowest_eig_third_energy": float(lowest),
                "middle_eig_third_energy": float(middle),
                "highest_eig_third_energy": float(highest),
                "energy_sum": float(energy_sum),
                "update_norm": float(np.linalg.norm(update)),
                "grad_norm": grad_norm,
                "norm_matched_one_step_loss_delta": float(norm_matched_delta),
                "initial_loss": float(initial_loss),
                "probe_loss": float(probe_loss),
                "eig_min": float(eigvals[0]),
                "eig_max": float(eigvals[-1]),
                "num_negative_eigvals": num_negative_eigvals,
                "trace": trace,
                "extreme_eigenvalue_ratio": extreme_eigenvalue_ratio,
                "hessian_symmetry_residual": hessian_symmetry_residual,
                "sv_full_equals_muon": sv_full_equals_muon,
                "sv_full_minus_muon_max_abs": sv_full_minus_muon_max_abs,
            })

    aggregate_rows = []
    for method_name in METHOD_ORDER:
        method_rows = [row for row in raw_rows if row["method"] == method_name]
        cwa_mean, cwa_std = _mean_std([row["curvature_weighted_alignment"] for row in method_rows])
        low_mean, low_std = _mean_std([row["lowest_eig_third_energy"] for row in method_rows])
        mid_mean, mid_std = _mean_std([row["middle_eig_third_energy"] for row in method_rows])
        high_mean, high_std = _mean_std([row["highest_eig_third_energy"] for row in method_rows])
        delta_mean, delta_std = _mean_std([row["norm_matched_one_step_loss_delta"] for row in method_rows])
        update_norm_mean, update_norm_std = _mean_std([row["update_norm"] for row in method_rows])

        aggregate_rows.append({
            "method": method_name,
            "n_seeds": len(method_rows),
            "curvature_weighted_alignment_mean": cwa_mean,
            "curvature_weighted_alignment_std": cwa_std,
            "lowest_eig_third_energy_mean": low_mean,
            "lowest_eig_third_energy_std": low_std,
            "middle_eig_third_energy_mean": mid_mean,
            "middle_eig_third_energy_std": mid_std,
            "highest_eig_third_energy_mean": high_mean,
            "highest_eig_third_energy_std": high_std,
            "norm_matched_one_step_loss_delta_mean": delta_mean,
            "norm_matched_one_step_loss_delta_std": delta_std,
            "update_norm_mean": update_norm_mean,
            "update_norm_std": update_norm_std,
        })

    aggregate_lookup = {row["method"]: row for row in aggregate_rows}
    sv_partial_low = aggregate_lookup["SV_partial_k2"]["lowest_eig_third_energy_mean"]
    sgd_low = aggregate_lookup["SGD"]["lowest_eig_third_energy_mean"]
    hessian_partial_low = aggregate_lookup["Hessian_partial_k10"]["lowest_eig_third_energy_mean"]

    prediction_checks = {
        "sv_partial_increases_lowest_eig_third_vs_sgd_by_0p02": bool(sv_partial_low > sgd_low + 0.02),
        "hessian_partial_has_less_lowest_eig_third_energy_than_sv_partial": bool(hessian_partial_low < sv_partial_low),
    }
    prediction_checks["original_prediction_supported"] = bool(
        prediction_checks["sv_partial_increases_lowest_eig_third_vs_sgd_by_0p02"]
        and prediction_checks["hessian_partial_has_less_lowest_eig_third_energy_than_sv_partial"]
    )

    diagnostics = {
        "energy_sum_max_abs_error": float(max(abs(row["energy_sum"] - 1.0) for row in raw_rows)),
        "sv_full_equals_muon_all_seeds": bool(all(row["sv_full_equals_muon"] for row in seed_rows)),
        "sv_full_minus_muon_max_abs_all_seeds": float(max(row["sv_full_minus_muon_max_abs"] for row in seed_rows)),
        "max_hessian_symmetry_residual": float(max(row["hessian_symmetry_residual"] for row in seed_rows)),
    }

    runtime_seconds = float(time.perf_counter() - start_time)

    return {
        "experiment": "H19b_HESSIAN_VS_SV_BASIS_ALIGNMENT",
        "config": {
            "dim": DIM,
            "n_layers": N_LAYERS,
            "n_params": N_PARAMS,
            "warmup_steps": WARMUP_STEPS,
            "num_seeds": NUM_SEEDS,
            "seeds": seeds,
            "learning_rate": LR,
            "momentum": MOMENTUM,
            "fd_eps": FD_EPS,
            "methods": list(METHOD_ORDER),
            "method_descriptions": METHOD_DESCRIPTIONS,
            "energy_sector_labels": ENERGY_SECTOR_LABELS,
            "one_step_loss_metric": "loss(theta - LR * normalized(update, ||g||)) - loss(theta)",
            "notes": [
                "Toy single-point probe after warm-starting; not an end-to-end training study.",
                "Energy sectors use algebraic Hessian eig-order, not |lambda| bins.",
                "SV_full_k4 is exactly identical to Muon in this square 4x4 setup.",
            ],
        },
        "seed_rows": seed_rows,
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_rows,
        "diagnostics": diagnostics,
        "prediction_checks": prediction_checks,
        "runtime_seconds": runtime_seconds,
    }


def _print_cli_summary(results):
    config = results["config"]

    print("=" * 110)
    print("H19b: Hessian-basis vs SV-basis partial equalization")
    print("Toy geometric probe at a single warm-start point; not a mechanistic proof.")
    print("=" * 110)
    print(
        f"Network: {config['n_layers']}-layer deep linear {config['dim']}x{config['dim']} "
        f"({config['n_params']} params)"
    )
    print(f"Seeds: {config['seeds']}")
    print(f"Warmup: {config['warmup_steps']} momentum-SGD steps with lr={config['learning_rate']} and momentum={config['momentum']}")
    print(f"Finite-difference Hessian eps: {config['fd_eps']}")
    print(f"Runtime: {results['runtime_seconds']:.3f} s")
    print("\nImportant caveats:")
    for note in config["notes"]:
        print(f"  - {note}")

    print("\nPer-seed probe diagnostics:")
    for row in results["seed_rows"]:
        print(
            f"  seed={row['seed']}: initial_loss={row['initial_loss']:.6f}, "
            f"probe_loss={row['probe_loss']:.6f}, grad_norm={row['grad_norm']:.6f}, "
            f"eig_min={row['eig_min']:.4e}, eig_max={row['eig_max']:.4e}, "
            f"neg_eigs={row['num_negative_eigvals']}, "
            f"H_sym_resid={row['hessian_symmetry_residual']:.2e}, "
            f"SV_full==Muon={row['sv_full_equals_muon']}"
        )

    print("\nAggregate metrics by method (mean ± std across seeds):")
    header = (
        f"{'Method':<22} {'CWA':>16} {'Lowest eig 1/3':>18} {'Middle eig 1/3':>18} "
        f"{'Highest eig 1/3':>18} {'Norm-matched Δloss':>20}"
    )
    print(header)
    print("-" * len(header))
    for row in results["aggregate_rows"]:
        print(
            f"{row['method']:<22} "
            f"{row['curvature_weighted_alignment_mean']:.4f}±{row['curvature_weighted_alignment_std']:.4f} "
            f"{row['lowest_eig_third_energy_mean']:.3f}±{row['lowest_eig_third_energy_std']:.3f} "
            f"{row['middle_eig_third_energy_mean']:.3f}±{row['middle_eig_third_energy_std']:.3f} "
            f"{row['highest_eig_third_energy_mean']:.3f}±{row['highest_eig_third_energy_std']:.3f} "
            f"{row['norm_matched_one_step_loss_delta_mean']:.6f}±{row['norm_matched_one_step_loss_delta_std']:.6f}"
        )

    checks = results["prediction_checks"]
    print("\nOriginal-prediction checks, stated honestly against the current implementation:")
    print(
        "  - Does SV_partial_k2 raise energy in the lowest algebraic eig-third vs SGD by > 0.02? "
        f"{checks['sv_partial_increases_lowest_eig_third_vs_sgd_by_0p02']}"
    )
    print(
        "  - Does Hessian_partial_k10 place less energy there than SV_partial_k2? "
        f"{checks['hessian_partial_has_less_lowest_eig_third_energy_than_sv_partial']}"
    )
    print(f"  - Do both hold simultaneously? {checks['original_prediction_supported']}")

    diagnostics = results["diagnostics"]
    print("\nDiagnostics:")
    print(f"  - Max |energy_sum - 1| across all seed/method rows: {diagnostics['energy_sum_max_abs_error']:.3e}")
    print(
        "  - SV_full_k4 identical to Muon across all seeds: "
        f"{diagnostics['sv_full_equals_muon_all_seeds']} "
        f"(max abs diff {diagnostics['sv_full_minus_muon_max_abs_all_seeds']:.3e})"
    )
    print(f"  - Max Hessian symmetry residual: {diagnostics['max_hessian_symmetry_residual']:.3e}")


def main():
    results = run_experiment(verbose=False)
    _print_cli_summary(results)


if __name__ == "__main__":
    main()
