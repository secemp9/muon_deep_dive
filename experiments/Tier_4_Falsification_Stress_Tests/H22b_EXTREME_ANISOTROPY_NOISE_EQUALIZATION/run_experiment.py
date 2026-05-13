#!/usr/bin/env python3
"""
H22b: Extreme Anisotropy -- Effective-Rank Tail Proxy vs Muon Advantage
========================================================================

This file implements a deterministic full-batch toy study of Muon under extreme
input anisotropy. It preserves the original single-layer linear-regression setup
but narrows the interpretation to match what is actually computed.

What this experiment *does* measure
-----------------------------------
- How the initial gradient spectrum changes as the synthetic input spectrum is
  made more anisotropic via a data-side condition parameter kappa.
- An effective-rank-based proxy for how much of Muon's equalized update would be
  allocated to the low-energy singular-value tail if one rounds the entropy
  effective rank of the initial gradient and treats the remaining directions as
  the equalized complement.
- Whether that proxy correlates with Muon's advantage over SGD in this toy model.
- Whether a fixed-rank Muon-clip variant helps at the largest tested kappa.

What this experiment does *not* measure
---------------------------------------
- No stochastic minibatch noise, explicit label noise, or repeated-gradient
  variance decomposition is introduced.
- The reported "tail fraction" / "noise proxy" is therefore *not* a direct
  empirical measurement of stochastic noise singular vectors.
- Muon-clip is *not* adaptive per step in this file. A single k is chosen once
  per kappa from the mean rounded effective rank of the initial gradient over
  the diagnostic seeds.
"""

import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

DIM = 32
NUM_STEPS = 500
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

KAPPA_VALUES = [1, 10, 100, 1000, 10000]

LR_SGD = np.logspace(-4, -1, 12)
LR_MUON = np.logspace(-4, -1, 12)
LR_CLIP = np.logspace(-4, -1, 12)

BASE_SEED = 42
SEED_STRIDE = 137
INIT_SEED_OFFSET = 5000
TAIL_PROXY_T1_THRESHOLD = -0.3
CLIP_HELP_FACTOR = 1.2


def newton_schulz(M, n_iters=NS_ITERS):
    norm = np.linalg.norm(M, "fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def effective_rank(M):
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv**2
    sv2 = sv2 / (np.sum(sv2) + 1e-30)
    sv2 = sv2[sv2 > 1e-30]
    entropy = -np.sum(sv2 * np.log(sv2))
    return float(np.exp(entropy))


def make_anisotropic_data(kappa, seed, dim=DIM, batch_size=BATCH_SIZE):
    """
    Create synthetic linear-regression data with a prescribed input singular-value
    envelope. This shapes the input geometry but does not imply an exact gradient
    condition number for every realized seed.
    """
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(dim, dim))
    sigmas = np.logspace(0, -np.log10(kappa), dim)
    X = U @ np.diag(sigmas) @ rng.randn(dim, batch_size)

    W_target = rng.randn(dim, dim) * 0.5
    Y = W_target @ X
    return X, Y


def initial_gradient(kappa, seed, dim=DIM, batch_size=BATCH_SIZE):
    X, Y = make_anisotropic_data(kappa, seed, dim=dim, batch_size=batch_size)
    rng = np.random.RandomState(seed + INIT_SEED_OFFSET)
    W = np.eye(dim) + rng.randn(dim, dim) * 0.1
    G = ((W @ X - Y) @ X.T) / X.shape[1]
    return X, Y, G


def tail_proxy_analysis(G):
    """
    Effective-rank-based proxy for the equalized low-energy tail.

    Procedure:
    - compute the entropy effective rank of G
    - round it to an integer k
    - treat the top-k singular directions as the retained signal proxy
    - treat the remaining directions as the equalized tail proxy

    This is not a direct stochastic signal/noise decomposition of ortho(G).
    """
    sigma = np.linalg.svd(G, compute_uv=False)
    er_continuous = effective_rank(G)
    er_rounded = int(np.round(er_continuous))
    er_rounded = max(1, min(er_rounded, len(sigma)))

    signal_fraction_proxy = er_rounded / len(sigma)
    tail_fraction_proxy = 1.0 - signal_fraction_proxy
    gradient_signal_proxy = float(np.sum(sigma[:er_rounded] ** 2) / (np.sum(sigma**2) + 1e-30))

    return {
        "effective_rank_continuous": float(er_continuous),
        "effective_rank_rounded": int(er_rounded),
        "signal_fraction_proxy": float(signal_fraction_proxy),
        "tail_fraction_proxy": float(tail_fraction_proxy),
        "gradient_signal_proxy": gradient_signal_proxy,
    }


def noise_fraction_analysis(G):
    """Backward-compatible alias for the effective-rank tail proxy diagnostic."""
    proxy = tail_proxy_analysis(G)
    return (
        proxy["signal_fraction_proxy"],
        proxy["tail_fraction_proxy"],
        proxy["effective_rank_rounded"],
        proxy["gradient_signal_proxy"],
    )


def muon_clip_step(G, k, ns_iters=NS_ITERS):
    """
    Fixed-rank Muon-clip: zero out the bottom singular values of G before
    orthogonalization, then apply Newton-Schulz to the truncated matrix.
    """
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    k = max(1, min(int(k), len(sigma)))
    sigma_clip = sigma.copy()
    sigma_clip[k:] = 0
    G_clip = U @ np.diag(sigma_clip) @ Vt
    return newton_schulz(G_clip, n_iters=ns_iters)


def train(X, Y, lr, opt, seed, num_steps=NUM_STEPS, momentum=MOMENTUM, ns_iters=NS_ITERS):
    dim = X.shape[0]
    rng = np.random.RandomState(seed)
    W = np.eye(dim) + rng.randn(dim, dim) * 0.1
    mom = np.zeros_like(W)

    for _ in range(num_steps):
        pred = W @ X
        loss = 0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0))
        if not np.isfinite(loss) or loss > 1e10:
            return float("inf")
        G = (pred - Y) @ X.T / X.shape[1]

        if opt == "muon":
            mom = momentum * mom + newton_schulz(G, n_iters=ns_iters)
        elif opt == "sgd":
            mom = momentum * mom + G
        elif opt.startswith("muon_clip_"):
            k = int(opt.split("_")[-1])
            mom = momentum * mom + muon_clip_step(G, k, ns_iters=ns_iters)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        W = W - lr * mom

    pred = W @ X
    return float(0.5 * np.mean(np.sum((pred - Y) ** 2, axis=0)))


def summarize_losses(losses):
    losses = [float(l) for l in losses]
    finite_losses = [l for l in losses if np.isfinite(l)]
    if finite_losses:
        mean_finite_loss = float(np.mean(finite_losses))
        std_finite_loss = float(np.std(finite_losses))
    else:
        mean_finite_loss = float("inf")
        std_finite_loss = float("inf")
    return {
        "losses": losses,
        "finite_count": int(len(finite_losses)),
        "total_count": int(len(losses)),
        "mean_finite_loss": mean_finite_loss,
        "std_finite_loss": std_finite_loss,
    }


def summarize_scalar_records(records, keys):
    means = {}
    stds = {}
    for key in keys:
        values = np.array([record[key] for record in records], dtype=float)
        means[key] = float(np.mean(values))
        stds[key] = float(np.std(values))
    return means, stds


def paired_advantages(sgd_losses, other_losses):
    ratios = []
    for sgd_loss, other_loss in zip(sgd_losses, other_losses):
        if np.isfinite(sgd_loss) and np.isfinite(other_loss):
            ratios.append(float(sgd_loss / max(other_loss, 1e-30)))
        else:
            ratios.append(float("inf"))
    return ratios


def lr_sweep_for_optimizer(
    kappa,
    opt_name,
    grid,
    seeds,
    *,
    dim,
    batch_size,
    num_steps,
    momentum,
    ns_iters,
):
    records = []
    best_lr = float(grid[-1])
    best_loss = float("inf")

    for lr in grid:
        losses = [
            train(
                *make_anisotropic_data(kappa, s, dim=dim, batch_size=batch_size),
                float(lr),
                opt_name,
                s + INIT_SEED_OFFSET,
                num_steps=num_steps,
                momentum=momentum,
                ns_iters=ns_iters,
            )
            for s in seeds
        ]
        summary = summarize_losses(losses)
        record = {
            "lr": float(lr),
            **summary,
        }
        records.append(record)
        if record["mean_finite_loss"] < best_loss:
            best_loss = record["mean_finite_loss"]
            best_lr = float(lr)

    return {
        "grid": [float(lr) for lr in grid],
        "records": records,
        "best_lr": float(best_lr),
        "best_mean_finite_loss": float(best_loss),
    }


def run_experiment(
    *,
    dim=DIM,
    num_steps=NUM_STEPS,
    momentum=MOMENTUM,
    ns_iters=NS_ITERS,
    num_seeds=NUM_SEEDS,
    batch_size=BATCH_SIZE,
    kappa_values=None,
    lr_sgd=None,
    lr_muon=None,
    lr_clip=None,
    diagnostic_seed_count=3,
    lr_search_seed_count=3,
    verbose=True,
):
    start_time = time.time()

    kappa_values = list(KAPPA_VALUES if kappa_values is None else kappa_values)
    lr_sgd = np.array(LR_SGD if lr_sgd is None else lr_sgd, dtype=float)
    lr_muon = np.array(LR_MUON if lr_muon is None else lr_muon, dtype=float)
    lr_clip = np.array(LR_CLIP if lr_clip is None else lr_clip, dtype=float)

    seeds = [BASE_SEED + i * SEED_STRIDE for i in range(num_seeds)]
    diagnostic_seeds = seeds[: min(diagnostic_seed_count, len(seeds))]
    lr_search_seeds = seeds[: min(lr_search_seed_count, len(seeds))]

    tuning_train_calls = len(kappa_values) * (len(lr_sgd) + len(lr_muon) + len(lr_clip)) * len(lr_search_seeds)
    final_eval_calls = len(kappa_values) * 3 * len(seeds)

    config = {
        "dim": int(dim),
        "num_steps": int(num_steps),
        "momentum": float(momentum),
        "ns_iters": int(ns_iters),
        "num_seeds": int(num_seeds),
        "batch_size": int(batch_size),
        "base_seed": int(BASE_SEED),
        "seed_stride": int(SEED_STRIDE),
        "init_seed_offset": int(INIT_SEED_OFFSET),
        "kappa_values": [float(k) for k in kappa_values],
        "lr_grids": {
            "sgd": [float(lr) for lr in lr_sgd],
            "muon": [float(lr) for lr in lr_muon],
            "muon_clip": [float(lr) for lr in lr_clip],
        },
        "seeds": [int(s) for s in seeds],
        "diagnostic_seeds": [int(s) for s in diagnostic_seeds],
        "lr_search_seeds": [int(s) for s in lr_search_seeds],
        "diagnostic_seed_count": int(len(diagnostic_seeds)),
        "lr_search_seed_count": int(len(lr_search_seeds)),
        "tuning_train_calls": int(tuning_train_calls),
        "final_eval_calls": int(final_eval_calls),
        "total_train_calls": int(tuning_train_calls + final_eval_calls),
    }

    if verbose:
        print("=" * 100)
        print("H22b: EXTREME ANISOTROPY -- EFFECTIVE-RANK TAIL PROXY vs MUON ADVANTAGE")
        print("=" * 100)
        print("Deterministic full-batch linear-regression toy study")
        print("Tail fraction is an effective-rank proxy, not a direct stochastic-noise measurement")
        print(f"kappa values: {kappa_values}")
        print(f"dim={dim}, batch_size={batch_size}, num_steps={num_steps}, num_seeds={num_seeds}")
        print(
            f"expected train() calls: tuning={tuning_train_calls}, final_eval={final_eval_calls}, total={tuning_train_calls + final_eval_calls}"
        )

    kappa_results = []
    summary_rows = []

    diagnostic_keys = [
        "effective_rank_continuous",
        "effective_rank_rounded",
        "signal_fraction_proxy",
        "tail_fraction_proxy",
        "gradient_signal_proxy",
        "grad_sigma_max",
        "grad_sigma_min",
        "grad_condition_proxy",
    ]

    for kappa in kappa_values:
        if verbose:
            print(f"\n  kappa={kappa}")

        diagnostic_records = []
        for seed in diagnostic_seeds:
            _, _, G = initial_gradient(kappa, seed, dim=dim, batch_size=batch_size)
            sv = np.linalg.svd(G, compute_uv=False)
            proxy = tail_proxy_analysis(G)
            diagnostic_records.append(
                {
                    "seed": int(seed),
                    "grad_sigma_max": float(sv[0]),
                    "grad_sigma_min": float(sv[-1]),
                    "grad_condition_proxy": float(sv[0] / max(sv[-1], 1e-15)),
                    **proxy,
                }
            )

        diagnostic_means, diagnostic_stds = summarize_scalar_records(diagnostic_records, diagnostic_keys)
        k_clip = max(1, int(np.round(np.mean([r["effective_rank_rounded"] for r in diagnostic_records]))))

        if verbose:
            print(
                "    initial-gradient diagnostics: "
                f"eff_rank≈{diagnostic_means['effective_rank_continuous']:.2f}, "
                f"rounded_eff_rank≈{diagnostic_means['effective_rank_rounded']:.2f}, "
                f"tail_proxy≈{diagnostic_means['tail_fraction_proxy']:.3f}, "
                f"grad_signal_proxy≈{diagnostic_means['gradient_signal_proxy']:.3f}, "
                f"k_clip={k_clip}"
            )

        optimizer_specs = [
            ("sgd", lr_sgd),
            ("muon", lr_muon),
            (f"muon_clip_{k_clip}", lr_clip),
        ]

        lr_search = {}
        best_lrs = {}
        for opt_name, grid in optimizer_specs:
            sweep = lr_sweep_for_optimizer(
                kappa,
                opt_name,
                grid,
                lr_search_seeds,
                dim=dim,
                batch_size=batch_size,
                num_steps=num_steps,
                momentum=momentum,
                ns_iters=ns_iters,
            )
            lr_search[opt_name] = sweep
            best_lrs[opt_name] = sweep["best_lr"]

        final_eval = {}
        for opt_name, _ in optimizer_specs:
            losses = [
                train(
                    *make_anisotropic_data(kappa, s, dim=dim, batch_size=batch_size),
                    best_lrs[opt_name],
                    opt_name,
                    s + INIT_SEED_OFFSET,
                    num_steps=num_steps,
                    momentum=momentum,
                    ns_iters=ns_iters,
                )
                for s in seeds
            ]
            final_eval[opt_name] = {
                "best_lr": float(best_lrs[opt_name]),
                **summarize_losses(losses),
            }

        muon_clip_key = f"muon_clip_{k_clip}"
        muon_advantage = float(
            final_eval["sgd"]["mean_finite_loss"] / max(final_eval["muon"]["mean_finite_loss"], 1e-30)
        )
        muon_clip_advantage = float(
            final_eval["sgd"]["mean_finite_loss"] / max(final_eval[muon_clip_key]["mean_finite_loss"], 1e-30)
        )

        pairwise_advantages = {
            "muon": paired_advantages(final_eval["sgd"]["losses"], final_eval["muon"]["losses"]),
            "muon_clip": paired_advantages(final_eval["sgd"]["losses"], final_eval[muon_clip_key]["losses"]),
        }

        kappa_result = {
            "kappa": float(kappa),
            "diagnostics": {
                "records": diagnostic_records,
                "means": diagnostic_means,
                "stds": diagnostic_stds,
            },
            "k_clip": int(k_clip),
            "muon_clip_key": muon_clip_key,
            "best_lrs": {name: float(lr) for name, lr in best_lrs.items()},
            "lr_search": lr_search,
            "final_eval": final_eval,
            "advantages": {
                "muon_advantage": muon_advantage,
                "muon_clip_advantage": muon_clip_advantage,
                "pairwise_muon_advantages": pairwise_advantages["muon"],
                "pairwise_muon_clip_advantages": pairwise_advantages["muon_clip"],
            },
        }
        kappa_results.append(kappa_result)

        summary_rows.append(
            {
                "kappa": float(kappa),
                "effective_rank_continuous": diagnostic_means["effective_rank_continuous"],
                "effective_rank_rounded": diagnostic_means["effective_rank_rounded"],
                "tail_fraction_proxy": diagnostic_means["tail_fraction_proxy"],
                "gradient_signal_proxy": diagnostic_means["gradient_signal_proxy"],
                "k_clip": int(k_clip),
                "sgd_mean_loss": final_eval["sgd"]["mean_finite_loss"],
                "muon_mean_loss": final_eval["muon"]["mean_finite_loss"],
                "muon_clip_mean_loss": final_eval[muon_clip_key]["mean_finite_loss"],
                "muon_advantage": muon_advantage,
                "muon_clip_advantage": muon_clip_advantage,
            }
        )

        if verbose:
            print(
                f"    final mean losses: SGD={final_eval['sgd']['mean_finite_loss']:.6g} "
                f"({final_eval['sgd']['finite_count']}/{final_eval['sgd']['total_count']} finite), "
                f"Muon={final_eval['muon']['mean_finite_loss']:.6g} "
                f"({final_eval['muon']['finite_count']}/{final_eval['muon']['total_count']} finite), "
                f"Muon-clip={final_eval[muon_clip_key]['mean_finite_loss']:.6g} "
                f"({final_eval[muon_clip_key]['finite_count']}/{final_eval[muon_clip_key]['total_count']} finite)"
            )
            print(
                f"    advantages: Muon={muon_advantage:.2f}x, "
                f"Muon-clip(k={k_clip})={muon_clip_advantage:.2f}x"
            )

    tail_fracs = [row["tail_fraction_proxy"] for row in summary_rows]
    muon_advantages = [row["muon_advantage"] for row in summary_rows]
    if len(tail_fracs) >= 2 and np.std(tail_fracs) > 0 and np.all(np.isfinite(muon_advantages)):
        tail_proxy_corr = float(np.corrcoef(tail_fracs, np.log(np.clip(muon_advantages, 1e-10, None)))[0, 1])
    else:
        tail_proxy_corr = float("nan")

    high_kappa_result = kappa_results[-1]
    clip_helps = bool(
        high_kappa_result["advantages"]["muon_clip_advantage"]
        > high_kappa_result["advantages"]["muon_advantage"] * CLIP_HELP_FACTOR
    )
    t1_pass = bool(np.isfinite(tail_proxy_corr) and tail_proxy_corr < TAIL_PROXY_T1_THRESHOLD)

    runtime_seconds = float(time.time() - start_time)
    results = {
        "study_id": "H22b_EXTREME_ANISOTROPY_NOISE_EQUALIZATION",
        "title": "Extreme anisotropy tail-proxy study for Muon",
        "scope": {
            "deterministic_full_batch": True,
            "direct_stochastic_noise_measurement": False,
            "tail_fraction_is_proxy": True,
            "muon_clip_fixed_k_per_kappa": True,
        },
        "config": config,
        "summary_rows": summary_rows,
        "kappa_results": kappa_results,
        "correlations": {
            "tail_fraction_proxy_vs_log_muon_advantage": tail_proxy_corr,
        },
        "tests": {
            "T1": {
                "description": "Higher tail-fraction proxy should correlate with lower Muon advantage",
                "threshold": float(TAIL_PROXY_T1_THRESHOLD),
                "observed_r": tail_proxy_corr,
                "pass": t1_pass,
            },
            "T2": {
                "description": f"Muon-clip should beat Muon by more than {(CLIP_HELP_FACTOR - 1.0) * 100:.0f}% at highest kappa",
                "high_kappa": float(high_kappa_result["kappa"]),
                "required_factor": float(CLIP_HELP_FACTOR),
                "muon_advantage": high_kappa_result["advantages"]["muon_advantage"],
                "muon_clip_advantage": high_kappa_result["advantages"]["muon_clip_advantage"],
                "pass": clip_helps,
            },
        },
        "runtime_seconds": runtime_seconds,
    }

    if verbose:
        print_summary(results)

    return results


def print_summary(results):
    config = results["config"]
    print(f"\n\n{'=' * 100}")
    print("RESULTS: EFFECTIVE-RANK TAIL PROXY vs MUON ADVANTAGE")
    print(f"{'=' * 100}")
    print("Tail fraction is a proxy based on rounded effective rank of the initial gradient.")
    print("It is not a direct decomposition of stochastic signal and noise in ortho(G).")

    header = (
        f"\n  {'kappa':>8}  {'tail%':>8}  {'eff.rank':>10}  {'k_clip':>8}  "
        f"{'Muon adv':>10}  {'Clip adv':>10}  {'clip>muon?':>12}"
    )
    print(header)
    print("  " + "-" * 76)
    for row in results["summary_rows"]:
        clip_better = "YES" if row["muon_clip_advantage"] > row["muon_advantage"] * CLIP_HELP_FACTOR else "NO"
        print(
            f"  {int(row['kappa']):>8}  {row['tail_fraction_proxy'] * 100:>7.1f}%  "
            f"{row['effective_rank_continuous']:>10.2f}  {row['k_clip']:>8d}  "
            f"{row['muon_advantage']:>10.1f}x  {row['muon_clip_advantage']:>10.1f}x  {clip_better:>12}"
        )

    tail_corr = results["correlations"]["tail_fraction_proxy_vs_log_muon_advantage"]
    print(f"\n  Correlation(tail_fraction_proxy, log(Muon advantage)): r = {tail_corr:.3f}")

    t1 = results["tests"]["T1"]
    t2 = results["tests"]["T2"]
    print(
        f"\n  T1: Higher tail proxy -> lower Muon advantage?  --> {'PASS' if t1['pass'] else 'FAIL'}  "
        f"(r={t1['observed_r']:.3f}, threshold={t1['threshold']})"
    )
    print(
        f"  T2: Fixed-k Muon-clip helps at kappa={int(t2['high_kappa'])}?  --> {'PASS' if t2['pass'] else 'FAIL'}  "
        f"(Muon={t2['muon_advantage']:.2f}x, Clip={t2['muon_clip_advantage']:.2f}x)"
    )

    print(f"\n  Runtime: {results['runtime_seconds']:.2f}s")
    print(f"  Train calls: {config['total_train_calls']} total")
    print(f"{'=' * 100}")


def main():
    results = run_experiment(verbose=True)
    return results


if __name__ == "__main__":
    main()
