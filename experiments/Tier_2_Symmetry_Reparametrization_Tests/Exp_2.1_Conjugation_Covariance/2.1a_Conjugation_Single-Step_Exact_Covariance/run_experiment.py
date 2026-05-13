#!/usr/bin/env python3
"""
Exp 2.1a: Conjugation Single-Step Exact Covariance
==================================================

Sampled numerical verification that the zero-momentum single Muon update,
implemented with a finite Newton-Schulz iterate, is equivariant under
orthogonal conjugation:

    muon_step(R W S^T, R G S^T) = R muon_step(W, G) S^T

for orthogonal R and S.

This script preserves the original toy setup:
  - sizes 4x4 and 8x8
  - 100 sampled orthogonal trials per size
  - a non-orthogonal control at 4x4
  - a Newton-Schulz iteration sensitivity sweep

It is a sampled numerical verification, not a proof over the full orthogonal
group, and it does not test momentum or multi-step training dynamics.
"""

import sys
import time
from pathlib import Path
import numpy as np


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

SIZES = [(4, 4), (8, 8)]
N_TRIALS = 100
LR = 0.02
NS_ITERS = 5
BASE_SEED = 42

NONORTHOGONAL_CONTROL_SIZE = (4, 4)
NONORTHOGONAL_SEED_OFFSET = 999
NONORTHOGONAL_PERTURBATION_SCALE = 0.5

NS_SENSITIVITY_ITERS = [1, 3, 5, 10, 20]
NS_SENSITIVITY_TRIALS = 50

ERROR_THRESHOLDS = [1e-15, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8]
LOG10_HISTOGRAM_BINS = [-18, -16, -15, -14, -13, -12, -10, -8, -5, 0]


# =============================================================================
# CORE LINEAR-ALGEBRA HELPERS
# =============================================================================


def size_label(size):
    return f"{size[0]}x{size[1]}"


def summarize_errors(errors):
    """Return scalar summary statistics for a 1D error array."""
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(errors.size),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "q05": float(np.quantile(errors, 0.05)),
        "q95": float(np.quantile(errors, 0.95)),
    }


def threshold_counts(errors, thresholds=ERROR_THRESHOLDS):
    """Count how many errors fall below each threshold."""
    errors = np.asarray(errors, dtype=np.float64)
    return {
        f"{threshold:.0e}": int(np.sum(errors < threshold))
        for threshold in thresholds
    }


def log10_histogram(errors, bins=LOG10_HISTOGRAM_BINS):
    """Histogram of log10(error) using fixed bins for reporting parity."""
    errors = np.asarray(errors, dtype=np.float64)
    log_errors = np.log10(errors + 1e-20)
    entries = []
    for left, right in zip(bins[:-1], bins[1:]):
        count = int(np.sum((log_errors >= left) & (log_errors < right)))
        entries.append({
            "left": float(left),
            "right": float(right),
            "count": count,
        })
    return {
        "bins": [float(x) for x in bins],
        "entries": entries,
    }


# =============================================================================
# FINITE NEWTON-SCHULZ ITERATION
# =============================================================================


def newton_schulz(M, n_iters=NS_ITERS):
    """
    Return the finite Newton-Schulz iterate after Frobenius rescaling.

    This is the map actually tested in the experiment. For finite iteration
    counts it need not have converged fully to the exact polar factor.
    """
    norm = np.linalg.norm(M, ord="fro")
    if norm < 1e-15:
        return M
    X = M / norm
    for _ in range(n_iters):
        A = X.T @ X
        X = 1.5 * X - 0.5 * X @ A
    return X


def muon_step(W, G, lr=LR, ns_iters=NS_ITERS):
    """Single zero-momentum Muon step: W_new = W - lr * NS_n(G)."""
    Q = newton_schulz(G, n_iters=ns_iters)
    return W - lr * Q


# =============================================================================
# RANDOM MATRICES AND TRIALS
# =============================================================================


def random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via QR decomposition."""
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    D = np.diag(signs)
    return Q @ D


def relative_equivariance_error(W, G, R, S, lr=LR, ns_iters=NS_ITERS):
    """
    Compare 'step then rotate' versus 'rotate then step'.

    Returns relative Frobenius error
        ||W1' - R W1 S^T|| / ||W1||.
    """
    W1 = muon_step(W, G, lr=lr, ns_iters=ns_iters)
    W1_rotated = R @ W1 @ S.T

    W_rot = R @ W @ S.T
    G_rot = R @ G @ S.T
    W1_prime = muon_step(W_rot, G_rot, lr=lr, ns_iters=ns_iters)

    diff = W1_prime - W1_rotated
    return np.linalg.norm(diff) / max(np.linalg.norm(W1), 1e-30)


def run_trial(m, n, rng, lr=LR, ns_iters=NS_ITERS):
    """Run one sampled orthogonal-conjugation trial."""
    W = rng.randn(m, n)
    G = rng.randn(m, n)
    R = random_orthogonal(m, rng)
    S = random_orthogonal(n, rng)
    return relative_equivariance_error(W, G, R, S, lr=lr, ns_iters=ns_iters)


# =============================================================================
# EXPERIMENT COMPONENTS
# =============================================================================


def run_orthogonal_trials(sizes=SIZES, n_trials=N_TRIALS, base_seed=BASE_SEED,
                          lr=LR, ns_iters=NS_ITERS):
    """
    Run the sampled orthogonal-conjugation trials.

    Legacy behavior is preserved exactly: the RNG is reinitialized to
    RandomState(BASE_SEED) separately for each tested size.
    """
    results = []
    for size in sizes:
        m, n = size
        rng = np.random.RandomState(base_seed)
        errors = np.array([
            run_trial(m, n, rng, lr=lr, ns_iters=ns_iters)
            for _ in range(n_trials)
        ], dtype=np.float64)
        results.append({
            "size": (int(m), int(n)),
            "size_label": size_label(size),
            "trial_count": int(n_trials),
            "rng_seed": int(base_seed),
            "errors": errors,
            "summary": summarize_errors(errors),
            "threshold_counts": threshold_counts(errors),
            "log10_histogram": log10_histogram(errors),
        })
    return results


def run_nonorthogonal_control(n_trials=N_TRIALS, base_seed=BASE_SEED, lr=LR,
                              ns_iters=NS_ITERS,
                              size=NONORTHOGONAL_CONTROL_SIZE,
                              seed_offset=NONORTHOGONAL_SEED_OFFSET,
                              perturbation_scale=NONORTHOGONAL_PERTURBATION_SCALE):
    """Run the non-orthogonal control, which should break equivariance."""
    m, n = size
    rng_seed = base_seed + seed_offset
    rng = np.random.RandomState(rng_seed)
    errors = []

    for _ in range(n_trials):
        W = rng.randn(m, n)
        G = rng.randn(m, n)
        R = rng.randn(m, m) * perturbation_scale + np.eye(m)
        S = rng.randn(n, n) * perturbation_scale + np.eye(n)
        err = relative_equivariance_error(W, G, R, S, lr=lr, ns_iters=ns_iters)
        errors.append(err)

    errors = np.array(errors, dtype=np.float64)
    return {
        "size": (int(m), int(n)),
        "size_label": size_label(size),
        "trial_count": int(n_trials),
        "rng_seed": int(rng_seed),
        "perturbation_scale": float(perturbation_scale),
        "description": "Non-orthogonal control with R = I + 0.5*N and S = I + 0.5*N.",
        "errors": errors,
        "summary": summarize_errors(errors),
        "threshold_counts": threshold_counts(errors),
        "log10_histogram": log10_histogram(errors),
    }


def run_ns_sensitivity(ns_iter_values=NS_SENSITIVITY_ITERS,
                       n_trials=NS_SENSITIVITY_TRIALS,
                       base_seed=BASE_SEED,
                       lr=LR,
                       size=NONORTHOGONAL_CONTROL_SIZE):
    """Sweep Newton-Schulz iteration count while holding the toy test fixed."""
    m, n = size
    results = []

    for ns_iter in ns_iter_values:
        rng_seed = base_seed + ns_iter
        rng = np.random.RandomState(rng_seed)
        errors = np.array([
            run_trial(m, n, rng, lr=lr, ns_iters=ns_iter)
            for _ in range(n_trials)
        ], dtype=np.float64)
        results.append({
            "ns_iters": int(ns_iter),
            "size": (int(m), int(n)),
            "size_label": size_label(size),
            "trial_count": int(n_trials),
            "rng_seed": int(rng_seed),
            "errors": errors,
            "summary": summarize_errors(errors),
            "threshold_counts": threshold_counts(errors),
            "log10_histogram": log10_histogram(errors),
        })

    return results


def evaluate_hypotheses(orthogonal_results, orthogonal_aggregate,
                        nonorthogonal_control):
    """Evaluate H1-H4 using the original thresholds."""
    per_size_max = {
        entry["size_label"]: float(entry["summary"]["max"])
        for entry in orthogonal_results
    }

    max_orthogonal_error = float(orthogonal_aggregate["summary"]["max"])
    mean_orthogonal_error = float(orthogonal_aggregate["summary"]["mean"])
    mean_nonorthogonal_error = float(nonorthogonal_control["summary"]["mean"])

    h1_passed = max_orthogonal_error < 1e-12
    h2_passed = mean_nonorthogonal_error > 0.01
    h3_passed = all(max_error < 1e-12 for max_error in per_size_max.values())
    h4_passed = mean_orthogonal_error < 1e-13

    return {
        "H1": {
            "statement": "All sampled orthogonal errors are below 1e-12.",
            "criterion_text": "max orthogonal error < 1e-12",
            "observed": {"max_error": max_orthogonal_error},
            "observed_text": f"max orthogonal error = {max_orthogonal_error:.2e}",
            "passed": bool(h1_passed),
        },
        "H2": {
            "statement": "The non-orthogonal control has mean error above 0.01.",
            "criterion_text": "mean non-orthogonal control error > 0.01",
            "observed": {"mean_error": mean_nonorthogonal_error},
            "observed_text": f"mean non-orthogonal error = {mean_nonorthogonal_error:.2e}",
            "passed": bool(h2_passed),
        },
        "H3": {
            "statement": "Each tested size has max orthogonal error below 1e-12.",
            "criterion_text": "per-size max orthogonal error < 1e-12",
            "observed": per_size_max,
            "observed_text": ", ".join(
                f"{label}: {value:.2e}" for label, value in per_size_max.items()
            ),
            "passed": bool(h3_passed),
        },
        "H4": {
            "statement": "The aggregate mean orthogonal error is below 1e-13.",
            "criterion_text": "mean orthogonal error < 1e-13",
            "observed": {"mean_error": mean_orthogonal_error},
            "observed_text": f"mean orthogonal error = {mean_orthogonal_error:.2e}",
            "passed": bool(h4_passed),
        },
    }


def build_verdict(hypotheses):
    """Construct a calibrated final verdict."""
    passed = int(sum(entry["passed"] for entry in hypotheses.values()))
    total = int(len(hypotheses))

    if passed == total:
        headline = (
            "All sampled checks passed: the tested zero-momentum single-step "
            "update is numerically consistent with exact orthogonal-conjugation "
            "equivariance at float64 roundoff in this toy setting."
        )
    elif hypotheses["H1"]["passed"]:
        headline = (
            "The sampled orthogonal-conjugation identity passes at the 1e-12 "
            "level, but not every auxiliary check passed."
        )
    else:
        headline = (
            "The sampled orthogonal-conjugation identity did not meet the 1e-12 "
            "equivariance target in this run."
        )

    return {
        "passed": passed,
        "total": total,
        "headline": headline,
        "limitations": (
            "This is a Monte Carlo numerical check over sampled matrices, sizes, "
            "and iteration counts; it is not a proof over the full orthogonal "
            "group and it does not test momentum or multi-step training."
        ),
    }


# =============================================================================
# FULL EXPERIMENT
# =============================================================================


def run_experiment(sizes=SIZES,
                   n_trials=N_TRIALS,
                   lr=LR,
                   ns_iters=NS_ITERS,
                   base_seed=BASE_SEED,
                   nonorthogonal_control_size=NONORTHOGONAL_CONTROL_SIZE,
                   nonorthogonal_seed_offset=NONORTHOGONAL_SEED_OFFSET,
                   nonorthogonal_perturbation_scale=NONORTHOGONAL_PERTURBATION_SCALE,
                   ns_sensitivity_iters=NS_SENSITIVITY_ITERS,
                   ns_sensitivity_trials=NS_SENSITIVITY_TRIALS):
    """Run the full experiment and return structured results."""
    start_time = time.perf_counter()

    orthogonal_results = run_orthogonal_trials(
        sizes=sizes,
        n_trials=n_trials,
        base_seed=base_seed,
        lr=lr,
        ns_iters=ns_iters,
    )

    all_orthogonal_errors = np.concatenate([
        entry["errors"] for entry in orthogonal_results
    ])
    orthogonal_aggregate = {
        "trial_count": int(all_orthogonal_errors.size),
        "errors": all_orthogonal_errors,
        "summary": summarize_errors(all_orthogonal_errors),
        "threshold_counts": threshold_counts(all_orthogonal_errors),
        "log10_histogram": log10_histogram(all_orthogonal_errors),
    }

    nonorthogonal_control = run_nonorthogonal_control(
        n_trials=n_trials,
        base_seed=base_seed,
        lr=lr,
        ns_iters=ns_iters,
        size=nonorthogonal_control_size,
        seed_offset=nonorthogonal_seed_offset,
        perturbation_scale=nonorthogonal_perturbation_scale,
    )

    ns_sensitivity = run_ns_sensitivity(
        ns_iter_values=ns_sensitivity_iters,
        n_trials=ns_sensitivity_trials,
        base_seed=base_seed,
        lr=lr,
        size=nonorthogonal_control_size,
    )

    hypotheses = evaluate_hypotheses(
        orthogonal_results=orthogonal_results,
        orthogonal_aggregate=orthogonal_aggregate,
        nonorthogonal_control=nonorthogonal_control,
    )
    verdict = build_verdict(hypotheses)

    runtime_sec = time.perf_counter() - start_time

    return {
        "experiment_id": "Exp 2.1a",
        "title": "Conjugation Single-Step Exact Covariance",
        "identity_under_test": (
            "muon_step(R W S^T, R G S^T) = R muon_step(W, G) S^T "
            "for orthogonal R and S"
        ),
        "scope": (
            "Sampled numerical verification of orthogonal-conjugation "
            "equivariance for the zero-momentum single Muon step using the "
            "finite Newton-Schulz iterate actually implemented here."
        ),
        "implementation_notes": {
            "map_under_test": (
                "Finite Newton-Schulz iterate after Frobenius rescaling; "
                "not an exact polar factor."
            ),
            "step_variant": (
                "Zero-momentum single-step Muon update only; no momentum "
                "buffer/state is modeled in this experiment."
            ),
            "raw_result_arrays": (
                "Per-trial error collections are returned as NumPy float64 "
                "arrays for notebook/reporting use and are not JSON-ready by "
                "default."
            ),
        },
        "config": {
            "sizes": [tuple(size) for size in sizes],
            "n_trials": int(n_trials),
            "lr": float(lr),
            "ns_iters": int(ns_iters),
            "base_seed": int(base_seed),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "script_path": str(Path(__file__).resolve()),
            "orthogonal_seed_policy": (
                "Legacy behavior preserved: RandomState(BASE_SEED) is "
                "reinitialized separately for each tested size."
            ),
            "orthogonal_trial_seeds": {
                size_label(size): int(base_seed) for size in sizes
            },
            "nonorthogonal_control_size": tuple(nonorthogonal_control_size),
            "nonorthogonal_control_seed": int(base_seed + nonorthogonal_seed_offset),
            "nonorthogonal_perturbation_scale": float(nonorthogonal_perturbation_scale),
            "ns_sensitivity_iters": [int(x) for x in ns_sensitivity_iters],
            "ns_sensitivity_trials": int(ns_sensitivity_trials),
            "ns_sensitivity_seeds": {
                int(ns_iter): int(base_seed + ns_iter)
                for ns_iter in ns_sensitivity_iters
            },
            "error_thresholds": [float(x) for x in ERROR_THRESHOLDS],
            "log10_histogram_bins": [float(x) for x in LOG10_HISTOGRAM_BINS],
            "dtype": "float64",
            "numpy_version": np.__version__,
        },
        "orthogonal_results": orthogonal_results,
        "orthogonal_aggregate": orthogonal_aggregate,
        "nonorthogonal_control": nonorthogonal_control,
        "ns_sensitivity": ns_sensitivity,
        "hypotheses": hypotheses,
        "verdict": verdict,
        "runtime_sec": float(runtime_sec),
    }


# =============================================================================
# CONSOLE REPORTING
# =============================================================================


def print_orthogonal_results(results):
    print("\nMain orthogonal-conjugation trials:")
    for entry in results["orthogonal_results"]:
        summary = entry["summary"]
        print(f"\nSize {entry['size_label']}:")
        print(f"  Trials:                 {entry['trial_count']}")
        print(f"  RNG seed:               {entry['rng_seed']}")
        print(f"  Mean relative error:    {summary['mean']:.2e}")
        print(f"  Max relative error:     {summary['max']:.2e}")
        print(f"  Min relative error:     {summary['min']:.2e}")
        print(f"  Median relative error:  {summary['median']:.2e}")
        print(f"  Std relative error:     {summary['std']:.2e}")
        print(f"  95th percentile:        {summary['q95']:.2e}")


def print_error_distributions(results):
    print(f"\n\n{'=' * 90}")
    print("ERROR DISTRIBUTIONS FOR SAMPLED ORTHOGONAL TRIALS")
    print(f"{'=' * 90}")

    thresholds = results["config"]["error_thresholds"]
    for entry in results["orthogonal_results"]:
        print(f"\n  Size {entry['size_label']}:")
        print("    Log10(error) distribution:")
        for hist_entry in entry["log10_histogram"]["entries"]:
            bar = '#' * hist_entry["count"]
            print(
                f"      [{hist_entry['left']:>4.0f}, {hist_entry['right']:>4.0f}): "
                f"{hist_entry['count']:>4}  {bar}"
            )
        for threshold in thresholds:
            key = f"{threshold:.0e}"
            count = entry["threshold_counts"][key]
            print(f"    Errors < {threshold:.0e}: {count}/{entry['trial_count']}")


def print_nonorthogonal_control(results):
    control = results["nonorthogonal_control"]
    summary = control["summary"]
    print(f"\n\n{'=' * 90}")
    print("CONTROL: NON-ORTHOGONAL CONJUGATION (SHOULD BREAK EQUIVARIANCE)")
    print(f"{'=' * 90}")
    print(f"\n  {control['description']}")
    print(f"  Size:                   {control['size_label']}")
    print(f"  Trials:                 {control['trial_count']}")
    print(f"  RNG seed:               {control['rng_seed']}")
    print(f"  Mean relative error:    {summary['mean']:.2e}")
    print(f"  Max relative error:     {summary['max']:.2e}")
    print(f"  Min relative error:     {summary['min']:.2e}")
    print(f"  95th percentile:        {summary['q95']:.2e}")


def print_ns_sensitivity(results):
    print(f"\n\n{'=' * 90}")
    print("SENSITIVITY: FINITE NEWTON-SCHULZ ITERATION COUNT")
    print(f"{'=' * 90}")
    print("  Equivariance is expected for each tested iteration count because the")
    print("  finite Newton-Schulz map itself is conjugation-equivariant at every step.")
    print()
    for entry in results["ns_sensitivity"]:
        summary = entry["summary"]
        print(
            f"  NS_iters={entry['ns_iters']:>2}: "
            f"mean={summary['mean']:.2e}, "
            f"max={summary['max']:.2e}, "
            f"seed={entry['rng_seed']}"
        )


def print_hypotheses(results):
    print(f"\n\n{'=' * 90}")
    print("HYPOTHESIS TESTS")
    print(f"{'=' * 90}")
    for label in ["H1", "H2", "H3", "H4"]:
        entry = results["hypotheses"][label]
        print(f"\n{label}: {entry['statement']}")
        print(f"    Criterion: {entry['criterion_text']}")
        print(f"    Observed:  {entry['observed_text']}")
        print(f"    --> {'PASS' if entry['passed'] else 'FAIL'}")


def print_final_verdict(results):
    control = results["nonorthogonal_control"]
    verdict = results["verdict"]

    orthogonal_lines = [
        f"    {entry['size_label']}: mean={entry['summary']['mean']:.2e}, "
        f"max={entry['summary']['max']:.2e}"
        for entry in results["orthogonal_results"]
    ]
    orthogonal_block = "\n".join(orthogonal_lines)

    print(f"\n\n{'=' * 90}")
    print("FINAL VERDICT: EXP 2.1A CONJUGATION COVARIANCE")
    print(f"{'=' * 90}")
    print(
        f"\n  Orthogonal sampled trials:\n"
        f"{orthogonal_block}\n"
        f"\n"
        f"  Non-orthogonal control:\n"
        f"    mean={control['summary']['mean']:.2e}\n"
        f"\n"
        f"  Tests passed: {verdict['passed']}/{verdict['total']}\n"
        f"  Runtime: {results['runtime_sec']:.3f} s\n"
    )
    print(f"  {verdict['headline']}")
    print(f"  {verdict['limitations']}")
    print(f"\n{'=' * 90}")


def print_report(results):
    print("=" * 90)
    print("Exp 2.1a: CONJUGATION SINGLE-STEP EXACT COVARIANCE")
    print("=" * 90)
    print("Sampled numerical verification of the identity")
    print(f"  {results['identity_under_test']}")
    print()
    print(f"Finite Newton-Schulz iterations: {results['config']['ns_iters']}")
    print(f"Learning rate: {results['config']['lr']}")
    print(f"Trials per size: {results['config']['n_trials']}")
    print(f"Sizes: {results['config']['sizes']}")
    print(f"Base seed: {results['config']['base_seed']}")
    print()
    print("Prediction: sampled orthogonal trials should agree up to float64 roundoff,")
    print("while the non-orthogonal control should not.")

    print_orthogonal_results(results)
    print_error_distributions(results)
    print_nonorthogonal_control(results)
    print_ns_sensitivity(results)
    print_hypotheses(results)
    print_final_verdict(results)


# =============================================================================
# ENTRYPOINT
# =============================================================================


def main():
    results = run_experiment()
    print_report(results)


if __name__ == "__main__":
    main()
