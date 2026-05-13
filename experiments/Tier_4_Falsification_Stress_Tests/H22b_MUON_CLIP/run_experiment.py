#!/usr/bin/env python3
"""
H22b: Fixed-rank Muon-clip under extreme anisotropy
===================================================

Motivation
----------
H17 suggested that plain Muon can lose advantage when the target map becomes
extremely anisotropic. One plausible failure mode is that the orthogonalization
step spends equal update budget on directions associated with very small singular
values of the current gradient.

What this script's default experiment actually reports
------------------------------------------------------
- Deep linear 32x32 network with 4 layers.
- For each target condition number ``kappa``, estimate a single scalar
  ``k_clip`` from the mean initial gradient effective rank pooled across the
  learning-rate sweep seeds and all layers.
- Compare SGD, Muon, and Muon-clip using that fixed scalar ``k_clip`` for the
  entire run at the corresponding ``kappa``.
- The script measures initial gradient effective rank and anisotropy. It does
  not directly measure any "noise-energy fraction".

Scope note
----------
The training code also supports an explicitly labeled optional adaptive mode in
which Muon-clip chooses ``k`` from the current momentum/gradient effective rank
at each step. That adaptive mode is *not* the default reported experiment here.
The default script entrypoint preserves the original fixed-``k_clip`` behavior.
"""

from dataclasses import asdict, dataclass
import os
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIM = 32
N_LAYERS = 4
NUM_STEPS = 300
MOMENTUM = 0.9
NS_ITERS = 5
NUM_SEEDS = 5
BATCH_SIZE = 64

KAPPA_VALUES = [1, 10, 100, 1000, 10000]

LR_SGD = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
LR_MUON = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]
LR_CLIP = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.001]

FIXED_INIT_ERANK_MODE = 'fixed_init_erank'
ADAPTIVE_MOMENTUM_ERANK_MODE = 'adaptive_momentum_erank'
VALID_CLIP_MODES = (FIXED_INIT_ERANK_MODE, ADAPTIVE_MOMENTUM_ERANK_MODE)


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = DIM
    n_layers: int = N_LAYERS
    num_steps: int = NUM_STEPS
    momentum: float = MOMENTUM
    ns_iters: int = NS_ITERS
    num_seeds: int = NUM_SEEDS
    batch_size: int = BATCH_SIZE
    kappa_values: tuple = tuple(KAPPA_VALUES)
    lr_sgd: tuple = tuple(LR_SGD)
    lr_muon: tuple = tuple(LR_MUON)
    lr_clip: tuple = tuple(LR_CLIP)
    seed_start: int = 42
    seed_stride: int = 137
    sweep_seed_count: int = 3
    clip_mode: str = FIXED_INIT_ERANK_MODE


# =============================================================================
# CONFIG / SERIALIZATION HELPERS
# =============================================================================


def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()


DEFAULT_CONFIG = get_default_config()


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    raw = asdict(config)
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.clip_mode not in VALID_CLIP_MODES:
        raise ValueError(f"clip_mode must be one of {VALID_CLIP_MODES}, got {config.clip_mode!r}")
    if config.sweep_seed_count < 1:
        raise ValueError("sweep_seed_count must be >= 1")
    if config.num_seeds < config.sweep_seed_count:
        raise ValueError("num_seeds must be >= sweep_seed_count")
    if config.n_layers < 1 or config.dim < 1 or config.num_steps < 1 or config.batch_size < 1:
        raise ValueError("dim, n_layers, num_steps, and batch_size must all be positive")
    return config


def get_seeds(config: Optional[ExperimentConfig] = None):
    cfg = validate_config(config or DEFAULT_CONFIG)
    seeds = [cfg.seed_start + i * cfg.seed_stride for i in range(cfg.num_seeds)]
    sweep_seeds = seeds[:cfg.sweep_seed_count]
    return seeds, sweep_seeds


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



def effective_rank(M):
    sv = np.linalg.svd(M, compute_uv=False)
    sv2 = sv ** 2
    total = np.sum(sv2)
    if total < 1e-30:
        return 1.0
    sv2 = sv2 / total
    sv2 = sv2[sv2 > 1e-30]
    entropy = -np.sum(sv2 * np.log(sv2))
    return np.exp(entropy)



def muon_clip_step(G, k, n_iters=NS_ITERS):
    """Keep the top-k singular values of the current gradient, then orthogonalize."""
    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    k = max(1, min(int(k), len(sigma)))
    sigma_clip = sigma.copy()
    sigma_clip[k:] = 0
    G_clip = U @ np.diag(sigma_clip) @ Vt
    return newton_schulz(G_clip, n_iters=n_iters)



def make_ill_conditioned_data(kappa, seed, config: Optional[ExperimentConfig] = None):
    """Create data with a target linear map of condition number approximately ``kappa``."""
    cfg = validate_config(config or DEFAULT_CONFIG)
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.randn(cfg.dim, cfg.dim))
    V, _ = np.linalg.qr(rng.randn(cfg.dim, cfg.dim))
    sigmas = np.logspace(0, -np.log10(max(kappa, 1)), cfg.dim)
    W_target = U @ np.diag(sigmas) @ V.T

    X = rng.randn(cfg.dim, cfg.batch_size) * 0.3
    Y = W_target @ X
    return X, Y



def init_weights(seed, config: Optional[ExperimentConfig] = None):
    cfg = validate_config(config or DEFAULT_CONFIG)
    rng = np.random.RandomState(seed)
    return [np.eye(cfg.dim) + rng.randn(cfg.dim, cfg.dim) * 0.1 for _ in range(cfg.n_layers)]



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


# =============================================================================
# TRAINING
# =============================================================================


def _resolve_clip_k(clip_k, layer_index: int, n_layers: int):
    if clip_k is None:
        return None
    if np.isscalar(clip_k):
        return max(1, int(np.round(float(clip_k))))
    if isinstance(clip_k, np.ndarray):
        values = clip_k.tolist()
    elif isinstance(clip_k, (list, tuple)):
        values = list(clip_k)
    else:
        raise TypeError("clip_k must be None, a scalar, or a per-layer list/tuple/ndarray")
    if len(values) != n_layers:
        raise ValueError(f"Per-layer clip_k must have length {n_layers}, got {len(values)}")
    return max(1, int(np.round(float(values[layer_index]))))



def train(weights_init, X, Y, lr, opt, clip_k=None, config: Optional[ExperimentConfig] = None):
    """
    Train with the specified optimizer.

    Parameters
    ----------
    weights_init : list[np.ndarray]
        Initial layer weights.
    X, Y : np.ndarray
        Training data and targets.
    lr : float
        Learning rate.
    opt : {'sgd', 'muon', 'muon_clip'}
        Optimizer choice.
    clip_k : None, scalar, or per-layer list/tuple/ndarray
        Muon-clip rank control. If ``None``, Muon-clip uses the optional adaptive
        per-step momentum/gradient effective-rank rule. If a scalar (the default
        experiment path), the same fixed ``k`` is used for every layer/step. A
        per-layer sequence is also supported.
    config : ExperimentConfig, optional
        Experiment settings. If omitted, module defaults are used.
    """
    cfg = validate_config(config or DEFAULT_CONFIG)
    weights = [W.copy() for W in weights_init]
    mom = [np.zeros_like(W) for W in weights]

    for _ in range(cfg.num_steps):
        loss = compute_loss(weights, X, Y)
        if not np.isfinite(loss) or loss > 1e10:
            return float('inf')
        grads = compute_gradients(weights, X, Y)

        for l in range(len(weights)):
            if opt == 'sgd':
                mom[l] = cfg.momentum * mom[l] + grads[l]
            elif opt == 'muon':
                mom[l] = cfg.momentum * mom[l] + newton_schulz(grads[l], n_iters=cfg.ns_iters)
            elif opt == 'muon_clip':
                k = _resolve_clip_k(clip_k, l, len(weights))
                if k is None:
                    ref = mom[l] if np.linalg.norm(mom[l]) > 1e-15 else grads[l]
                    k = max(1, int(np.round(effective_rank(ref))))
                mom[l] = cfg.momentum * mom[l] + muon_clip_step(grads[l], k, n_iters=cfg.ns_iters)
            else:
                raise ValueError(f"Unknown optimizer {opt!r}")

            weights[l] = weights[l] - lr * mom[l]

    return float(compute_loss(weights, X, Y))


# =============================================================================
# DIAGNOSTICS / EVALUATION HELPERS
# =============================================================================


def _loss_summary(loss_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    finite_losses = [record['loss'] for record in loss_records if record['finite']]
    mean_finite_loss = float(np.mean(finite_losses)) if finite_losses else float('inf')
    std_finite_loss = float(np.std(finite_losses)) if finite_losses else float('nan')
    median_finite_loss = float(np.median(finite_losses)) if finite_losses else float('nan')
    return {
        'loss_records': loss_records,
        'finite_losses': finite_losses,
        'finite_count': int(len(finite_losses)),
        'diverged_count': int(len(loss_records) - len(finite_losses)),
        'all_finite': bool(len(finite_losses) == len(loss_records)),
        'mean_finite_loss': mean_finite_loss,
        'std_finite_loss': std_finite_loss,
        'median_finite_loss': median_finite_loss,
    }



def measure_init_diagnostics(kappa, seeds, config: Optional[ExperimentConfig] = None):
    """
    Measure the initial gradient effective rank and anisotropy used to choose
    the default fixed scalar ``k_clip``.
    """
    cfg = validate_config(config or DEFAULT_CONFIG)
    layer_records: List[Dict[str, Any]] = []
    per_seed_means: List[Dict[str, Any]] = []
    eranks = []
    anisotropies = []

    for s in seeds:
        X, Y = make_ill_conditioned_data(kappa, s, cfg)
        w = init_weights(s + 5000, cfg)
        grads = compute_gradients(w, X, Y)

        seed_eranks = []
        seed_anisos = []
        for layer_index, G in enumerate(grads):
            sv = np.linalg.svd(G, compute_uv=False)
            erank = float(effective_rank(G))
            anisotropy = float(sv[0] / max(sv[-1], 1e-30))
            layer_records.append({
                'seed': int(s),
                'layer': int(layer_index),
                'effective_rank': erank,
                'anisotropy': anisotropy,
            })
            seed_eranks.append(erank)
            seed_anisos.append(anisotropy)
            eranks.append(erank)
            anisotropies.append(anisotropy)

        per_seed_means.append({
            'seed': int(s),
            'mean_effective_rank': float(np.mean(seed_eranks)),
            'mean_anisotropy': float(np.mean(seed_anisos)),
        })

    mean_erank = float(np.mean(eranks)) if eranks else float('nan')
    mean_aniso = float(np.mean(anisotropies)) if anisotropies else float('nan')
    k_clip = max(1, int(np.round(mean_erank))) if np.isfinite(mean_erank) else 1

    return {
        'kappa': int(kappa),
        'estimation_seeds': [int(s) for s in seeds],
        'layer_records': layer_records,
        'per_seed_means': per_seed_means,
        'mean_effective_rank': mean_erank,
        'std_effective_rank': float(np.std(eranks)) if eranks else float('nan'),
        'mean_anisotropy': mean_aniso,
        'std_anisotropy': float(np.std(anisotropies)) if anisotropies else float('nan'),
        'k_clip': int(k_clip),
        'k_clip_rule': 'round(mean initial gradient effective rank pooled across sweep seeds and layers)',
    }



def evaluate_optimizer(opt, lr, kappa, seeds, clip_k=None, config: Optional[ExperimentConfig] = None):
    cfg = validate_config(config or DEFAULT_CONFIG)
    loss_records = []
    for s in seeds:
        X, Y = make_ill_conditioned_data(kappa, s, cfg)
        w = init_weights(s + 5000, cfg)
        final_loss = float(train(w, X, Y, lr, opt, clip_k=clip_k, config=cfg))
        loss_records.append({
            'seed': int(s),
            'loss': final_loss,
            'finite': bool(np.isfinite(final_loss)),
        })

    summary = _loss_summary(loss_records)
    summary.update({
        'optimizer': opt,
        'lr': float(lr),
        'kappa': int(kappa),
        'clip_k': None if clip_k is None else clip_k,
    })
    return summary



def sweep_lr(opt, lr_candidates, kappa, eval_seeds, clip_k=None, config: Optional[ExperimentConfig] = None):
    """Sweep learning rates and return structured candidate results plus the best choice."""
    cfg = validate_config(config or DEFAULT_CONFIG)
    candidate_results = []
    best_lr = float(lr_candidates[-1])
    best_loss = float('inf')

    for lr in lr_candidates:
        candidate = evaluate_optimizer(opt, lr, kappa, eval_seeds, clip_k=clip_k, config=cfg)
        candidate_results.append(candidate)
        if candidate['mean_finite_loss'] < best_loss:
            best_loss = candidate['mean_finite_loss']
            best_lr = float(lr)

    for candidate in candidate_results:
        candidate['is_best'] = bool(np.isclose(candidate['lr'], best_lr) and candidate['mean_finite_loss'] == best_loss)

    return {
        'optimizer': opt,
        'kappa': int(kappa),
        'clip_k': None if clip_k is None else clip_k,
        'lr_candidates': [float(lr) for lr in lr_candidates],
        'candidate_results': candidate_results,
        'best_lr': float(best_lr),
        'best_loss': float(best_loss),
    }



def _clip_setting_for_kappa(init_diagnostics, config: ExperimentConfig):
    if config.clip_mode == FIXED_INIT_ERANK_MODE:
        return {
            'mode': FIXED_INIT_ERANK_MODE,
            'clip_k': int(init_diagnostics['k_clip']),
            'description': (
                'Fixed scalar k_clip per kappa, estimated from the mean initial '
                'gradient effective rank over sweep seeds and layers.'
            ),
        }
    if config.clip_mode == ADAPTIVE_MOMENTUM_ERANK_MODE:
        return {
            'mode': ADAPTIVE_MOMENTUM_ERANK_MODE,
            'clip_k': None,
            'description': (
                'Adaptive per-layer/per-step k from the current momentum effective '
                'rank (falling back to the current gradient on the first step).'
            ),
        }
    raise ValueError(f"Unknown clip_mode {config.clip_mode!r}")



def _safe_ratio(numerator, denominator):
    return float(numerator / max(denominator, 1e-30))



def _summary_row(kappa_result):
    init_diag = kappa_result['init_diagnostics']
    best_lrs = kappa_result['best_lrs']
    evals = kappa_result['evaluations']
    summary = kappa_result['summary']
    return {
        'kappa': int(kappa_result['kappa']),
        'clip_mode': kappa_result['clip_setting']['mode'],
        'k_clip': kappa_result['clip_setting']['clip_k'],
        'mean_effective_rank': float(init_diag['mean_effective_rank']),
        'std_effective_rank': float(init_diag['std_effective_rank']),
        'mean_anisotropy': float(init_diag['mean_anisotropy']),
        'std_anisotropy': float(init_diag['std_anisotropy']),
        'best_sgd_lr': float(best_lrs['sgd']),
        'best_muon_lr': float(best_lrs['muon']),
        'best_clip_lr': float(best_lrs['muon_clip']),
        'sgd_mean': float(summary['sgd_mean']),
        'sgd_std': float(evals['sgd']['std_finite_loss']),
        'sgd_finite_count': int(evals['sgd']['finite_count']),
        'sgd_diverged_count': int(evals['sgd']['diverged_count']),
        'muon_mean': float(summary['muon_mean']),
        'muon_std': float(evals['muon']['std_finite_loss']),
        'muon_finite_count': int(evals['muon']['finite_count']),
        'muon_diverged_count': int(evals['muon']['diverged_count']),
        'clip_mean': float(summary['clip_mean']),
        'clip_std': float(evals['muon_clip']['std_finite_loss']),
        'clip_finite_count': int(evals['muon_clip']['finite_count']),
        'clip_diverged_count': int(evals['muon_clip']['diverged_count']),
        'adv_muon': float(summary['adv_muon']),
        'adv_clip': float(summary['adv_clip']),
        'clip_vs_muon': float(summary['clip_vs_muon']),
        'clip_better_than_muon': bool(summary['clip_better_than_muon']),
        'clip_restored_vs_muon_heuristic': bool(summary['clip_restored_vs_muon_heuristic']),
        'muon_status': summary['muon_status'],
    }


# =============================================================================
# EXPERIMENT DRIVER / REPORTING
# =============================================================================


def print_banner(config: ExperimentConfig, seeds, sweep_seeds):
    print("=" * 100)
    print("H22b: Fixed-rank Muon-clip under extreme anisotropy")
    print("=" * 100)
    print(f"Network: {config.n_layers}-layer deep linear {config.dim}x{config.dim}")
    print(f"Steps: {config.num_steps}, Seeds: {config.num_seeds} (sweep seeds: {len(sweep_seeds)})")
    print(f"kappa_target: {list(config.kappa_values)}")
    print(f"Clip mode: {config.clip_mode}")
    if config.clip_mode == FIXED_INIT_ERANK_MODE:
        print("Default reported mode: fixed scalar k_clip per kappa from initial gradient effective rank.")
    else:
        print("Optional mode: adaptive momentum/gradient effective-rank clipping.")
    print("Measured diagnostics: initial gradient effective rank and anisotropy.")
    print("Not measured directly: any true noise-energy fraction.")
    print(f"Evaluation seeds: {seeds}")
    print(f"LR sweep seeds: {sweep_seeds}")
    print()



def print_final_report(results):
    config = results['config']
    rows = results['summary_rows']

    print(f"\n\n{'=' * 100}")
    print("RESULTS: MUON-CLIP vs MUON vs SGD ACROSS ANISOTROPY")
    print(f"{'=' * 100}")
    print(f"Mode reported here: {config['clip_mode']}")
    if config['clip_mode'] == FIXED_INIT_ERANK_MODE:
        print("Interpret k_clip as a fixed scalar chosen separately for each kappa from initial diagnostics.")
    else:
        print("Interpret k as adaptive and step-dependent inside training.")

    print(
        f"\n  {'kappa':>8}  {'k_clip':>6}  {'erank':>6}  {'aniso':>10}  {'SGD':>12}  {'Muon':>12}  "
        f"{'Clip':>12}  {'Muon adv':>10}  {'Clip adv':>10}  {'Clip>Muon?':>11}"
    )
    print("  " + "-" * 112)
    for row in rows:
        clip_better = "YES" if row['clip_better_than_muon'] else "NO"
        clip_k_str = f"{row['k_clip']}" if row['k_clip'] is not None else "adapt"
        print(
            f"  {row['kappa']:>8}  {clip_k_str:>6}  {row['mean_effective_rank']:>6.1f}  "
            f"{row['mean_anisotropy']:>10.0f}  {row['sgd_mean']:>12.4e}  {row['muon_mean']:>12.4e}  "
            f"{row['clip_mean']:>12.4e}  {row['adv_muon']:>10.1f}x  {row['adv_clip']:>10.1f}x  "
            f"{clip_better:>11}"
        )

    print("\n  Finite-run counts by optimizer (out of total seeds):")
    for row in rows:
        print(
            f"    kappa={row['kappa']}: "
            f"SGD {row['sgd_finite_count']}/{results['config']['num_seeds']}, "
            f"Muon {row['muon_finite_count']}/{results['config']['num_seeds']}, "
            f"Clip {row['clip_finite_count']}/{results['config']['num_seeds']}"
        )

    print("\n  === Heuristic checks (interpret cautiously) ===")
    high_kappas = [row for row in rows if row['kappa'] >= 1000]
    for row in high_kappas:
        label = 'RESTORED' if row['clip_restored_vs_muon_heuristic'] else 'NOT RESTORED'
        print(
            f"    kappa={row['kappa']}: Muon adv={row['adv_muon']:.2f}x, "
            f"Clip adv={row['adv_clip']:.2f}x  --> {label}"
        )

    print("\n  === Muon struggle detection ===")
    for row in rows:
        print(f"    kappa={row['kappa']}: Muon adv={row['adv_muon']:.2f}x  [{row['muon_status']}]")

    print(f"\n{'=' * 100}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 100}")



def run_experiment(config: Optional[ExperimentConfig] = None, verbose: bool = True):
    cfg = validate_config(config or DEFAULT_CONFIG)
    seeds, sweep_seeds = get_seeds(cfg)

    if verbose:
        print_banner(cfg, seeds, sweep_seeds)

    kappa_results = []

    for kappa in cfg.kappa_values:
        init_diagnostics = measure_init_diagnostics(kappa, sweep_seeds, cfg)
        clip_setting = _clip_setting_for_kappa(init_diagnostics, cfg)
        clip_k_for_training = clip_setting['clip_k']

        if verbose:
            print(f"\n  kappa={kappa}")
            if clip_k_for_training is None:
                clip_descriptor = 'adaptive'
            else:
                clip_descriptor = f"fixed clip k={clip_k_for_training}"
            print(
                "    Initial gradient diagnostics: "
                f"erank={init_diagnostics['mean_effective_rank']:.1f}/{cfg.dim}, "
                f"anisotropy={init_diagnostics['mean_anisotropy']:.1f}, "
                f"mode={clip_descriptor}"
            )

        lr_sweeps = {
            'sgd': sweep_lr('sgd', cfg.lr_sgd, kappa, sweep_seeds, config=cfg),
            'muon': sweep_lr('muon', cfg.lr_muon, kappa, sweep_seeds, config=cfg),
            'muon_clip': sweep_lr('muon_clip', cfg.lr_clip, kappa, sweep_seeds, clip_k=clip_k_for_training, config=cfg),
        }
        best_lrs = {
            'sgd': lr_sweeps['sgd']['best_lr'],
            'muon': lr_sweeps['muon']['best_lr'],
            'muon_clip': lr_sweeps['muon_clip']['best_lr'],
        }

        if verbose:
            print(f"    SGD best LR: {best_lrs['sgd']}")
            print(f"    Muon best LR: {best_lrs['muon']}")
            print(f"    Muon-clip best LR: {best_lrs['muon_clip']}")

        evaluations = {
            'sgd': evaluate_optimizer('sgd', best_lrs['sgd'], kappa, seeds, config=cfg),
            'muon': evaluate_optimizer('muon', best_lrs['muon'], kappa, seeds, config=cfg),
            'muon_clip': evaluate_optimizer('muon_clip', best_lrs['muon_clip'], kappa, seeds, clip_k=clip_k_for_training, config=cfg),
        }

        sgd_mean = evaluations['sgd']['mean_finite_loss']
        muon_mean = evaluations['muon']['mean_finite_loss']
        clip_mean = evaluations['muon_clip']['mean_finite_loss']
        adv_muon = _safe_ratio(sgd_mean, muon_mean)
        adv_clip = _safe_ratio(sgd_mean, clip_mean)
        clip_vs_muon = _safe_ratio(muon_mean, clip_mean)

        summary = {
            'sgd_mean': float(sgd_mean),
            'muon_mean': float(muon_mean),
            'clip_mean': float(clip_mean),
            'adv_muon': float(adv_muon),
            'adv_clip': float(adv_clip),
            'clip_vs_muon': float(clip_vs_muon),
            'clip_better_than_muon': bool(clip_vs_muon > 1.1),
            'clip_restored_vs_muon_heuristic': bool(adv_clip > 1.2 * adv_muon),
            'muon_status': 'UNSET',
        }

        kappa_result = {
            'kappa': int(kappa),
            'clip_setting': clip_setting,
            'init_diagnostics': init_diagnostics,
            'lr_sweeps': lr_sweeps,
            'best_lrs': best_lrs,
            'evaluations': evaluations,
            'summary': summary,
        }
        kappa_results.append(kappa_result)

        if verbose:
            print(
                f"    SGD={sgd_mean:.4e}  Muon={muon_mean:.4e}  Clip={clip_mean:.4e}"
            )
            print(
                f"    Muon adv: {adv_muon:.2f}x   Clip adv: {adv_clip:.2f}x   Clip/Muon: {clip_vs_muon:.2f}x"
            )

    if kappa_results:
        low_adv_muon = kappa_results[0]['summary']['adv_muon']
        for kappa_result in kappa_results:
            adv_muon = kappa_result['summary']['adv_muon']
            lost = adv_muon < 1.0
            declined = adv_muon < low_adv_muon * 0.5
            kappa_result['summary']['muon_status'] = 'LOST' if lost else ('DECLINED' if declined else 'OK')

    summary_rows = [_summary_row(kappa_result) for kappa_result in kappa_results]

    results = {
        'experiment_name': 'H22b_MUON_CLIP',
        'script_path': os.path.abspath(__file__),
        'counterpart_notebook_path': os.path.join(SCRIPT_DIR, 'run_experiment.ipynb'),
        'config': config_to_dict(cfg),
        'notes': [
            'Default reported mode is fixed_init_erank: one scalar k_clip per kappa.',
            'k_clip is estimated from initial gradient effective rank over sweep seeds and layers.',
            'The script measures effective rank and anisotropy but does not directly measure noise-energy fraction.',
            'Adaptive momentum-based clipping is available as an optional mode but is not the default reported result.',
        ],
        'seeds': [int(s) for s in seeds],
        'sweep_seeds': [int(s) for s in sweep_seeds],
        'kappa_results': kappa_results,
        'summary_rows': summary_rows,
    }

    if verbose:
        print_final_report(results)

    return results



def main():
    return run_experiment(config=DEFAULT_CONFIG, verbose=True)


if __name__ == '__main__':
    main()
