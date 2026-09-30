#!/usr/bin/env python3
"""
CartPole Benchmark — Comparison of Learning Strategies

Compares two state representations and two learning strategies:

  Cortex Only: Uses just the learned representation for decision-making.

  Cortex + Handcrafted Features: Adds raw state features and derivatives
  to improve learning speed and stability.

  Reactive: Learns from immediate rewards only (model-free).

  Predictive: Uses learned forward models to anticipate outcomes (model-based).

Usage:
    python benchmark_cartpole.py              # Run with 1500 episodes
    python benchmark_cartpole.py --episodes 200  # Quick test
    python benchmark_cartpole.py --seed 42       # Specify random seed
"""

import sys
import os
import time
import argparse
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    print("Error: gymnasium is required. Install with: pip install gymnasium")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cerebral_cortex import CerebralCortex
from basal_ganglia import BasalGanglia
from cerebellum import (
    ForwardModel,
    InverseModel,
    CerebellarCorrector,
    HybridController,
)

from train_cartpole import (
    _action_for_cerebellum,
    _select_action_reactive,
    _select_action_predictive,
    train,
    run_test,
    _smooth,
)


# ======================================================================
# Protocol configuration
# ======================================================================

MODES = ["reactive", "predictive"]

# Registry of all available conditions.
# Each entry defines how the BG state is built from (raw_obs, cortex_repr).
CONDITION_REGISTRY = {
    "cortex_only": {
        "label":       "Cortex simple (6D)",
        "use_augment": False,
        "use_raw_obs": False,
        "color":       "#2196F3",   # blue
    },
    "cortex_with_raw": {
        "label":       "Raw obs + Cortex (10D)",
        "use_augment": False,
        "use_raw_obs": True,
        "color":       "#2196F3",   # blue
    },
    "env_augmented": {
        "label":       "Env + features (14D)",
        "use_augment": True,
        "use_raw_obs": False,
        "color":       "#FF5722",   # deep orange
    },
}

# Active conditions for this run (overridable via --cond-a / --cond-b).
DEFAULT_COND_A = "cortex_with_raw"
DEFAULT_COND_B = "env_augmented"

# Resolved at runtime by run_benchmark() / main().
CONDITIONS = [DEFAULT_COND_A, DEFAULT_COND_B]

# Convenience accessors (populated from registry when CONDITIONS is set).
def _build_condition_maps(conditions):
    labels     = {c: CONDITION_REGISTRY[c]["label"]       for c in conditions}
    use_augment = {c: CONDITION_REGISTRY[c]["use_augment"] for c in conditions}
    use_raw_obs = {c: CONDITION_REGISTRY[c]["use_raw_obs"] for c in conditions}
    colors     = {c: CONDITION_REGISTRY[c]["color"]       for c in conditions}
    return labels, use_augment, use_raw_obs, colors

CONDITION_LABELS, CONDITION_USE_AUGMENT, CONDITION_USE_RAW_OBS, CONDITION_COLORS = \
    _build_condition_maps(CONDITIONS)

MODE_LABELS = {
    "reactive":   "Reactif  (Sec. 4.1)",
    "predictive": "Predictif (Sec. 4.2.1)",
}
MODE_LINESTYLES = {
    "reactive":   "-",
    "predictive": "--",
}

STRATEGY_LABELS = {
    "bg":         "Basal Ganglia\n(4.1, Fig. 6)",
    "predictive": "Predictive\n(4.2.1, Fig. 7)",
    "inverse":    "Inverse model\n(Fig. 11)",
    "hybrid":     "Hybrid\n(v2)",
}
STRATEGY_COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]


# ======================================================================
# Individual report — one per (condition × mode)
# ======================================================================

def plot_individual_report(
    metrics: dict,
    condition: str,
    mode: str,
    test_results: dict,
    seed: int,
    elapsed: float,
    filename: str,
):
    """Detailed 5-panel report for ONE (condition, mode) combination."""

    n_ep = len(metrics["rewards"])
    window = min(50, max(1, n_ep // 5))
    episodes = np.arange(1, n_ep + 1)
    smooth_x = np.arange(window, n_ep + 1)
    color = CONDITION_COLORS[condition]

    fig, axes = plt.subplots(5, 1, figsize=(12, 17.5))
    fig.suptitle(
        f"Doya (1999) — CartPole-v1\n"
        f"{CONDITION_LABELS[condition]} | {MODE_LABELS[mode]}"
        f"   seed={seed}   {elapsed:.0f}s",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Panel 1 — Reward
    ax = axes[0]
    ax.plot(episodes, metrics["rewards"], alpha=0.2, color=color, linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["rewards"], window), color=color, linewidth=2.5,
            label=f"Moving avg (w={window})")
    ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
    ax.set_ylabel("Steps survived"); ax.set_title("Reward per episode")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    # Panel 2 — TD error
    ax = axes[1]
    ax.plot(episodes, metrics["td_magnitudes"], alpha=0.2, color="C1", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["td_magnitudes"], window), color="C1", linewidth=2.5)
    ax.set_ylabel("|delta| (DA)"); ax.set_title("TD error — Dopaminergic signal (Eq. 10)")
    ax.grid(True, alpha=0.3)

    # Panel 3 — Cortex error
    ax = axes[2]
    ax.plot(episodes, metrics["cortex_errors"], alpha=0.2, color="C2", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["cortex_errors"], window), color="C2", linewidth=2.5)
    ax.set_ylabel("||x - W'y||^2"); ax.set_title("Cortex error — Unsupervised Hebb (Eq. 17-18)")
    ax.grid(True, alpha=0.3)

    # Panel 4 — Cerebellar errors
    ax = axes[3]
    ax.plot(episodes, metrics["forward_errors"], alpha=0.2, color="C3", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["forward_errors"], window),
            color="C3", linewidth=2.5, label="Forward (Eq. 21)")
    ax.plot(episodes, metrics["inverse_errors"], alpha=0.2, color="C4", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["inverse_errors"], window),
            color="C4", linewidth=2.5, label="Inverse (Fig. 11)")
    ax.set_ylabel("MSE"); ax.set_title("Cerebellar errors — Supervised (Eq. 5)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3); ax.set_xlabel("Episode")

    # Panel 5 — Test bars
    ax = axes[4]
    names = list(test_results.keys())
    means = [np.mean(v) for v in test_results.values()]
    stds  = [np.std(v)  for v in test_results.values()]
    bars = ax.bar(
        [STRATEGY_LABELS.get(n, n) for n in names], means, yerr=stds,
        color=STRATEGY_COLORS[:len(names)], alpha=0.85, capsize=6,
        edgecolor="black", linewidth=0.5,
    )
    ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
    ax.set_ylabel("Steps survived"); ax.set_title("Test — 4 strategies (50 trials)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


# ======================================================================
# Comparative report — GridSpec(4, 2)
# ======================================================================

def plot_comparison_report(all_results: dict, filename: str):
    """4×2 comparison: columns = reactive vs predictive,
    each panel overlays cortex_only (blue) vs env_augmented (orange)."""

    fig = plt.figure(figsize=(16, 22))
    gs = GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.3)

    label_a = CONDITION_LABELS[CONDITIONS[0]]
    label_b = CONDITION_LABELS[CONDITIONS[1]]
    fig.suptitle(
        f"Doya (1999) — CartPole-v1\n"
        f"{label_a} (bleu) vs {label_b} (orange)  |  "
        f"Reactif (trait plein) vs Predictif (pointille)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Rows 0-2: metrics per mode (cols) × condition (lines) ──
    metric_panels = [
        ("rewards",      "Steps survived",  "Courbe d'apprentissage"),
        ("td_magnitudes", "|delta| (DA)",   "TD error (Eq. 10)"),
        ("cortex_errors", "||x-W'y||^2",   "Erreur cortex (Eq. 17-18)"),
    ]

    for row, (metric_key, ylabel, title) in enumerate(metric_panels):
        for col, mode in enumerate(MODES):
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f"{title}\n{MODE_LABELS[mode]}", fontsize=10)
            for cond in CONDITIONS:
                data = all_results[cond][mode]["metrics"][metric_key]
                n_ep = len(data)
                window = min(50, max(1, n_ep // 5))
                episodes = np.arange(1, n_ep + 1)
                smooth_x = np.arange(window, n_ep + 1)
                color = CONDITION_COLORS[cond]
                label = CONDITION_LABELS[cond]
                ax.plot(episodes, data, alpha=0.08, color=color, linewidth=0.5)
                ax.plot(smooth_x, _smooth(data, window), color=color,
                        linewidth=2.5, label=label)
            if metric_key == "rewards":
                ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Episode")
            ax.legend(loc="upper left" if metric_key == "rewards" else "upper right",
                      fontsize=8)
            ax.grid(True, alpha=0.3)

    # ── Row 3: Test bars — one panel per mode ──
    for col, mode in enumerate(MODES):
        ax = fig.add_subplot(gs[3, col])
        strategies = list(all_results[CONDITIONS[0]][mode]["test_results"].keys())
        n_strat = len(strategies)
        x = np.arange(n_strat)
        bar_width = 0.35

        for i, cond in enumerate(CONDITIONS):
            test = all_results[cond][mode]["test_results"]
            means = [np.mean(test[s]) for s in strategies]
            stds  = [np.std(test[s])  for s in strategies]
            offset = (i - 0.5) * bar_width
            bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                          color=CONDITION_COLORS[cond], alpha=0.85, capsize=5,
                          edgecolor="black", linewidth=0.5,
                          label=CONDITION_LABELS[cond])
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
                        f"{m:.0f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], fontsize=8)
        ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
        ax.set_ylabel("Steps survived (50 trials)")
        ax.set_title(f"Test — {MODE_LABELS[mode]}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


# ======================================================================
# Text report
# ======================================================================

def write_text_report(all_results: dict, filename: str):
    lines = []
    lines.append("=" * 80)
    lines.append("  BENCHMARK — Doya (1999) CartPole-v1")
    lines.append("  Raw+Cortex (10D) vs Env+features (14D)  x  Reactif vs Predictif")
    lines.append("=" * 80)

    for cond in CONDITIONS:
        for mode in MODES:
            data = all_results[cond][mode]
            metrics = data["metrics"]
            test    = data["test_results"]
            rewards = metrics["rewards"]
            n = len(rewards); q = max(1, n // 4)
            lines.append("")
            lines.append(f"  [{CONDITION_LABELS[cond]} | {MODE_LABELS[mode]}]  "
                         f"seed={data['seed']}  ep={n}  t={data['elapsed']:.0f}s")
            lines.append(f"  {'─' * 75}")
            for i in range(4):
                s, e = i * q, min((i + 1) * q, n)
                if s >= n: break
                lines.append(f"    ep {s+1:5d}-{e:5d}:  "
                              f"{np.mean(rewards[s:e]):7.1f} +/- {np.std(rewards[s:e]):5.1f}"
                              f"  max={np.max(rewards[s:e]):.0f}")
            tail = max(1, n // 10)
            lines.append(f"    Final 10%: R={np.mean(rewards[-tail:]):.1f}  "
                         f"DA={np.mean(metrics['td_magnitudes'][-tail:]):.4f}  "
                         f"Cx={np.mean(metrics['cortex_errors'][-tail:]):.4f}")
            lines.append(f"    Test (50 trials):")
            for strat, label in [("bg","BG"),("predictive","Pred."),
                                  ("inverse","Inv."),("hybrid","Hybrid")]:
                r = test[strat]
                n500 = sum(1 for x in r if x >= 500)
                lines.append(f"      {label:<8s}: {np.mean(r):7.1f} +/- {np.std(r):5.1f}"
                              f"  max={np.max(r):.0f}  >=500: {n500}/50")

    # Summary comparison table
    lines.append("")
    lines.append("=" * 80)
    lines.append("  COMPARAISON — Test BG (principal)")
    ca, cb = CONDITIONS[0], CONDITIONS[1]
    lines.append(f"  {'':20s} | {CONDITION_LABELS[ca]:>14s} | {CONDITION_LABELS[cb]:>14s} | {'Delta':>10s}")
    lines.append(f"  {'─' * 65}")
    for mode in MODES:
        r_c = np.mean(all_results[ca][mode]["test_results"]["bg"])
        r_a = np.mean(all_results[cb][mode]["test_results"]["bg"])
        d = r_a - r_c
        sign = "+" if d >= 0 else ""
        lines.append(f"  {MODE_LABELS[mode]:<20s} | {r_c:14.1f} | {r_a:14.1f} | {sign}{d:9.1f}")
    lines.append("=" * 80)

    report = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    sys.stdout.buffer.write((report + "\n").encode("utf-8", errors="replace"))
    print(f"\n  [OK] {filename}")


# ======================================================================
# Main benchmark loop
# ======================================================================

def run_benchmark(n_episodes: int = 1500, seed: int = 42,
                  cond_a: str = DEFAULT_COND_A, cond_b: str = DEFAULT_COND_B):
    global CONDITIONS, CONDITION_LABELS, CONDITION_USE_AUGMENT, CONDITION_USE_RAW_OBS, CONDITION_COLORS
    CONDITIONS = [cond_a, cond_b]
    CONDITION_LABELS, CONDITION_USE_AUGMENT, CONDITION_USE_RAW_OBS, CONDITION_COLORS = \
        _build_condition_maps(CONDITIONS)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # all_results[cond][mode]
    all_results = {cond: {} for cond in CONDITIONS}

    for cond in CONDITIONS:
        use_augment  = CONDITION_USE_AUGMENT[cond]
        use_raw_obs  = CONDITION_USE_RAW_OBS[cond]
        for mode in MODES:
            print()
            print("=" * 72)
            print(f"  {CONDITION_LABELS[cond]} | {MODE_LABELS[mode]}")
            print("=" * 72)
            print()

            t0 = time.time()

            results = train(
                mode=mode,
                n_episodes=n_episodes,
                print_every=200,
                seed=seed,
                use_augment=use_augment,
                use_raw_obs=use_raw_obs,
            )
            cortex, bg, fwd, inv, rewards, lengths, metrics = results

            test_results = run_test(
                cortex, bg, fwd, inv,
                n_trials=50, seed=999,
                use_augment=use_augment,
                use_raw_obs=use_raw_obs,
            )

            elapsed = time.time() - t0

            all_results[cond][mode] = {
                "metrics":      metrics,
                "test_results": test_results,
                "elapsed":      elapsed,
                "seed":         seed,
                "n_episodes":   n_episodes,
            }

            individual_file = os.path.join(output_dir, f"cartpole_{cond}_{mode}.png")
            plot_individual_report(
                metrics, cond, mode, test_results, seed, elapsed, individual_file
            )

    comparison_file = os.path.join(output_dir, "cartpole_representation_comparison_9.png")
    plot_comparison_report(all_results, comparison_file)

    text_file = os.path.join(output_dir, "cartpole_representation_metrics.txt")
    write_text_report(all_results, text_file)

    return all_results


# ======================================================================
# Entry point
# ======================================================================

def main():
    valid = sorted(CONDITION_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Benchmark — 2 conditions x Reactif/Predictif sur CartPole-v1",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--episodes", "-n", type=int, default=1500,
                        help="Episodes per run (default: 1500)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cond-a", type=str, default=DEFAULT_COND_A,
                        choices=valid,
                        help=f"Condition A (default: {DEFAULT_COND_A})\n"
                             + "\n".join(f"  {k}: {CONDITION_REGISTRY[k]['label']}" for k in valid))
    parser.add_argument("--cond-b", type=str, default=DEFAULT_COND_B,
                        choices=valid,
                        help=f"Condition B (default: {DEFAULT_COND_B})")
    args = parser.parse_args()

    label_a = CONDITION_REGISTRY[args.cond_a]["label"]
    label_b = CONDITION_REGISTRY[args.cond_b]["label"]

    print()
    print("=" * 72)
    print("  BENCHMARK — Representation BG sur CartPole-v1")
    print(f"  Cond A : {label_a}")
    print(f"  Cond B : {label_b}")
    print("  x  Reactif (Sec. 4.1) vs Predictif (Sec. 4.2.1)")
    print("=" * 72)
    print(f"  Episodes : {args.episodes} par run  (4 runs total)")
    print(f"  Seed     : {args.seed}")
    print(f"  Trials   : 50 par strategie de test")
    print("=" * 72)

    run_benchmark(n_episodes=args.episodes, seed=args.seed,
                  cond_a=args.cond_a, cond_b=args.cond_b)

    print()
    print("  Generated files:")
    for cond in CONDITIONS:
        for mode in MODES:
            fn = f"cartpole_{cond}_{mode}.png"
            print(f"    - {fn}")

    print("    - cartpole_representation_comparison.png")
    print("    - cartpole_representation_metrics.txt")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
