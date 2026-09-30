#!/usr/bin/env python3
"""
Benchmark Final — sur CartPole-v1
================================================

Expérience : Three-factor neuromodulation with improved cortex, SANS AUGMENT
    Mirrors benchmark_cartpolev5.py but with CerebralCortexImproved as base cortex.
    Conditions: 2F / 3F-Cortex / 3F-Cerebellum / 3F-Full
    Output: report_vfinal_comparison.png

Usage
-----
    python benchmark_final.py                  # 1500 episodes
    python benchmark_final.py --episodes 300   # smoke test
    python benchmark_final.py --seed 7
"""

import sys
import os
import time
import argparse
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    print("Error: gymnasium required. pip install gymnasium")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from cerebral_cortex_improved import CerebralCortexImproved, CerebralCortexImproved3F
from basal_ganglia_actrace import BasalGangliaACTrace
from cerebellum import ForwardModel, InverseModel, CerebellarCorrector, HybridController
from cerebellum_3f import ForwardModel3F, InverseModel3F
from train_cartpole import _augment_state, _action_for_cerebellum, _smooth


# ======================================================================
# Common utilities
# ======================================================================

def _noaug_state(raw_obs: np.ndarray, cortex_repr: np.ndarray) -> np.ndarray:
    """SANS AUGMENT: concatenate raw observation and cortex representation."""
    return np.concatenate([raw_obs, cortex_repr])


def _bg_state(raw_obs: np.ndarray, cortex_repr: np.ndarray, use_augment: bool) -> np.ndarray:
    """Build BG state with or without derived features."""
    return _augment_state(raw_obs, cortex_repr) if use_augment else _noaug_state(raw_obs, cortex_repr)


def _discrete_from_bg_action(action) -> int:
    """Convert any BG output to a discrete action int."""
    if isinstance(action, np.ndarray):
        return 1 if action[0] > 0 else 0
    return int(action)


def _bg_step_discrete(bg, bg_state: np.ndarray, explore: bool = True) -> int:
    """Universal bg.step() → discrete int (handles linear BG's continuous output)."""
    return _discrete_from_bg_action(bg.step(bg_state, explore=explore))


def _bg_policy_discrete(bg, bg_state: np.ndarray, explore: bool = False) -> int:
    """Universal bg.policy() → discrete int (no side effects)."""
    return _discrete_from_bg_action(bg.policy(bg_state, explore=explore))


def _select_predictive(
    bg, fwd_model, cortex, raw_state: np.ndarray, bg_state: np.ndarray,
    use_augment: bool, explore: bool = True,
) -> int:
    """Forward-model lookahead for any BG type; returns discrete action."""
    current_value = bg.value(bg_state)
    gamma = bg.gamma
    best_delta, best_discrete = -1e10, 0

    for d_action in [0, 1]:
        action_enc = _action_for_cerebellum(d_action)
        pred_next = fwd_model.predict_next_state(raw_state, action_enc)
        pred_cr = cortex.encode(pred_next)
        pred_bgs = _bg_state(pred_next, pred_cr, use_augment)
        delta_star = gamma * bg.value(pred_bgs) - current_value
        if delta_star > best_delta:
            best_delta = delta_star
            best_discrete = d_action

    if explore:
        temp = getattr(bg, "temperature", 0.5)
        if np.random.rand() < min(0.5, temp / 4.0):
            best_discrete = np.random.randint(2)

    return best_discrete


def _run_test(
    cortex, bg, fwd_model, inv_model,
    n_trials: int = 50,
    seed: int = 999,
    use_augment: bool = False,
) -> dict:
    """Test 4 strategies (bg, predictive, inverse, hybrid). Universal BG support."""
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    corrector = CerebellarCorrector(fwd_model, correction_gain=0.3)
    hybrid = HybridController(
        fwd_model, inv_model, corrector, surprise_threshold_factor=2.0,
    )
    results = {"bg": [], "predictive": [], "inverse": [], "hybrid": []}

    for trial in range(n_trials):
        ts = seed + trial

        def _bgs(raw, cr):
            return _bg_state(raw, cr, use_augment)

        def _reset_cortex():
            if hasattr(cortex, "reset_episode"):
                cortex.reset_episode()

        # ---- Strategy 1: BG reactive ----
        obs, _ = env.reset(seed=ts)
        raw = np.array(obs, dtype=np.float64)
        bg.reset()
        _reset_cortex()
        total_r = 0.0
        done = trunc = False
        while not (done or trunc):
            bgs = _bgs(raw, cortex.encode(raw))
            d = _bg_policy_discrete(bg, bgs)
            obs, r, done, trunc, _ = env.step(d)
            total_r += r
            raw = np.array(obs, dtype=np.float64)
        results["bg"].append(total_r)

        # ---- Strategy 2: Predictive (forward-model lookahead) ----
        obs, _ = env.reset(seed=ts)
        raw = np.array(obs, dtype=np.float64)
        bg.reset()
        _reset_cortex()
        total_r = 0.0
        done = trunc = False
        while not (done or trunc):
            cr = cortex.encode(raw)
            bgs = _bgs(raw, cr)
            d = _select_predictive(bg, fwd_model, cortex, raw, bgs, use_augment, explore=False)
            obs, r, done, trunc, _ = env.step(d)
            total_r += r
            raw = np.array(obs, dtype=np.float64)
        results["predictive"].append(total_r)

        # ---- Strategy 3: Inverse model ----
        obs, _ = env.reset(seed=ts)
        raw = np.array(obs, dtype=np.float64)
        _reset_cortex()
        target = np.zeros(4)
        total_r = 0.0
        done = trunc = False
        while not (done or trunc):
            action = inv_model.compute_action(raw, target)
            d = 1 if action[0] > 0 else 0
            obs, r, done, trunc, _ = env.step(d)
            total_r += r
            raw = np.array(obs, dtype=np.float64)
        results["inverse"].append(total_r)

        # ---- Strategy 4: Hybrid ----
        obs, _ = env.reset(seed=ts)
        raw = np.array(obs, dtype=np.float64)
        bg.reset()
        _reset_cortex()
        hybrid.reset()
        total_r = 0.0
        done = trunc = False
        while not (done or trunc):
            cr = cortex.encode(raw)
            bgs = _bgs(raw, cr)
            action, _, _ = hybrid.select_action(raw, bgs, np.zeros(4), bg, observed_state=raw)
            if isinstance(action, (int, np.integer)):
                action = np.array([1.0 if action == 1 else -1.0])
            corrector.begin_step(raw, action)
            d = 1 if action[0] > 0 else 0
            obs, r, done, trunc, _ = env.step(d)
            total_r += r
            raw = np.array(obs, dtype=np.float64)
        results["hybrid"].append(total_r)

    env.close()
    return results


# ======================================================================
# Shared training loop (ACTrace + SANS AUGMENT, optional 3-factor)
# ======================================================================

def _train_actrace_noaug(
    cortex,
    cortex_3f: bool,
    bg: BasalGangliaACTrace,
    fwd_model,
    inv_model,
    cerebellum_3f: bool,
    mode: str,
    n_episodes: int,
    seed: int,
    print_every: int = 100,
    label: str = "",
):
    """Training loop for ACTrace BG, SANS AUGMENT, optional 3-factor.

    The cortex can be CerebralCortexImproved or CerebralCortexImproved3F.
    """
    np.random.seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = 4

    initial_temp, final_temp = 2.0, 0.5
    prev_delta = 0.0

    ep_rewards, ep_cx, ep_fwd, ep_inv, ep_td = [], [], [], [], []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        raw_state = np.array(obs, dtype=np.float64)
        bg.reset()
        if cortex_3f:
            cortex.reset_episode()
        elif hasattr(cortex, "reset_episode"):
            cortex.reset_episode()

        progress = episode / n_episodes
        bg.temperature = initial_temp * (1 - progress) + final_temp * progress
        prev_delta = 0.0

        step_cx, step_fwd, step_inv, step_td = [], [], [], []
        total_reward = 0.0

        # First encode
        if cortex_3f:
            cortex_repr, cx_err = cortex.encode_and_learn(raw_state, delta=prev_delta)
        else:
            cortex_repr, cx_err = cortex.encode_and_learn(raw_state)
        step_cx.append(cx_err)
        bg_state = _noaug_state(raw_state, cortex_repr)

        if mode == "reactive":
            discrete = int(bg.step(bg_state, explore=True))
        else:
            bg._prev_features = bg_state.copy()
            bg._prev_value = bg.value(bg_state)
            discrete = _select_predictive(bg, fwd_model, cortex, raw_state, bg_state,
                                          use_augment=False, explore=True)

        continuous = np.array([1.0 if discrete == 1 else -1.0])
        done = trunc = False

        while not (done or trunc):
            obs, reward, done, trunc, _ = env.step(discrete)
            raw_next = np.array(obs, dtype=np.float64)
            total_reward += reward
            terminal = done or trunc
            bg_reward = reward if not done else -10.0

            next_cr = cortex.encode(raw_next)
            next_bgs = _noaug_state(raw_next, next_cr)

            if mode == "predictive":
                delta = bg.learn(bg_reward, next_bgs, terminal, action_override=discrete)
            else:
                delta = bg.learn(bg_reward, next_bgs, terminal)
            step_td.append(abs(delta))

            action_enc = _action_for_cerebellum(discrete)
            if cerebellum_3f:
                step_fwd.append(fwd_model.learn_transition(raw_state, action_enc, raw_next, delta=delta))
                step_inv.append(inv_model.learn_action(raw_state, np.zeros(obs_dim), continuous, delta=delta))
            else:
                step_fwd.append(fwd_model.learn_transition(raw_state, action_enc, raw_next))
                step_inv.append(inv_model.learn_action(raw_state, np.zeros(obs_dim), continuous))

            prev_delta = delta

            if not terminal:
                raw_state = raw_next
                if cortex_3f:
                    cortex_repr, cx_err = cortex.encode_and_learn(raw_state, delta=prev_delta)
                else:
                    cortex_repr, cx_err = cortex.encode_and_learn(raw_state)
                step_cx.append(cx_err)
                bg_state = _noaug_state(raw_state, cortex_repr)

                if mode == "reactive":
                    discrete = int(bg.step(bg_state, explore=True))
                else:
                    bg._prev_features = bg_state.copy()
                    bg._prev_value = bg.value(bg_state)
                    discrete = _select_predictive(bg, fwd_model, cortex, raw_state, bg_state,
                                                  use_augment=False, explore=True)

                continuous = np.array([1.0 if discrete == 1 else -1.0])

        ep_rewards.append(total_reward)
        ep_cx.append(np.mean(step_cx))
        ep_fwd.append(np.mean(step_fwd) if step_fwd else 0.0)
        ep_inv.append(np.mean(step_inv) if step_inv else 0.0)
        ep_td.append(np.mean(step_td) if step_td else 0.0)

        if episode % print_every == 0 or episode == 1:
            w = min(print_every, episode)
            print(f"  {label} [{mode}] Ep {episode:4d} | "
                  f"R {np.mean(ep_rewards[-w:]):7.1f} | "
                  f"DA {np.mean(ep_td[-w:]):.3f} | "
                  f"Cx {np.mean(ep_cx[-w:]):.3f}")

    env.close()
    metrics = {
        "rewards": ep_rewards, "cortex_errors": ep_cx,
        "forward_errors": ep_fwd, "inverse_errors": ep_inv,
        "td_magnitudes": ep_td,
    }
    return metrics


# ======================================================================
#  3-Factor with improved cortex, SANS AUGMENT
# ======================================================================

EXP4_CONDITIONS = ["2f", "3f_cortex", "3f_cb", "3f_full"]
EXP4_LABELS = {
    "2f":        "2F (Hebb + LTD, Doya strict)",
    "3f_cortex": "3F-Cortex (DA gate cortex amélioré)",
    "3f_cb":     "3F-Cervelet (DA gate LTD)",
    "3f_full":   "3F-Full (DA gate Hebb + LTD)",
}
EXP4_COLORS = {
    "2f":        "#2196F3",
    "3f_cortex": "#4CAF50",
    "3f_cb":     "#FF9800",
    "3f_full":   "#E91E63",
}
EXP4_FLAGS = {
    "2f":        (False, False),
    "3f_cortex": (True,  False),
    "3f_cb":     (False, True),
    "3f_full":   (True,  True),
}
EXP4_MODES = ["reactive", "predictive"]
EXP4_MODE_LABELS = {
    "reactive":   "Section 4.1 — Reactive",
    "predictive": "Section 4.2 — Predictive",
}


def _make_models_exp4(cortex_3f: bool, cerebellum_3f: bool):
    obs_dim = 4
    repr_dim = 8  # improved cortex output dim
    state_dim = obs_dim + repr_dim  # 12D

    if cortex_3f:
        cortex = CerebralCortexImproved3F(
            input_dim=obs_dim, n_frames=4, repr_dim=repr_dim, hidden_dim=16,
            lr=0.005, sparsity=0.1, relaxation_steps=20, relaxation_dt=0.05,
            delta_clip=10.0, delta_baseline_ema=0.1,
        )
    else:
        cortex = CerebralCortexImproved(
            input_dim=obs_dim, n_frames=4, repr_dim=repr_dim, hidden_dim=16,
            lr=0.005, sparsity=0.1, relaxation_steps=20, relaxation_dt=0.05,
        )

    bg = BasalGangliaACTrace(
        state_dim=state_dim, n_actions=2,
        gamma=0.99, lr_critic=1e-3, lr_actor=1e-3,
        decay_lambda=0.9, entropy_beta=0.01,
        temperature=2.0, delta_clip=10.0, max_weight=5.0, device="cpu",
    )

    if cerebellum_3f:
        fwd_model = ForwardModel3F(
            state_dim=obs_dim, action_dim=1,
            n_granule=256, lr=0.01, grad_clip=2.0,
            da_lr_scale=2.0, da_baseline_ema=0.05,
        )
        inv_model = InverseModel3F(
            state_dim=obs_dim, action_dim=1,
            n_granule=256, lr=0.01, grad_clip=2.0,
            da_lr_scale=2.0, da_baseline_ema=0.05,
        )
    else:
        fwd_model = ForwardModel(state_dim=obs_dim, action_dim=1, n_granule=256, lr=0.01, grad_clip=2.0)
        inv_model = InverseModel(state_dim=obs_dim, action_dim=1, n_granule=256, lr=0.01, grad_clip=2.0)

    return cortex, bg, fwd_model, inv_model


def run_experiment_4(n_episodes: int, seed: int) -> dict:
    all_results = {cond: {} for cond in EXP4_CONDITIONS}

    for cond in EXP4_CONDITIONS:
        cortex_3f, cerebellum_3f = EXP4_FLAGS[cond]
        for mode in EXP4_MODES:
            print(f"\n  [Exp 4] Condition: {cond}  mode: {mode}")
            t0 = time.time()

            cortex, bg, fwd_model, inv_model = _make_models_exp4(cortex_3f, cerebellum_3f)

            metrics = _train_actrace_noaug(
                cortex=cortex, cortex_3f=cortex_3f,
                bg=bg, fwd_model=fwd_model, inv_model=inv_model,
                cerebellum_3f=cerebellum_3f, mode=mode,
                n_episodes=n_episodes, seed=seed,
                label=f"Exp4/{cond}",
            )
            elapsed = time.time() - t0
            test = _run_test(cortex, bg, fwd_model, inv_model,
                             n_trials=50, seed=999, use_augment=False)
            all_results[cond][mode] = {
                "metrics": metrics, "test_results": test,
                "elapsed": elapsed, "seed": seed,
            }

    return all_results


# ======================================================================
# Comparison plot (GridSpec 3×2)
# ======================================================================

STRATEGY_LABELS = {
    "bg":         "Basal Ganglia\n(reactive)",
    "predictive": "Predictive\n(forward model)",
    "inverse":    "Inverse model\n(cerebellum)",
    "hybrid":     "Hybrid\n(switch)",
}
STRATEGY_COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
STRATEGIES = ["bg", "predictive", "inverse", "hybrid"]


def _plot_comparison_final(
    all_results: dict,
    conditions: list,
    col_keys: list,
    condition_labels: dict,
    condition_colors: dict,
    col_labels: dict,
    title: str,
    filename: str,
):
    """Generic GridSpec 3×2 comparison plot.

    Row 0: learning curves, one subplot per col_key.
    Row 1: TD error (left) and cortex reconstruction error (right).
    Row 2: test bars — strategies × conditions × col_keys.
    """
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    # ------------------------------------------------------------------
    # Row 0: Learning curves per column
    # ------------------------------------------------------------------
    for col, col_key in enumerate(col_keys):
        ax = fig.add_subplot(gs[0, col])
        for cond in conditions:
            rewards = all_results[cond][col_key]["metrics"]["rewards"]
            w = min(50, len(rewards) // 2) or 1
            eps = np.arange(1, len(rewards) + 1)
            sx = np.arange(w, len(rewards) + 1)
            ax.plot(eps, rewards, alpha=0.1, color=condition_colors[cond], linewidth=0.5)
            ax.plot(sx, _smooth(rewards, w), color=condition_colors[cond],
                    linewidth=2.5, label=condition_labels[cond])
        ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
        ax.set_ylabel("Steps survived")
        ax.set_xlabel("Episode")
        ax.set_title(f"Courbes d'apprentissage — {col_labels[col_key]}")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 1: TD error and cortex reconstruction error
    # ------------------------------------------------------------------
    for col, (metric_key, ylabel, panel_title) in enumerate([
        ("td_magnitudes", "|δ| (signal DA)", "TD error — signal dopaminergique"),
        ("cortex_errors", "||h - W'y||²", "Erreur reconstruction cortex"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        for cond in conditions:
            for ci, col_key in enumerate(col_keys):
                data = all_results[cond][col_key]["metrics"][metric_key]
                w = min(50, len(data) // 2) or 1
                eps = np.arange(1, len(data) + 1)
                sx = np.arange(w, len(data) + 1)
                ls = "-" if ci == 0 else "--"
                label = f"{condition_labels[cond].split('(')[0].strip()} [{col_key[:4]}]"
                ax.plot(eps, data, alpha=0.05, color=condition_colors[cond], linewidth=0.5)
                ax.plot(sx, _smooth(data, w), color=condition_colors[cond],
                        linewidth=2.0, linestyle=ls, label=label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode")
        ax.set_title(panel_title)
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 2: Test bars — strategies × conditions × col_keys
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    n_groups = len(conditions) * len(col_keys)
    bar_width = 0.09
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * bar_width
    x = np.arange(len(STRATEGIES))

    bar_idx = 0
    for cond in conditions:
        for ci, col_key in enumerate(col_keys):
            test = all_results[cond][col_key]["test_results"]
            means = [np.mean(test[s]) for s in STRATEGIES]
            stds = [np.std(test[s]) for s in STRATEGIES]
            hatch = None if ci == 0 else "//"
            alpha = 0.80 if ci == 0 else 0.55
            label = f"{condition_labels[cond].split('(')[0].strip()} [{col_key[:4]}]"
            bars = ax.bar(
                x + offsets[bar_idx], means, bar_width,
                yerr=stds, color=condition_colors[cond],
                alpha=alpha, hatch=hatch, capsize=3,
                edgecolor="black", linewidth=0.4, label=label,
            )
            for bar, m in zip(bars, means):
                if m > 30:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5,
                        f"{m:.0f}",
                        ha="center", va="bottom", fontsize=6, fontweight="bold",
                    )
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in STRATEGIES], fontsize=10)
    ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
    ax.set_ylabel("Steps survived (50 trials)")
    ax.set_title(
        "Phase test — Stratégie × Condition × Colonne\n"
        "(plein = col. 0, hachuré = col. 1)"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


# ======================================================================
# Main benchmark function
# ======================================================================

def benchmark_final(n_episodes: int = 1500, seed: int = 42):
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print()
    print("+" + "=" * 70 + "+")
    print("|  3-Factor avec cortex amélioré, SANS AUGMENT          |")
    print("|  2F / 3F-Cortex / 3F-Cérébellum / 3F-Full | CerebralCortexImproved  |")
    print("+" + "=" * 70 + "+")
    all_results_4 = run_experiment_4(n_episodes, seed)
    _plot_comparison_final(
        all_results=all_results_4,
        conditions=EXP4_CONDITIONS,
        col_keys=EXP4_MODES,
        condition_labels=EXP4_LABELS,
        condition_colors=EXP4_COLORS,
        col_labels=EXP4_MODE_LABELS,
        title=(
            "Benchmark Final — CartPole-v1\n"
            "3-Factor avec CerebralCortexImproved (8D) | SANS AUGMENT"
        ),
        filename=os.path.join(output_dir, "report_vfinal_comparison.png"),
    )

    print()
    print("  Fichier généré :")
    print("    report_vfinal_comparison.png")
    print()
    return all_results_4


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Final sur CartPole-v1"
    )
    parser.add_argument("--episodes", "-n", type=int, default=1500,
                        help="Number of training episodes per condition (default: 1500)")
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    print()
    print("+" + "=" * 70 + "+")
    print("|  BENCHMARK FINAL — Expérience 4, CartPole-v1, SANS AUGMENT           |")
    print("+" + "=" * 70 + "+")
    print(f"|  Episodes: {args.episodes:<59d}|")
    print(f"|  Seed:     {args.seed:<59d}|")
    print("|  Exp 4: 3-Factor + cortex amélioré (4 conditions, 2 modes)           |")
    print("+" + "=" * 70 + "+")

    benchmark_final(n_episodes=args.episodes, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
