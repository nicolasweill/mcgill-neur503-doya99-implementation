#!/usr/bin/env python3
"""
Ablation & Neuropathology Benchmark v2 — 3F-full, no augmented state
=====================================================================

Identical to benchmark_ablations.py except:
  - Uses CerebralCortex3F (three-factor neuromodulated Hebbian learning)
  - Uses ForwardModel3F + InverseModel3F (dopamine-gated cerebellar LTD)
  - BG state = obs (4D) + cortex_repr (6D) = 10D  [no handcrafted features]
  - Each condition trained in BOTH reactive AND predictive mode
    (new Panel G: learning curves reactive vs predictive)

Validates that each module (cortex, BG, cerebellum) plays its assigned role
by measuring performance degradation when each is selectively impaired.
Simulates two clinical conditions: Parkinson's disease and cerebellar ataxia.

Training-time ablations (each trains from scratch, in reactive + predictive mode):
  intact           - full ACTrace + CerebralCortex3F + full cerebellum3F
  no_cortex        - CerebralCortex3F bypassed (BG sees zero cortical repr)
  no_inverse       - inverse cerebellar model disabled (lr=0)
  park_mild        - Parkinson stage 1: BG learning rates x0.75
  park_moderate    - Parkinson stage 2: BG learning rates x0.40
  park_severe      - Parkinson stage 3: BG learning rates x0.15

Test-time modifications applied to the intact trained model:
  no_fwd_test      - forward model pc_weights zeroed (predictive strategy fails)
  ataxia_test      - Gaussian noise on forward+inverse pc_weights (dysmetria)
  park_late_test   - BG actor+critic weights zeroed after intact training
                     (Parkinson onset AFTER skills consolidated in cerebellum)

Key comparisons:
  park_severe vs park_late_test  =>  encapsulation protection
  intact vs no_cortex            =>  role of cortical representations
  intact vs no_inverse           =>  role of cerebellar encapsulation
  intact vs no_fwd_test          =>  role of forward model for planning
  intact vs ataxia_test          =>  BG/cerebellum dissociation

Usage
-----
    python benchmark_ablationsv2.py                  # 1500 episodes (default)
    python benchmark_ablationsv2.py --episodes 300   # fast smoke test
    python benchmark_ablationsv2.py --seed 7
"""

import sys
import os
import copy
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

from cerebral_cortex_3f import CerebralCortex3F
from basal_ganglia_actrace import BasalGangliaACTrace
from cerebellum import CerebellarCorrector, HybridController
from cerebellum_3f import ForwardModel3F, InverseModel3F
from train_cartpole import _action_for_cerebellum, _smooth


# ======================================================================
# NullCortex — cortical ablation
# ======================================================================

class NullCortex:
    """Cortical ablation: returns a zero representation.

    Simulates a partial cortical lesion where the cortex fails to produce
    any useful representation. The BG still receives the raw state
    (via concatenation) but the cortical repr dims are all zero.

    This tests whether the learned cortical representation is necessary,
    or if the raw features alone are sufficient.
    """

    def __init__(self, repr_dim: int = 6):
        self.repr_dim = repr_dim

    def encode_and_learn(self, x: np.ndarray, delta=None):
        """Return zero repr and zero reconstruction error."""
        return np.zeros(self.repr_dim), 0.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(self.repr_dim)


# ======================================================================
# Training loop — ACTrace with ablation hooks
# ======================================================================

def train_ablation(
    n_episodes: int = 800,
    seed: int = 42,
    print_every: int = 200,
    # Ablation flags
    cortex_disabled: bool = False,
    inverse_disabled: bool = False,
    delta_scale: float = 1.0,        # Parkinson: scale BG learning rates
    # Training mode
    training_mode: str = "predictive",  # "reactive" or "predictive"
):
    """Train the Doya (1999) model with ACTrace BG, with optional ablations.

    Uses 3F modules. BG state = obs (4D) + cortex features (6D) = 10D.

    Parameters
    ----------
    cortex_disabled : bool
        If True, CerebralCortex3F is replaced by NullCortex.
        BG receives raw obs with zero cortical dims.
    inverse_disabled : bool
        If True, inverse cerebellar model learning rate = 0.
        Encapsulation never occurs.
    delta_scale : float
        Multiplies BG learning rates to simulate dopamine reduction.
        1.0 = intact, 0.75 = mild Parkinson, 0.40 = moderate, 0.15 = severe.
    training_mode : str
        "reactive"   — actions selected by BG stochastic policy (ACTrace).
        "predictive" — actions selected by forward model lookahead:
                       both actions evaluated via fwd_model + BG critic,
                       best action taken (still trains BG via TD).

    Returns
    -------
    cortex, bg, fwd_model, inv_model, episode_rewards, episode_lengths, metrics
    """
    np.random.seed(seed)
    env = gym.make("CartPole-v1")

    obs_dim = 4
    repr_dim = 6
    action_dim = 1
    state_dim = obs_dim + repr_dim   # 10D: 4 raw obs + 6 cortical

    # --- Cortex ---
    if cortex_disabled:
        cortex = NullCortex(repr_dim=repr_dim)
    else:
        cortex = CerebralCortex3F(
            input_dim=obs_dim, repr_dim=repr_dim,
            lr=0.005, sparsity=0.02,
            relaxation_steps=10, relaxation_dt=0.03,
        )

    # --- Basal Ganglia: ACTrace ---
    # Scale learning rates to simulate dopamine signal reduction (Parkinson)
    bg = BasalGangliaACTrace(
        state_dim=state_dim, n_actions=2,
        gamma=0.99,
        lr_critic=1e-3 * delta_scale,
        lr_actor=1e-3 * delta_scale,
        decay_lambda=0.9,
        entropy_beta=0.01,
        temperature=2.0,
        delta_clip=10.0, max_weight=5.0,
        device="cpu",
    )

    # --- Cerebellar models (3F: dopamine-gated LTD) ---
    fwd_model = ForwardModel3F(
        state_dim=obs_dim, action_dim=action_dim,
        n_granule=256, lr=0.01, grad_clip=2.0,
    )
    inv_model = InverseModel3F(
        state_dim=obs_dim, action_dim=action_dim,
        n_granule=256, lr=0.01, grad_clip=2.0,
    )

    if inverse_disabled:
        inv_model.lr = 0.0   # Purkinje cells for inverse model never update

    # --- Tracking ---
    episode_rewards, episode_lengths = [], []
    cortex_errors, forward_errors, inverse_errors, td_magnitudes = [], [], [], []

    initial_temperature = 2.0
    final_temperature = 0.5

    def _select_action(raw_s, bg_s):
        """Select action according to training_mode."""
        if training_mode == "predictive":
            # step() must be called to register _prev_features/_prev_value;
            # without it bg.learn() early-returns 0.0 (no learning at all).
            bg.step(bg_s, explore=True)   # state recorded; BG action discarded
            best_d, best_val = 0, -1e10
            for d_action in [0, 1]:
                action_enc = _action_for_cerebellum(d_action)
                pred_next = fwd_model.predict_next_state(raw_s, action_enc)
                pred_cortex = cortex.encode(pred_next)
                pred_bg = np.concatenate([pred_next, pred_cortex])
                val = bg.gamma * bg.value(pred_bg) - bg.value(bg_s)
                if val > best_val:
                    best_val, best_d = val, d_action
            return best_d
        else:
            return int(bg.step(bg_s, explore=True))

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        raw_state = np.array(obs, dtype=np.float64)
        bg.reset()

        progress = episode / n_episodes
        bg.temperature = initial_temperature * (1 - progress) + final_temperature * progress

        ep_cx, ep_fwd, ep_inv, ep_td = [], [], [], []
        total_reward = 0.0
        prev_delta = None   # synaptic tag: δ from previous step

        cortex_repr, cx_err = cortex.encode_and_learn(raw_state, delta=None)
        ep_cx.append(cx_err)
        bg_state = np.concatenate([raw_state, cortex_repr])

        discrete_action = _select_action(raw_state, bg_state)
        continuous_action = np.array([1.0 if discrete_action == 1 else -1.0])

        done = False
        truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, _ = env.step(discrete_action)
            raw_next = np.array(obs, dtype=np.float64)
            total_reward += reward

            # Cortex encodes next state; δ from previous step gates plasticity
            next_cortex_repr, cx_err = cortex.encode_and_learn(raw_next, delta=prev_delta)
            ep_cx.append(cx_err)
            next_bg_state = np.concatenate([raw_next, next_cortex_repr])

            terminal = done or truncated
            bg_reward = reward if not done else -10.0
            # In predictive mode, pass the lookahead action so the actor
            # eligibility traces are computed for the action actually taken.
            _override = discrete_action if training_mode == "predictive" else None
            delta = bg.learn(bg_reward, next_bg_state, terminal, action_override=_override)
            ep_td.append(abs(delta))

            action_enc = _action_for_cerebellum(discrete_action)
            fwd_err = fwd_model.learn_transition(raw_state, action_enc, raw_next, delta=delta)
            ep_fwd.append(fwd_err)

            inv_err = inv_model.learn_action(
                raw_state, np.zeros(obs_dim), continuous_action, delta=delta
            )
            ep_inv.append(inv_err)

            prev_delta = delta   # store for synaptic tag on next cortex encode

            if not terminal:
                raw_state = raw_next
                bg_state = next_bg_state
                discrete_action = _select_action(raw_state, bg_state)
                continuous_action = np.array([1.0 if discrete_action == 1 else -1.0])

        episode_rewards.append(total_reward)
        episode_lengths.append(total_reward)
        cortex_errors.append(np.mean(ep_cx))
        forward_errors.append(np.mean(ep_fwd) if ep_fwd else 0)
        inverse_errors.append(np.mean(ep_inv) if ep_inv else 0)
        td_magnitudes.append(np.mean(ep_td) if ep_td else 0)

        if episode % print_every == 0 or episode == 1:
            w = min(print_every, episode)
            print(
                f"  [{training_mode}] Ep {episode:4d} | "
                f"R {np.mean(episode_rewards[-w:]):6.1f} | "
                f"DA {np.mean(td_magnitudes[-w:]):.3f} | "
                f"Cx {np.mean(cortex_errors[-w:]):.3f} | "
                f"Fwd {np.mean(forward_errors[-w:]):.4f}"
            )

    env.close()

    metrics = {
        "rewards": episode_rewards,
        "cortex_errors": cortex_errors,
        "forward_errors": forward_errors,
        "inverse_errors": inverse_errors,
        "td_magnitudes": td_magnitudes,
    }
    return cortex, bg, fwd_model, inv_model, episode_rewards, episode_lengths, metrics


# ======================================================================
# Test loop — with optional post-hoc modifications
# ======================================================================

def run_test_ablation(
    cortex,
    bg,
    fwd_model: ForwardModel3F,
    inv_model: InverseModel3F,
    n_trials: int = 50,
    seed: int = 999,
    # Post-hoc test-time modifications
    bg_zeroed: bool = False,          # Parkinson late onset (zero BG weights)
    fwd_zeroed: bool = False,          # Forward model ablation at test
    ataxia_sigma: float = 0.0,        # Cerebellar ataxia: noise on pc_weights
) -> dict:
    """Evaluate 4 control strategies with optional test-time modifications.

    All modifications are applied to deep copies to preserve the originals.

    Returns
    -------
    dict with keys "bg", "predictive", "inverse", "hybrid"
    """
    import torch

    np.random.seed(seed)
    env = gym.make("CartPole-v1")

    # Deep copies so we never corrupt the trained models
    bg_test   = copy.deepcopy(bg)
    fwd_test  = copy.deepcopy(fwd_model)
    inv_test  = copy.deepcopy(inv_model)

    # --- Apply post-hoc modifications ---
    if bg_zeroed:
        # Simulate Parkinson onset AFTER cerebellar encapsulation:
        # BG actor + critic weights zeroed → policy outputs uniform distribution
        with torch.no_grad():
            for p in bg_test.actor.parameters():
                p.data.zero_()
            for p in bg_test.critic.parameters():
                p.data.zero_()

    if fwd_zeroed:
        # Forward model ablation: pc_weights = 0 → predicts zeros
        fwd_test.pc_weights[:] = 0.0
        fwd_test.pc_bias[:]    = 0.0

    if ataxia_sigma > 0.0:
        # Cerebellar ataxia: Gaussian noise on Purkinje cell weights
        # Simulates LTD disruption → dysmetric predictions
        fwd_test.pc_weights += np.random.randn(*fwd_test.pc_weights.shape) * ataxia_sigma
        inv_test.pc_weights += np.random.randn(*inv_test.pc_weights.shape) * ataxia_sigma

    corrector = CerebellarCorrector(fwd_test, correction_gain=0.3)
    hybrid    = HybridController(fwd_test, inv_test, corrector, surprise_threshold_factor=2.0)

    results = {"bg": [], "predictive": [], "inverse": [], "hybrid": []}

    for trial in range(n_trials):
        trial_seed = seed + trial

        # Strategy 1: BG reactive policy
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        bg_test.reset()

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            cortex_repr = cortex.encode(raw_state)
            bg_state = np.concatenate([raw_state, cortex_repr])
            discrete = bg_test.policy(bg_state, explore=False)
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
        results["bg"].append(total_r)

        # Strategy 2: Predictive (forward model lookahead)
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        bg_test.reset()

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            cortex_repr = cortex.encode(raw_state)
            bg_state = np.concatenate([raw_state, cortex_repr])
            # Evaluate both discrete actions via forward model
            best_d, best_val = 0, -1e10
            for d_action in [0, 1]:
                action_enc = _action_for_cerebellum(d_action)
                pred_next = fwd_test.predict_next_state(raw_state, action_enc)
                pred_cortex = cortex.encode(pred_next)
                pred_bg = np.concatenate([pred_next, pred_cortex])
                val = bg_test.gamma * bg_test.value(pred_bg) - bg_test.value(bg_state)
                if val > best_val:
                    best_val, best_d = val, d_action
            obs, r, done, truncated, _ = env.step(best_d)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
        results["predictive"].append(total_r)

        # Strategy 3: Inverse model (cerebellar autonomous control)
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        target_state = np.zeros(4)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = inv_test.compute_action(raw_state, target_state)
            discrete = 1 if action[0] > 0 else 0
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
        results["inverse"].append(total_r)

        # Strategy 4: Hybrid (adaptive BG / cerebellum switch)
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        hybrid.reset()

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            cortex_repr = cortex.encode(raw_state)
            bg_state = np.concatenate([raw_state, cortex_repr])
            action, _, _ = hybrid.select_action(
                raw_state, bg_state, np.zeros(4), bg_test,
                observed_state=raw_state,
            )
            if isinstance(action, (int, np.integer)):
                action = np.array([1.0 if action == 1 else -1.0])
            corrector.begin_step(raw_state, action)
            discrete = 1 if action[0] > 0 else 0
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
        results["hybrid"].append(total_r)

    env.close()
    return results


# ======================================================================
# Plotting helpers
# ======================================================================

STRATEGY_NAMES  = ["bg", "predictive", "inverse", "hybrid"]
STRATEGY_LABELS = ["BG reactive\n(ACTrace)", "Predictive\n(forward model)", "Inverse\n(cerebellum)", "Hybrid\n(switch)"]
STRATEGY_COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]


def _sw(n_ep, window=50):
    """Safe smoothing window — avoids empty arrays for short runs."""
    return min(window, n_ep // 2) or 1


def plot_learning_curves(all_train: dict, filename: str, mode_label: str = "reactive"):
    """Panel A+B: Learning curves for structural ablations and Parkinson stages."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Doya (1999) + ACTrace [3F-full, {mode_label}] -- Ablation Learning Curves (CartPole-v1)",
        fontsize=14, fontweight="bold",
    )

    # Panel A: structural ablations
    structural_conditions = {
        "intact":      ("#2196F3", "Intact (reference)"),
        "no_cortex":   ("#FF5722", "No cortex (lesion corticale)"),
        "no_inverse":  ("#9C27B0", "No inverse model (no encapsulation)"),
    }
    ax = axes[0]
    for cond, (color, label) in structural_conditions.items():
        if cond not in all_train:
            continue
        rewards = all_train[cond]["metrics"]["rewards"]
        w = _sw(len(rewards))
        eps = np.arange(1, len(rewards) + 1)
        sx  = np.arange(w, len(rewards) + 1)
        ax.plot(eps, rewards, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(sx, _smooth(rewards, w), color=color, linewidth=2.5, label=label)
    ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
    ax.set_title("A — Structural ablations")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps survived")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Parkinson stages
    park_conditions = {
        "intact":          ("#2196F3", "Intact (reference)"),
        "park_mild":       ("#FFC107", "Parkinson stage 1 (delta x0.75)"),
        "park_moderate":   ("#FF9800", "Parkinson stage 2 (delta x0.40)"),
        "park_severe":     ("#F44336", "Parkinson stage 3 (delta x0.15)"),
    }
    ax = axes[1]
    for cond, (color, label) in park_conditions.items():
        if cond not in all_train:
            continue
        rewards = all_train[cond]["metrics"]["rewards"]
        w = _sw(len(rewards))
        eps = np.arange(1, len(rewards) + 1)
        sx  = np.arange(w, len(rewards) + 1)
        ax.plot(eps, rewards, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(sx, _smooth(rewards, w), color=color, linewidth=2.5, label=label)
    ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
    ax.set_title("B -- Parkinson progression (reduction du signal TD)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps survived")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


def plot_training_modes(all_train_reactive: dict, all_train_predictive: dict, filename: str):
    """Panel G: Reactive vs Predictive training — learning curve comparison.

    For each ablation group (structural + Parkinson), plots the smoothed
    learning curve in reactive mode (solid) and predictive mode (dashed)
    with the same color. Allows direct comparison of how action-selection
    strategy during training affects final performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Doya (1999) + ACTrace [3F-full] -- Reactive vs Predictive Training (CartPole-v1)",
        fontsize=14, fontweight="bold",
    )

    condition_groups = [
        # (panel_idx, title, conditions)
        (0, "G1 — Structural ablations: reactive (—) vs predictive (--)", {
            "intact":      "#2196F3",
            "no_cortex":   "#FF5722",
            "no_inverse":  "#9C27B0",
        }),
        (1, "G2 — Parkinson stages: reactive (—) vs predictive (--)", {
            "intact":        "#2196F3",
            "park_mild":     "#FFC107",
            "park_moderate": "#FF9800",
            "park_severe":   "#F44336",
        }),
    ]

    mode_label_suffix = {"reactive": " (reactive)", "predictive": " (predictive)"}

    for panel_idx, title, conditions in condition_groups:
        ax = axes[panel_idx]
        for cond, color in conditions.items():
            for mode, ls, src in [
                ("reactive",   "-",  all_train_reactive),
                ("predictive", "--", all_train_predictive),
            ]:
                if cond not in src:
                    continue
                rewards = src[cond]["metrics"]["rewards"]
                w = _sw(len(rewards))
                sx = np.arange(w, len(rewards) + 1)
                label = cond + mode_label_suffix[mode] if mode == "reactive" else None
                ax.plot(sx, _smooth(rewards, w), color=color, linewidth=2.0,
                        linestyle=ls, label=label, alpha=0.9)
        ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps survived")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")



def plot_parkinson_encapsulation(all_test: dict, filename: str):
    """Panel D: Parkinson + cerebellar encapsulation protection.

    Key prediction: strategies that rely on the cerebellum (inverse, hybrid)
    survive late-onset Parkinson (park_late_test) because encapsulation
    occurred BEFORE the lesion. They do NOT survive early-onset Parkinson
    (park_severe) because the cerebellum also trained with impaired delta.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Doya (1999) [3F-full] -- Parkinson: encapsulation protects cerebellar strategies\n"
        "Prediction: inverse/hybrid survive late-onset PD but not early-onset PD",
        fontsize=13, fontweight="bold",
    )

    comp_conditions = [
        ("intact",         "#2196F3", "Intact (reference)"),
        ("park_severe",    "#F44336", "PD stage 3 (trained from scratch)"),
        ("park_late_test", "#E91E63", "PD stage 3 LATE (encapsulation before lesion)"),
    ]

    for col, (strat_key, strat_label) in enumerate([
        ("bg",      "Strategy: BG reactive"),
        ("inverse", "Strategy: Inverse cerebellar"),
    ]):
        ax = axes[col]
        bar_x = np.arange(len(comp_conditions))
        for i, (cond, color, label) in enumerate(comp_conditions):
            if cond not in all_test:
                continue
            vals = all_test[cond][strat_key]
            mean, std = np.mean(vals), np.std(vals)
            bar = ax.bar(i, mean, color=color, alpha=0.85,
                         yerr=std, capsize=6, edgecolor="black", linewidth=0.5,
                         label=label)
            ax.text(i, mean + std + 10, f"{mean:.0f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.axhline(500, color="gray", ls="--", alpha=0.4)
        ax.set_xticks(bar_x)
        ax.set_xticklabels(
            [label for _, _, label in comp_conditions],
            fontsize=8, rotation=15, ha="right",
        )
        ax.set_ylabel("Steps survived (50 trials)")
        ax.set_title(f"D{col+1} -- {strat_label}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 560)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


def plot_ataxia_dissociation(all_test: dict, filename: str):
    """Panel E: Cerebellar ataxia vs intact — BG/cerebellum dissociation.

    Key prediction: BG reactive strategy is unaffected by ataxia, while
    predictive and inverse strategies degrade (forward/inverse models impaired).
    This confirms the anatomical dissociation between BG and cerebellum.
    """
    comp_conditions = [
        ("intact",     "#2196F3", "Intact"),
        ("no_inverse", "#9C27B0", "No inverse model"),
        ("no_fwd_test","#795548", "Forward model ablated"),
        ("ataxia_test","#607D8B", "Cerebellar ataxia (noise)"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Doya (1999) [3F-full] -- Cerebellar ataxia: BG/cerebellum dissociation\n"
        "Prediction: BG reactive survives ataxia; predictive/inverse degrade",
        fontsize=13, fontweight="bold",
    )

    n_strat = len(STRATEGY_NAMES)
    n_cond  = sum(1 for (c, _, _) in comp_conditions if c in all_test)
    bar_width = 0.22
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * bar_width
    x = np.arange(n_strat)

    legend_handles = []
    for i, (cond, color, label) in enumerate(comp_conditions):
        if cond not in all_test:
            continue
        test = all_test[cond]
        means = [np.mean(test[s]) for s in STRATEGY_NAMES]
        stds  = [np.std(test[s])  for s in STRATEGY_NAMES]
        ax.bar(
            x + offsets[i], means, bar_width,
            yerr=stds, color=color, alpha=0.85,
            capsize=4, edgecolor="black", linewidth=0.4,
        )
        for bar_x, m in zip(x + offsets[i], means):
            ax.text(bar_x, m + 8, f"{m:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))

    ax.axhline(500, color="gray", ls="--", alpha=0.4, label="Max (500)")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGY_LABELS, fontsize=10)
    ax.set_ylabel("Steps survived (50 trials)")
    ax.set_title("E -- Cerebellar ataxia: strategy-specific deficits")
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


def plot_error_traces(all_train: dict, filename: str, mode_label: str = "reactive"):
    """Panel F: Module error traces across conditions.

    Shows how TD error, cortex error, and forward model error differ
    between intact, cortex-ablated, and Parkinson conditions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Doya (1999) + ACTrace [3F-full, {mode_label}] -- Module error traces across ablation conditions",
        fontsize=13, fontweight="bold",
    )

    metrics_to_plot = [
        ("td_magnitudes",  "|delta| (DA signal)",           "TD error (dopaminergic)"),
        ("cortex_errors",  "||x - W'y||^2",                  "Cortex reconstruction error"),
        ("forward_errors", "MSE forward model",              "Cerebellar forward model error"),
    ]

    condition_styles = {
        "intact":        ("#2196F3", "-",  "Intact"),
        "no_cortex":     ("#FF5722", "-",  "No cortex"),
        "park_mild":     ("#FFC107", "--", "PD stage 1"),
        "park_moderate": ("#FF9800", "--", "PD stage 2"),
        "park_severe":   ("#F44336", "--", "PD stage 3"),
    }

    for col, (metric, ylabel, title) in enumerate(metrics_to_plot):
        ax = axes[col]
        for cond, (color, ls, label) in condition_styles.items():
            if cond not in all_train:
                continue
            data = all_train[cond]["metrics"][metric]
            w = _sw(len(data))
            eps = np.arange(1, len(data) + 1)
            sx  = np.arange(w, len(data) + 1)
            ax.plot(eps, data, alpha=0.08, color=color, linewidth=0.5)
            ax.plot(sx, _smooth(data, w), color=color, linewidth=2.0,
                    linestyle=ls, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"F{col+1} -- {title}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


# ======================================================================
# Text report
# ======================================================================

def write_text_report(all_train: dict, all_test: dict, filename: str):
    """Full numerical summary of all ablation conditions."""
    lines = []
    lines.append("=" * 88)
    lines.append("  ABLATION & NEUROPATHOLOGY BENCHMARK v2 [3F-full] -- Doya (1999) + ACTrace on CartPole-v1")
    lines.append("=" * 88)
    lines.append("")

    train_cond_labels = {
        "intact":        "Intact (full model)",
        "no_cortex":     "Cortex ablation (NullCortex)",
        "no_inverse":    "Inverse model ablation (lr=0)",
        "park_mild":     "Parkinson stage 1 (delta x0.75)",
        "park_moderate": "Parkinson stage 2 (delta x0.40)",
        "park_severe":   "Parkinson stage 3 (delta x0.15)",
    }
    test_only_labels = {
        "no_fwd_test":   "Fwd ablation at test (pc_weights=0)",
        "ataxia_test":   "Cerebellar ataxia at test (noise sigma=0.5)",
        "park_late_test":"PD late onset (BG zeroed post-training)",
    }

    # Training conditions
    for cond, label in train_cond_labels.items():
        if cond not in all_train:
            continue
        data = all_train[cond]
        rewards = data["metrics"]["rewards"]
        elapsed = data.get("elapsed", 0)
        n = len(rewards)
        tail = max(1, n // 10)
        q = n // 4

        lines.append("-" * 88)
        lines.append(f"  CONDITION: {label}")
        lines.append(f"  Duration: {elapsed:.0f}s | Episodes: {n}")
        lines.append("-" * 88)
        lines.append("")
        lines.append("  Training quartiles:")
        for i in range(4):
            s, e = i * q, (i + 1) * q
            lines.append(
                f"    Ep {s+1:4d}-{e:4d}: {np.mean(rewards[s:e]):6.1f} +/- {np.std(rewards[s:e]):5.1f}"
                f"  (max: {np.max(rewards[s:e]):.0f})"
            )
        lines.append(f"  Final 10%: {np.mean(rewards[-tail:]):.1f} +/- {np.std(rewards[-tail:]):.1f}")
        lines.append("")

        if cond in all_test:
            test = all_test[cond]
            lines.append("  Test phase (50 trials):")
            for s, slabel in zip(STRATEGY_NAMES, ["BG reactive", "Predictive", "Inverse", "Hybrid"]):
                r = test[s]
                n500 = sum(1 for x in r if x >= 500)
                lines.append(
                    f"    {slabel:<20s}: {np.mean(r):6.1f} +/- {np.std(r):5.1f}"
                    f"  max: {np.max(r):.0f}  >=500: {n500}/50"
                )
        lines.append("")

    # Test-only conditions
    lines.append("=" * 88)
    lines.append("  TEST-TIME MODIFICATIONS (applied to intact trained model)")
    lines.append("=" * 88)
    for cond, label in test_only_labels.items():
        if cond not in all_test:
            continue
        test = all_test[cond]
        lines.append(f"\n  {label}")
        for s, slabel in zip(STRATEGY_NAMES, ["BG reactive", "Predictive", "Inverse", "Hybrid"]):
            r = test[s]
            n500 = sum(1 for x in r if x >= 500)
            lines.append(
                f"    {slabel:<20s}: {np.mean(r):6.1f} +/- {np.std(r):5.1f}"
                f"  max: {np.max(r):.0f}  >=500: {n500}/50"
            )

    lines.append("")
    lines.append("=" * 88)
    report = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n  [OK] {filename}")


# ======================================================================
# Main benchmark
# ======================================================================

# (condition_name, ablation_kwargs)
TRAINING_CONDITIONS = [
    ("intact",        dict(cortex_disabled=False, inverse_disabled=False, delta_scale=1.00)),
    ("no_cortex",     dict(cortex_disabled=True,  inverse_disabled=False, delta_scale=1.00)),
    ("no_inverse",    dict(cortex_disabled=False, inverse_disabled=True,  delta_scale=1.00)),
    ("park_mild",     dict(cortex_disabled=False, inverse_disabled=False, delta_scale=0.75)),
    ("park_moderate", dict(cortex_disabled=False, inverse_disabled=False, delta_scale=0.40)),
    ("park_severe",   dict(cortex_disabled=False, inverse_disabled=False, delta_scale=0.15)),
]


def run_benchmark_ablations(n_episodes: int = 800, seed: int = 42):
    """Run all training and test conditions, produce all figures."""
    output_dir = os.path.dirname(os.path.abspath(__file__))

    all_train_reactive   = {}   # condition -> {metrics, elapsed, ...}
    all_train_predictive = {}
    all_test             = {}   # condition -> {bg, predictive, inverse, hybrid}
    _intact              = None  # set by whichever phase runs "intact"

    # ── Phase 1a: reactive training conditions ──
    # for cond, ablation_kwargs in TRAINING_CONDITIONS:
    #     print()
    #     print("+" + "=" * 70 + "+")
    #     print(f"|  TRAINING [reactive]: {cond:<48s} |")
    #     print("+" + "=" * 70 + "+")
    #     t0 = time.time()
    #     cortex, bg, fwd, inv, rewards, lengths, metrics = train_ablation(
    #         n_episodes=n_episodes, seed=seed,
    #         training_mode="reactive", **ablation_kwargs
    #     )
    #     elapsed = time.time() - t0
    #     all_train_reactive[cond] = {"metrics": metrics, "elapsed": elapsed}
    #     # Store intact reactive model for test-time modifications
    #     if cond == "intact":
    #         _intact = (cortex, bg, fwd, inv)
    #     # Test the trained model
    #     test_results = run_test_ablation(cortex, bg, fwd, inv, n_trials=50, seed=999)
    #     all_test[cond] = test_results

    # ── Phase 1b: predictive training conditions ──
    for cond, ablation_kwargs in TRAINING_CONDITIONS:
        print()
        print("+" + "=" * 70 + "+")
        print(f"|  TRAINING [predictive]: {cond:<46s} |")
        print("+" + "=" * 70 + "+")
        t0 = time.time()
        cortex, bg, fwd, inv, rewards, lengths, metrics = train_ablation(
            n_episodes=n_episodes, seed=seed,
            training_mode="predictive", **ablation_kwargs
        )
        elapsed = time.time() - t0
        all_train_predictive[cond] = {"metrics": metrics, "elapsed": elapsed}
        # Store intact predictive model (used if Phase 1a was skipped)
        if cond == "intact" and _intact is None:
            _intact = (cortex, bg, fwd, inv)
        # Populate all_test if Phase 1a was skipped
        if cond not in all_test:
            test_results = run_test_ablation(cortex, bg, fwd, inv, n_trials=50, seed=999)
            all_test[cond] = test_results

    # ── Phase 2: test-time modifications on intact model ──
    cortex_intact, bg_intact, fwd_intact, inv_intact = _intact

    print()
    print("+" + "=" * 70 + "+")
    print("|  TEST-TIME: forward model ablation                                   |")
    print("+" + "=" * 70 + "+")
    all_test["no_fwd_test"] = run_test_ablation(
        cortex_intact, bg_intact, fwd_intact, inv_intact,
        n_trials=50, seed=999, fwd_zeroed=True,
    )

    print()
    print("+" + "=" * 70 + "+")
    print("|  TEST-TIME: cerebellar ataxia (sigma=0.5)                            |")
    print("+" + "=" * 70 + "+")
    all_test["ataxia_test"] = run_test_ablation(
        cortex_intact, bg_intact, fwd_intact, inv_intact,
        n_trials=50, seed=999, ataxia_sigma=0.5,
    )

    print()
    print("+" + "=" * 70 + "+")
    print("|  TEST-TIME: Parkinson late onset (BG zeroed post-training)           |")
    print("+" + "=" * 70 + "+")
    all_test["park_late_test"] = run_test_ablation(
        cortex_intact, bg_intact, fwd_intact, inv_intact,
        n_trials=50, seed=999, bg_zeroed=True,
    )

    # ── Phase 3: figures ──
    print()
    print("Generating figures...")

    plot_learning_curves(all_train_reactive,
        os.path.join(output_dir, "ablation_v2p_learning_curves.png"),
        mode_label="reactive")

    plot_training_modes(all_train_reactive, all_train_predictive,
        os.path.join(output_dir, "ablation_v2p_training_modes.png"))

    plot_parkinson_encapsulation(all_test,
        os.path.join(output_dir, "ablation_v2p_parkinson_encapsulation.png"))

    plot_ataxia_dissociation(all_test,
        os.path.join(output_dir, "ablation_v2p_ataxia_dissociation.png"))

    plot_error_traces(all_train_reactive,
        os.path.join(output_dir, "ablation_v2p_error_traces.png"),
        mode_label="reactive")

    write_text_report(all_train_reactive, all_test,
        os.path.join(output_dir, "ablation_v2p_metrics.txt"))

    return all_train_reactive, all_train_predictive, all_test


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation & Neuropathology Benchmark v2 [3F-full] -- Doya (1999) + ACTrace"
    )
    parser.add_argument("--episodes", "-n", type=int, default=800)
    parser.add_argument("--seed",     "-s", type=int, default=42)
    args = parser.parse_args()

    print()
    print("+" + "=" * 70 + "+")
    print("|  ABLATION BENCHMARK v2 [3F-full] -- Doya (1999) + ACTrace CartPole  |")
    print("+" + "=" * 70 + "+")
    print(f"|  Episodes: {args.episodes:<59d} |")
    print(f"|  Seed:     {args.seed:<59d} |")
    print("|  Conditions: 6 training x 2 modes + 3 test-time modifications       |")
    print("|  Modules: CerebralCortex3F + ForwardModel3F + InverseModel3F        |")
    print("|  BG state: obs (4D) + cortex_repr (6D) = 10D [no handcrafted feat.] |")
    print("+" + "=" * 70 + "+")

    run_benchmark_ablations(n_episodes=args.episodes, seed=args.seed)

    print()
    print("  Generated files:")
    for f in [
        "ablation_v2p_learning_curves.png    -- structural ablations + Parkinson (reactive)",
        "ablation_v2p_training_modes.png     -- reactive vs predictive training comparison",
        "ablation_v2p_parkinson_encapsulation.png -- encapsulation protection",
        "ablation_v2p_ataxia_dissociation.png    -- BG/cerebellum dissociation",
        "ablation_v2p_error_traces.png           -- module error traces",
        "ablation_v2p_metrics.txt                -- full numerical report",
    ]:
        print(f"    {f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
