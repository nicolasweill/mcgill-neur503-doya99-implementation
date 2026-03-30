"""
Training & Testing — Doya (1999) Architecture on Mario Bros / Atari

Applies the full Doya (1999, 2000) tri-module framework to pixel-based
game environments:

    - Cerebral cortex (unsupervised): CNN sparse autoencoder learns a 64D
      representation of 84x84x4 preprocessed frames (V1->V2->V4->IT).
    - Basal ganglia (reinforcement learning): MLP actor-critic with discrete
      actions, TD learning with dopamine-like error signal (Eq. 10-14).
    - Cerebellum (supervised learning):
        * Forward model predicts next cortical features (Eq. 21).
        * Inverse model encapsulates BG policy (Fig. 11).
        * Hybrid controller: surprise-based switching (v2).

Two architectures selectable at startup (Doya 1999 Section 4):
    [1] Section 4.1  — Reactive: BG selects actions from value function alone
    [2] Section 4.2.1 — Predictive: forward model evaluates candidates (Fig. 7)

Usage
-----
    # Training
    python train_mario.py                           # interactive menu
    python train_mario.py reactive                  # Section 4.1 directly
    python train_mario.py predictive -n 500         # 500 episodes
    python train_mario.py predictive --report       # generate PNG report

    # Inference (load saved model)
    python train_mario.py --load checkpoints/best   # run test phase
    python train_mario.py --load checkpoints/best --render          # visual playback (all strategies)
    python train_mario.py --load checkpoints/best --render --strategy bg  # specific strategy
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mario_wrappers import make_mario_env
from cerebral_cortex_mario import CerebralCortexMario
from basal_ganglia_mario import BasalGangliaMario
from cerebellum_mario import (
    ForwardModelMario,
    InverseModelMario,
    CerebellarCorrectorMario,
    HybridControllerMario,
)


# ======================================================================
# Checkpoint save / load
# ======================================================================

def save_checkpoint(
    path: str,
    cortex: CerebralCortexMario,
    basal_ganglia: BasalGangliaMario,
    forward_model: ForwardModelMario,
    inverse_model: InverseModelMario,
):
    """Save all four brain modules to a directory."""
    os.makedirs(path, exist_ok=True)
    cortex.save(os.path.join(path, "cortex.pt"))
    basal_ganglia.save(os.path.join(path, "basal_ganglia.pt"))
    forward_model.save(os.path.join(path, "forward_model.npz"))
    inverse_model.save(os.path.join(path, "inverse_model.npz"))
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    n_actions: int = 7,
    feature_dim: int = 64,
) -> tuple[CerebralCortexMario, BasalGangliaMario, ForwardModelMario, InverseModelMario]:
    """Load all four brain modules from a checkpoint directory."""
    cortex = CerebralCortexMario(in_channels=4, repr_dim=feature_dim)
    basal_ganglia = BasalGangliaMario(state_dim=feature_dim, n_actions=n_actions)
    forward_model = ForwardModelMario(feature_dim=feature_dim, n_actions=n_actions, n_granule=512)
    inverse_model = InverseModelMario(feature_dim=feature_dim, n_actions=n_actions, n_granule=512)

    cortex.load(os.path.join(path, "cortex.pt"))
    basal_ganglia.load(os.path.join(path, "basal_ganglia.pt"))
    forward_model.load(os.path.join(path, "forward_model.npz"))
    inverse_model.load(os.path.join(path, "inverse_model.npz"))

    print(f"  Checkpoint loaded: {path}")
    return cortex, basal_ganglia, forward_model, inverse_model


# ======================================================================
# Section 4.2.1 — Discrete model-based action selection (Fig. 7)
# ======================================================================

def _select_action_predictive(
    basal_ganglia: BasalGangliaMario,
    forward_model: ForwardModelMario,
    features: np.ndarray,
    explore: bool,
) -> int:
    """Discrete model-based action selection (Section 4.2.1, Fig. 7).

    Evaluates each candidate action using the predictive TD error:
        δ*(t+1) = γ V(x*(t+1)) − V(x(t))     (Eq. 22)

    The action maximizing δ* is selected.
    """
    current_value = basal_ganglia.value(features)
    gamma = basal_ganglia.gamma
    n_actions = forward_model.n_actions

    best_delta = -1e10
    best_action = 0

    for a in range(n_actions):
        pred_features = forward_model.predict_next_features(features, a)
        pred_value = basal_ganglia.value(pred_features)
        delta_star = gamma * pred_value - current_value

        if delta_star > best_delta:
            best_delta = delta_star
            best_action = a

    # When exploring, sometimes take BG's stochastic action instead
    if explore and np.random.random() < 0.3:
        best_action = basal_ganglia.policy(features, explore=True)

    return best_action


# ======================================================================
# Cortex warmup (collect frames with random actions)
# ======================================================================

def _warmup_cortex(cortex: CerebralCortexMario, n_frames: int = 5000):
    """Pre-train cortex autoencoder on random frames."""
    env, n_actions, _ = make_mario_env()
    frames = []

    print(f"  Collecting {n_frames} random frames for cortex warmup...")
    obs, _ = env.reset()
    for i in range(n_frames):
        action = np.random.randint(n_actions)
        obs, _, done, truncated, _ = env.step(action)
        frames.append(obs)
        if done or truncated:
            obs, _ = env.reset()

    env.close()
    cortex.warmup(frames, n_epochs=10)
    print()


# ======================================================================
# Training loop
# ======================================================================

def train(
    mode: str = "reactive",
    n_episodes: int = 2000,
    print_every: int = 50,
    seed: int = 42,
    warmup_frames: int = 5000,
    checkpoint_dir: str = "checkpoints",
):
    """Train the integrated model on Mario Bros / Atari.

    Parameters
    ----------
    mode : str
        "reactive" (Section 4.1) or "predictive" (Section 4.2.1).
    checkpoint_dir : str
        Directory for auto-saving best model.
    """
    np.random.seed(seed)

    env, n_actions, env_name = make_mario_env()
    feature_dim = 64

    # --- Initialize the three brain modules ---
    cortex = CerebralCortexMario(
        in_channels=4,
        repr_dim=feature_dim,
        lr=1e-4,
        sparsity_lambda=0.01,
    )

    basal_ganglia = BasalGangliaMario(
        state_dim=feature_dim,
        n_actions=n_actions,
        gamma=0.99,
        lr_critic=1e-4,
        lr_actor=1e-4,
        temperature=2.0,
    )

    forward_model = ForwardModelMario(
        feature_dim=feature_dim,
        n_actions=n_actions,
        n_granule=512,
        lr=0.005,
        grad_clip=2.0,
    )

    inverse_model = InverseModelMario(
        feature_dim=feature_dim,
        n_actions=n_actions,
        n_granule=512,
        lr=0.005,
        grad_clip=2.0,
    )

    mode_label = {
        "reactive": "Section 4.1 — Reactive (model-free, Fig. 6)",
        "predictive": "Section 4.2.1 — Discrete model-based (Fig. 7)",
    }

    print("=" * 72)
    print(f"  Doya (1999) — {mode_label[mode]}")
    print("=" * 72)
    print(f"  Environment:   {env_name}, {n_actions} actions")
    print(f"  Cortex:        CNN autoencoder, 84x84x4 -> {feature_dim}D")
    print(f"  Basal Ganglia: MLP actor-critic, {feature_dim}D -> {n_actions} actions")
    print(f"  Cerebellum:    forward ({feature_dim}+{n_actions}D -> {feature_dim}D)")
    print(f"                 inverse ({feature_dim}*2D -> {n_actions}D)")
    print(f"  Training:      {n_episodes} episodes")
    print(f"  Auto-save:     {checkpoint_dir}/best")
    print("=" * 72)
    print()

    # --- Cortex warmup (unsupervised pre-training) ---
    _warmup_cortex(cortex, n_frames=warmup_frames)

    # --- Tracking ---
    episode_rewards = []
    episode_lengths = []
    cortex_errors = []
    forward_errors = []
    inverse_errors = []
    td_magnitudes = []

    initial_temp = 2.0
    final_temp = 0.1
    best_reward = -float("inf")
    best_path = os.path.join(checkpoint_dir, "best")

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        basal_ganglia.reset()

        # Anneal temperature (Doya Sec. 6.5)
        progress = episode / n_episodes
        basal_ganglia.temperature = initial_temp * (1 - progress) + final_temp * progress

        total_reward = 0.0
        ep_cx, ep_fwd, ep_inv, ep_td = [], [], [], []
        step_count = 0

        # Cortex encode
        features, cx_err = cortex.encode_and_learn(obs)
        ep_cx.append(cx_err)

        # First action
        if mode == "reactive":
            action = basal_ganglia.step(features, explore=True)
        else:
            _ = basal_ganglia.step(features, explore=True)
            action = _select_action_predictive(
                basal_ganglia, forward_model, features, explore=True
            )

        done = False
        truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Cortex
            next_features, cx_err = cortex.encode_and_learn(obs)
            ep_cx.append(cx_err)

            # BG learning
            terminal = done or truncated
            delta = basal_ganglia.learn(reward, next_features, terminal)
            ep_td.append(abs(delta))

            # Cerebellum: forward model
            fwd_err = forward_model.learn_transition(features, action, next_features)
            ep_fwd.append(fwd_err)

            # Cerebellum: inverse model encapsulation
            inv_err = inverse_model.learn_action(features, next_features, action)
            ep_inv.append(inv_err)

            if not terminal:
                features = next_features
                if mode == "reactive":
                    action = basal_ganglia.step(features, explore=True)
                else:
                    _ = basal_ganglia.step(features, explore=True)
                    action = _select_action_predictive(
                        basal_ganglia, forward_model, features, explore=True
                    )

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        cortex_errors.append(np.mean(ep_cx) if ep_cx else 0)
        forward_errors.append(np.mean(ep_fwd) if ep_fwd else 0)
        inverse_errors.append(np.mean(ep_inv) if ep_inv else 0)
        td_magnitudes.append(np.mean(ep_td) if ep_td else 0)

        # --- Auto-save best model ---
        # Use rolling average over last 10 episodes to reduce noise
        window = min(10, len(episode_rewards))
        avg_recent = np.mean(episode_rewards[-window:])
        if episode >= 10 and avg_recent > best_reward:
            best_reward = avg_recent
            save_checkpoint(best_path, cortex, basal_ganglia, forward_model, inverse_model)

        if episode % print_every == 0 or episode == 1:
            w = min(print_every, episode)
            avg_r = np.mean(episode_rewards[-w:])
            avg_len = np.mean(episode_lengths[-w:])
            avg_cx = np.mean(cortex_errors[-w:])
            avg_fwd = np.mean(forward_errors[-w:])
            avg_inv = np.mean(inverse_errors[-w:])
            avg_td = np.mean(td_magnitudes[-w:])
            best_mark = " *" if avg_recent >= best_reward else ""
            print(
                f"Ep {episode:4d} | "
                f"R {avg_r:7.1f} | "
                f"Len {avg_len:5.0f} | "
                f"DA {avg_td:.3f} | "
                f"Cx {avg_cx:.4f} | "
                f"Fwd {avg_fwd:.4f} | "
                f"Inv {avg_inv:.4f}{best_mark}"
            )

    env.close()

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final")
    save_checkpoint(final_path, cortex, basal_ganglia, forward_model, inverse_model)

    metrics = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "cortex_errors": cortex_errors,
        "forward_errors": forward_errors,
        "inverse_errors": inverse_errors,
        "td_magnitudes": td_magnitudes,
    }

    return cortex, basal_ganglia, forward_model, inverse_model, metrics


# ======================================================================
# Testing
# ======================================================================

def run_test(
    cortex: CerebralCortexMario,
    basal_ganglia: BasalGangliaMario,
    forward_model: ForwardModelMario,
    inverse_model: InverseModelMario,
    n_trials: int = 20,
    seed: int = 999,
) -> dict:
    """Compare control strategies (no rendering)."""
    np.random.seed(seed)
    env, n_actions, env_name = make_mario_env()

    corrector = CerebellarCorrectorMario(forward_model)
    hybrid = HybridControllerMario(
        forward_model, inverse_model, corrector,
        surprise_threshold_factor=2.0,
    )

    print()
    print("=" * 72)
    print(f"  TEST PHASE: {env_name} — Comparing Control Strategies")
    print("=" * 72)

    results = {"bg": [], "predictive": [], "inverse": [], "hybrid": []}

    for trial in range(n_trials):
        trial_seed = seed + trial

        # --- BG policy (4.1) ---
        obs, _ = env.reset(seed=trial_seed)
        basal_ganglia.reset()
        features = cortex.encode(obs)
        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = basal_ganglia.policy(features, explore=False)
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            features = cortex.encode(obs)
        results["bg"].append(total_r)

        # --- Predictive (4.2.1) ---
        obs, _ = env.reset(seed=trial_seed)
        basal_ganglia.reset()
        features = cortex.encode(obs)
        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = _select_action_predictive(
                basal_ganglia, forward_model, features, explore=False
            )
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            features = cortex.encode(obs)
        results["predictive"].append(total_r)

        # --- Inverse model (Fig. 11) ---
        obs, _ = env.reset(seed=trial_seed)
        features = cortex.encode(obs)
        target_features = np.zeros(cortex.repr_dim)
        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = inverse_model.compute_action(features, target_features)
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            features = cortex.encode(obs)
        results["inverse"].append(total_r)

        # --- Hybrid (v2) ---
        obs, _ = env.reset(seed=trial_seed)
        hybrid.reset()
        features = cortex.encode(obs)
        target_features = np.zeros(cortex.repr_dim)
        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, source, _ = hybrid.select_action(
                features, target_features, basal_ganglia,
                observed_features=features,
            )
            corrector.begin_step(features, action)
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            features = cortex.encode(obs)
        results["hybrid"].append(total_r)

    env.close()

    # Print
    print()
    print(f"  {'Strategy':<45s} | {'Reward':>12s} | {'Max':>6s}")
    print("  " + "-" * 70)
    for name, label in [
        ("bg", "Basal Ganglia (4.1, Fig. 6)"),
        ("predictive", "Discrete model-based (4.2.1, Fig. 7)"),
        ("inverse", "Cerebellum inverse model (Fig. 11)"),
        ("hybrid", "Hybrid — correction + switch (v2)"),
    ]:
        r = results[name]
        print(
            f"  {label:<45s} | "
            f"{np.mean(r):5.1f}+/-{np.std(r):4.1f} | "
            f"{np.max(r):6.1f}"
        )

    print()
    print(f"  Hybrid: {hybrid.cerebellum_ratio*100:.1f}% cerebellar, "
          f"{(1 - hybrid.cerebellum_ratio)*100:.1f}% BG fallback")
    print()

    return results


# ======================================================================
# Visual simulation
# ======================================================================

def simulate(
    cortex: CerebralCortexMario,
    basal_ganglia: BasalGangliaMario,
    forward_model: ForwardModelMario,
    inverse_model: InverseModelMario,
    strategy: str = "bg",
    n_episodes: int = 2,
    seed: int = 777,
):
    """Run with rendering to visualize a control strategy."""
    env, n_actions, env_name = make_mario_env(render_mode="human")

    corrector = CerebellarCorrectorMario(forward_model)
    hybrid = HybridControllerMario(
        forward_model, inverse_model, corrector,
        surprise_threshold_factor=2.0,
    )

    labels = {
        "bg": "Basal Ganglia (4.1)",
        "predictive": "Predictive (4.2.1)",
        "inverse": "Inverse model (Fig. 11)",
        "hybrid": "Hybrid (v2)",
    }
    print(f"  Simulating: {labels[strategy]}...")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        basal_ganglia.reset()
        hybrid.reset()
        features = cortex.encode(obs)
        target_features = np.zeros(cortex.repr_dim)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            if strategy == "bg":
                action = basal_ganglia.policy(features, explore=False)
            elif strategy == "predictive":
                action = _select_action_predictive(
                    basal_ganglia, forward_model, features, explore=False
                )
            elif strategy == "inverse":
                action = inverse_model.compute_action(features, target_features)
            elif strategy == "hybrid":
                action, _, _ = hybrid.select_action(
                    features, target_features, basal_ganglia,
                    observed_features=features,
                )
                corrector.begin_step(features, action)

            obs, r, done, truncated, _ = env.step(action)
            try:
                env.render()
            except Exception:
                pass
            total_r += r
            features = cortex.encode(obs)

        print(f"    Episode {ep}: reward = {total_r:.1f}")

    env.close()
    print()


# ======================================================================
# Training report (PNG)
# ======================================================================

def _smooth(data, window=30):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_report(
    metrics: dict,
    mode: str,
    test_results: dict | None = None,
    filename: str = "mario_training_report.png",
):
    """Multi-panel PNG training report."""
    mode_label = {
        "reactive": "Section 4.1 — Reactive (Fig. 6)",
        "predictive": "Section 4.2.1 — Predictive (Fig. 7)",
    }

    has_test = test_results is not None and len(test_results) > 0
    n_panels = 6 if has_test else 5

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.2 * n_panels))
    fig.suptitle(
        f"Doya (1999) — Game Training Report\n{mode_label.get(mode, mode)}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    episodes = np.arange(1, len(metrics["rewards"]) + 1)
    w = 30
    smooth_x = np.arange(w, len(metrics["rewards"]) + 1)

    # Panel 1: Reward
    ax = axes[0]
    ax.plot(episodes, metrics["rewards"], alpha=0.2, color="C0", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["rewards"], w), color="C0", linewidth=2,
            label=f"Avg (w={w})")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: Episode length
    ax = axes[1]
    ax.plot(episodes, metrics["lengths"], alpha=0.2, color="C1", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["lengths"], w), color="C1", linewidth=2)
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    # Panel 3: TD magnitude
    ax = axes[2]
    ax.plot(episodes, metrics["td_magnitudes"], alpha=0.2, color="C2", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["td_magnitudes"], w), color="C2", linewidth=2)
    ax.set_ylabel("|δ| (DA)")
    ax.set_title("TD Error — Dopamine Signal (Eq. 10)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Cortex error
    ax = axes[3]
    ax.plot(episodes, metrics["cortex_errors"], alpha=0.2, color="C3", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["cortex_errors"], w), color="C3", linewidth=2)
    ax.set_ylabel("Loss")
    ax.set_title("Cortex Autoencoder Loss — Unsupervised (Eq. 17)")
    ax.grid(True, alpha=0.3)

    # Panel 5: Cerebellar errors
    ax = axes[4]
    ax.plot(episodes, metrics["forward_errors"], alpha=0.2, color="C4", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["forward_errors"], w), color="C4", linewidth=2,
            label="Forward (Eq. 21)")
    ax.plot(episodes, metrics["inverse_errors"], alpha=0.2, color="C5", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["inverse_errors"], w), color="C5", linewidth=2,
            label="Inverse (Fig. 11)")
    ax.set_ylabel("MSE")
    ax.set_title("Cerebellar Model Errors — Supervised (Eq. 5)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Episode")

    # Panel 6: Test comparison
    if has_test:
        ax = axes[5]
        names = list(test_results.keys())
        means = [np.mean(v) for v in test_results.values()]
        stds = [np.std(v) for v in test_results.values()]
        colors = ["C0", "C3", "C4", "C5"][:len(names)]
        labels_map = {
            "bg": "BG\n(4.1)",
            "predictive": "Predictive\n(4.2.1)",
            "inverse": "Inverse\n(Fig.11)",
            "hybrid": "Hybrid\n(v2)",
        }
        x_labels = [labels_map.get(n, n) for n in names]
        bars = ax.bar(x_labels, means, yerr=stds, color=colors, alpha=0.8,
                      capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Reward")
        ax.set_title("Test — Strategy Comparison")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training report saved: {filename}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Doya (1999) architecture on Mario Bros / Atari"
    )
    parser.add_argument(
        "mode", nargs="?", choices=["reactive", "predictive"], default=None,
        help="Training architecture: reactive (4.1) or predictive (4.2.1)",
    )
    parser.add_argument("--episodes", "-n", type=int, default=2000,
                        help="Number of training episodes")
    parser.add_argument("--warmup", type=int, default=5000,
                        help="Number of random frames for cortex warmup")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for saving model checkpoints")

    # Inference / playback
    parser.add_argument("--load", type=str, default=None, metavar="PATH",
                        help="Load a saved checkpoint (skip training)")
    parser.add_argument("--render", action="store_true",
                        help="Visual playback of model on the game")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["bg", "predictive", "inverse", "hybrid"],
                        help="Strategy for --render (default: all)")

    # Report
    parser.add_argument("--report", action="store_true",
                        help="Generate PNG training report")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip visual simulation after training")

    args = parser.parse_args()

    # === Inference mode: load checkpoint ===
    if args.load is not None:
        # Detect n_actions from environment
        env, n_actions, env_name = make_mario_env()
        env.close()

        cortex, bg, fwd, inv = load_checkpoint(args.load, n_actions=n_actions)

        if args.render:
            # Visual playback
            strategies = [args.strategy] if args.strategy else ["bg", "predictive", "inverse", "hybrid"]
            for strat in strategies:
                try:
                    simulate(cortex, bg, fwd, inv, strategy=strat, n_episodes=3)
                except KeyboardInterrupt:
                    print("\n  Skipped.")
                    break
        else:
            # Run test phase
            run_test(cortex, bg, fwd, inv)

        return 0

    # === Training mode ===
    if args.mode is None:
        print()
        print("=" * 56)
        print("  Doya (1999) — Mario Bros / Atari")
        print("=" * 56)
        print()
        print("  [1] Section 4.1  — Reactive (Fig. 6)")
        print("  [2] Section 4.2.1 — Predictive model-based (Fig. 7)")
        print()
        while True:
            choice = input("  Enter 1 or 2: ").strip()
            if choice == "1":
                mode = "reactive"
                break
            elif choice == "2":
                mode = "predictive"
                break
            print("  Please enter 1 or 2.")
    else:
        mode = args.mode

    print()

    cortex, bg, fwd, inv, metrics = train(
        mode=mode,
        n_episodes=args.episodes,
        warmup_frames=args.warmup,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Learning curve summary
    print()
    print("-" * 72)
    print("  Learning curve summary:")
    n = len(metrics["rewards"])
    bs = n // 4
    for i in range(4):
        s, e = i * bs, (i + 1) * bs
        avg_r = np.mean(metrics["rewards"][s:e])
        avg_l = np.mean(metrics["lengths"][s:e])
        print(f"    Episodes {s+1:4d}-{e:4d}: reward = {avg_r:7.1f}, length = {avg_l:5.0f}")
    print("-" * 72)

    # Test
    test_results = run_test(cortex, bg, fwd, inv)

    # Report (optional, with --report flag)
    if args.report:
        plot_training_report(metrics, mode, test_results=test_results)

    # Visual simulation (unless --no-render)
    if not args.no_render:
        print("  Visual simulation — press Ctrl+C to skip")
        for strat in ["bg", "predictive", "inverse", "hybrid"]:
            try:
                simulate(cortex, bg, fwd, inv, strategy=strat, n_episodes=1)
            except KeyboardInterrupt:
                print("\n  Skipped.")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
