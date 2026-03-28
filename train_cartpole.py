"""
Training & Testing — Doya (1999) Architectures on CartPole-v1

Implements two global architectures from Doya (1999) Section 4:

  1. Section 4.1 — Reactive action selection (Fig. 6):
     Model-free actor-critic. The basal ganglia (striosome = critic,
     matrix = actor) learn from the dopaminergic TD error alone.
     No forward model is used during action selection.

  2. Section 4.2.1 — Discrete model-based action selection (Fig. 7):
     The cerebellar forward model predicts the outcome of each
     candidate action. The predictive TD error (Eq. 22) is used
     to select the best action:

         δ*(t+1) = R(x,u) + γ V(x*(t+1)) − V(x(t))

     This architecture links prefrontal/premotor cortex (candidate
     generation), lateral cerebellum (forward model), and the
     anterior basal ganglia (predictive TD evaluation).

Both architectures share:
  - Cerebral cortex: unsupervised state representation (Eq. 17-18)
  - Basal ganglia: actor-critic TD learning (Eq. 10-14)
  - Cerebellum: forward model (Eq. 21) + inverse model (Fig. 11)
  - v2 features: online correction + hybrid switching

After training, a visual simulation compares control strategies.

Usage
-----
    python train_cartpole.py              # interactive menu
    python train_cartpole.py reactive     # Section 4.1 directly
    python train_cartpole.py predictive   # Section 4.2.1 directly
"""

import sys
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

from cerebral_cortex import CerebralCortex
from basal_ganglia import BasalGanglia
from cerebellum import (
    ForwardModel,
    InverseModel,
    CerebellarCorrector,
    HybridController,
)


# ======================================================================
# State augmentation
# ======================================================================

def _augment_state(obs: np.ndarray, cortex_repr: np.ndarray) -> np.ndarray:
    """Augment CartPole observation with cortical features.

    CartPole obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    Derived features help the linear actor-critic:
      - abs(angle): magnitude of deviation
      - angle * angular_vel: falling indicator
      - cart_pos sign: side of center
      - vel alignment with tilt
    """
    cart_pos, cart_vel, angle, ang_vel = obs[0], obs[1], obs[2], obs[3]

    features = np.concatenate([
        obs,                          # 4: raw observation
        [abs(angle)],                 # 1: deviation magnitude
        [angle * ang_vel],            # 1: falling indicator
        [np.sign(cart_pos)],          # 1: side of center
        [cart_vel * np.sign(angle)],  # 1: velocity alignment with tilt
        cortex_repr,                  # repr_dim: learned features
    ])
    return features


def _action_for_cerebellum(discrete_action: int) -> np.ndarray:
    """Encode CartPole discrete action as continuous scalar for cerebellum."""
    return np.array([1.0 if discrete_action == 1 else -1.0])


# ======================================================================
# Section 4.1 — Reactive action selection (Fig. 6)
# ======================================================================

def _select_action_reactive(
    basal_ganglia: BasalGanglia,
    bg_state: np.ndarray,
    explore: bool,
) -> tuple[np.ndarray, int]:
    """Model-free stochastic action selection (Section 4.1, Fig. 6).

    The actor (matrix compartment) outputs a continuous action
    based solely on the current state representation. No forward
    model is consulted — this is the simplest architecture.
    """
    continuous = basal_ganglia.policy(bg_state, explore=explore)
    discrete = 1 if continuous[0] > 0 else 0
    return continuous, discrete


# ======================================================================
# Section 4.2.1 — Discrete model-based action selection (Fig. 7)
# ======================================================================

def _select_action_predictive(
    basal_ganglia: BasalGanglia,
    forward_model: ForwardModel,
    cortex: CerebralCortex,
    raw_state: np.ndarray,
    bg_state: np.ndarray,
    explore: bool,
    n_candidates: int = 5,
) -> tuple[np.ndarray, int]:
    """Discrete model-based action selection (Section 4.2.1, Fig. 7).

    For each candidate action u*(t), the forward model predicts
    the resulting state x*(t+1), and the predictive TD error

        δ*(t+1) = R(x,u) + γ V(x*(t+1)) − V(x(t))     (Eq. 22)

    is computed. The action with the highest δ* is selected.

    In CartPole the action space is binary (left/right), so we
    evaluate both discrete actions. When explore=True, we also
    perturb the continuous BG output to generate additional
    candidates, implementing the serial evaluation described in
    Section 4.2.1: "consider a candidate action u*(t) one at a
    time, predict the resulting future state x*(t+1) and its
    value V(x*(t+1)), and accept it for execution if it is good
    enough."
    """
    current_value = basal_ganglia.value(bg_state)
    gamma = basal_ganglia.gamma

    best_delta = -1e10
    best_continuous = None
    best_discrete = 0

    # Evaluate both discrete actions explicitly
    for d_action in [0, 1]:
        action_enc = _action_for_cerebellum(d_action)
        pred_next_raw = forward_model.predict_next_state(raw_state, action_enc)
        pred_cortex = cortex.encode(pred_next_raw)
        pred_bg_state = _augment_state(pred_next_raw, pred_cortex)

        # Predictive TD error (Eq. 22)
        pred_value = basal_ganglia.value(pred_bg_state)
        delta_star = gamma * pred_value - current_value

        if delta_star > best_delta:
            best_delta = delta_star
            best_discrete = d_action
            best_continuous = action_enc.copy()

    # When exploring, also try random perturbations of BG output
    if explore:
        bg_action = basal_ganglia.policy(bg_state, explore=True)
        for _ in range(n_candidates - 2):
            candidate = bg_action + np.random.randn(1) * basal_ganglia.noise_std
            d_cand = 1 if candidate[0] > 0 else 0
            action_enc = _action_for_cerebellum(d_cand)
            pred_next_raw = forward_model.predict_next_state(raw_state, action_enc)
            pred_cortex = cortex.encode(pred_next_raw)
            pred_bg_state = _augment_state(pred_next_raw, pred_cortex)

            pred_value = basal_ganglia.value(pred_bg_state)
            delta_star = gamma * pred_value - current_value

            if delta_star > best_delta:
                best_delta = delta_star
                best_discrete = d_cand
                best_continuous = action_enc.copy()

    # The continuous action stored for BG learning is the selected encoding
    if best_continuous is None:
        best_continuous = _action_for_cerebellum(best_discrete)

    return best_continuous, best_discrete


# ======================================================================
# Training loop (shared by both architectures)
# ======================================================================

def train(
    mode: str = "reactive",
    n_episodes: int = 1500,
    print_every: int = 100,
    seed: int = 42,
):
    """Train the integrated model on CartPole-v1.

    Parameters
    ----------
    mode : str
        "reactive" for Section 4.1 (model-free), or
        "predictive" for Section 4.2.1 (discrete model-based).
    """
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = 4
    repr_dim = 6
    action_dim = 1

    augmented_dim = 8 + repr_dim

    # --- Initialize the three brain modules ---
    cortex = CerebralCortex(
        input_dim=obs_dim,
        repr_dim=repr_dim,
        lr=0.005,
        sparsity=0.02,
        relaxation_steps=10,
        relaxation_dt=0.03,
    )

    basal_ganglia = BasalGanglia(
        state_dim=augmented_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr_critic=0.02,
        lr_actor=0.01,
        noise_std=0.5,
    )

    forward_model = ForwardModel(
        state_dim=obs_dim,
        action_dim=action_dim,
        n_granule=256,
        lr=0.01,
        grad_clip=2.0,
    )

    inverse_model = InverseModel(
        state_dim=obs_dim,
        action_dim=action_dim,
        n_granule=256,
        lr=0.01,
        grad_clip=2.0,
    )

    # --- Tracking ---
    episode_rewards = []
    episode_lengths = []
    cortex_errors = []
    forward_errors = []
    inverse_errors = []
    td_magnitudes = []

    mode_label = {
        "reactive": "Section 4.1 — Reactive (model-free, Fig. 6)",
        "predictive": "Section 4.2.1 — Discrete model-based (Fig. 7)",
    }

    print("=" * 72)
    print(f"  Doya (1999) — {mode_label[mode]}")
    print("=" * 72)
    print(f"  Cortex:        unsupervised, {obs_dim}D -> {repr_dim}D repr")
    print(f"  Basal Ganglia: actor-critic RL, {augmented_dim}D -> {action_dim}D")
    print(f"  Cerebellum:    forward model ({obs_dim}+{action_dim}D -> {obs_dim}D)")
    print(f"                 inverse model ({obs_dim}*2D -> {action_dim}D)")
    print(f"  Environment:   CartPole-v1, {n_episodes} episodes")
    print("=" * 72)
    print()

    initial_noise = 0.5
    final_noise = 0.05

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        raw_state = np.array(obs, dtype=np.float64)
        basal_ganglia.reset()

        progress = episode / n_episodes
        basal_ganglia.noise_std = (
            initial_noise * (1 - progress) + final_noise * progress
        )

        total_reward = 0.0
        ep_cx, ep_fwd, ep_inv, ep_td = [], [], [], []

        cortex_repr, cx_err = cortex.encode_and_learn(raw_state)
        ep_cx.append(cx_err)
        bg_state = _augment_state(raw_state, cortex_repr)

        # First action selection depends on architecture
        if mode == "reactive":
            # Section 4.1: BG step stores state/value for TD learning
            continuous_action = basal_ganglia.step(bg_state, explore=True)
            discrete_action = 1 if continuous_action[0] > 0 else 0
        else:
            # Section 4.2.1: predictive selection, but still need BG step
            # for TD learning bookkeeping
            _ = basal_ganglia.step(bg_state, explore=True)
            continuous_action, discrete_action = _select_action_predictive(
                basal_ganglia, forward_model, cortex,
                raw_state, bg_state, explore=True,
            )

        done = False
        truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, _ = env.step(discrete_action)
            raw_next = np.array(obs, dtype=np.float64)
            total_reward += reward

            next_cortex, cx_err = cortex.encode_and_learn(raw_next)
            ep_cx.append(cx_err)
            next_bg_state = _augment_state(raw_next, next_cortex)

            # BG learning (shared: both architectures update TD)
            terminal = done or truncated
            bg_reward = reward if not done else -10.0
            delta = basal_ganglia.learn(bg_reward, next_bg_state, terminal)
            ep_td.append(abs(delta))

            # Cerebellum: forward model learns state transition
            action_enc = _action_for_cerebellum(discrete_action)
            fwd_err = forward_model.learn_transition(
                raw_state, action_enc, raw_next
            )
            ep_fwd.append(fwd_err)

            # Cerebellum: inverse model encapsulation
            target_state = np.zeros(obs_dim)
            inv_err = inverse_model.learn_action(
                raw_state, target_state, continuous_action
            )
            ep_inv.append(inv_err)

            if not terminal:
                raw_state = raw_next
                bg_state = next_bg_state

                if mode == "reactive":
                    continuous_action = basal_ganglia.step(bg_state, explore=True)
                    discrete_action = 1 if continuous_action[0] > 0 else 0
                else:
                    _ = basal_ganglia.step(bg_state, explore=True)
                    continuous_action, discrete_action = _select_action_predictive(
                        basal_ganglia, forward_model, cortex,
                        raw_state, bg_state, explore=True,
                    )

        episode_rewards.append(total_reward)
        episode_lengths.append(total_reward)
        cortex_errors.append(np.mean(ep_cx))
        forward_errors.append(np.mean(ep_fwd) if ep_fwd else 0)
        inverse_errors.append(np.mean(ep_inv) if ep_inv else 0)
        td_magnitudes.append(np.mean(ep_td) if ep_td else 0)

        if episode % print_every == 0 or episode == 1:
            w = min(print_every, episode)
            avg_r = np.mean(episode_rewards[-w:])
            avg_cx = np.mean(cortex_errors[-w:])
            avg_fwd = np.mean(forward_errors[-w:])
            avg_inv = np.mean(inverse_errors[-w:])
            avg_td = np.mean(td_magnitudes[-w:])
            print(
                f"Ep {episode:4d} | "
                f"R {avg_r:7.1f} | "
                f"DA {avg_td:.3f} | "
                f"Cx {avg_cx:.3f} | "
                f"Fwd {avg_fwd:.4f} | "
                f"Inv {avg_inv:.4f}"
            )

    env.close()

    metrics = {
        "rewards": episode_rewards,
        "cortex_errors": cortex_errors,
        "forward_errors": forward_errors,
        "inverse_errors": inverse_errors,
        "td_magnitudes": td_magnitudes,
    }

    return (
        cortex, basal_ganglia, forward_model, inverse_model,
        episode_rewards, episode_lengths, metrics,
    )


# ======================================================================
# Visual simulation
# ======================================================================

def simulate(
    cortex: CerebralCortex,
    basal_ganglia: BasalGanglia,
    forward_model: ForwardModel,
    inverse_model: InverseModel,
    strategy: str = "bg",
    n_episodes: int = 3,
    seed: int = 777,
):
    """Run CartPole with rendering to visualize a control strategy.

    Parameters
    ----------
    strategy : str
        "bg"       — Basal Ganglia policy (Section 4.1)
        "predictive" — Discrete model-based selection (Section 4.2.1)
        "inverse"  — Cerebellar inverse model (encapsulation, Fig. 11)
        "hybrid"   — v2: correction + auto-switch
    """
    env = gym.make("CartPole-v1", render_mode="human")

    corrector = CerebellarCorrector(forward_model, correction_gain=0.3)
    hybrid = HybridController(
        forward_model, inverse_model, corrector,
        surprise_threshold_factor=2.0,
    )

    strategy_labels = {
        "bg": "Basal Ganglia (Section 4.1, Fig. 6)",
        "predictive": "Discrete model-based (Section 4.2.1, Fig. 7)",
        "inverse": "Cerebellar inverse model (Fig. 11)",
        "hybrid": "Hybrid — correction + auto-switch (v2)",
    }

    print()
    print(f"  Simulating: {strategy_labels[strategy]}")
    print(f"  {n_episodes} episodes with rendering...")
    print()

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        raw_state = np.array(obs, dtype=np.float64)
        basal_ganglia.reset()
        hybrid.reset()

        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            if strategy == "bg":
                action = basal_ganglia.policy(bg_state, explore=False)
                discrete = 1 if action[0] > 0 else 0

            elif strategy == "predictive":
                _, discrete = _select_action_predictive(
                    basal_ganglia, forward_model, cortex,
                    raw_state, bg_state, explore=False,
                )

            elif strategy == "inverse":
                target = np.zeros(4)
                action = inverse_model.compute_action(raw_state, target)
                discrete = 1 if action[0] > 0 else 0

            elif strategy == "hybrid":
                target_raw = np.zeros(4)
                action, source, _ = hybrid.select_action(
                    raw_state, bg_state, target_raw, basal_ganglia,
                    observed_state=raw_state,
                )
                corrector.begin_step(raw_state, action)
                discrete = 1 if action[0] > 0 else 0

            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)

        print(f"    Episode {ep}: {int(total_r)} steps")

    env.close()
    print()


# ======================================================================
# Testing (non-visual, statistical)
# ======================================================================

def run_test(
    cortex: CerebralCortex,
    basal_ganglia: BasalGanglia,
    forward_model: ForwardModel,
    inverse_model: InverseModel,
    n_trials: int = 50,
    seed: int = 999,
):
    """Compare control strategies on CartPole (no rendering). Returns results dict."""
    np.random.seed(seed)
    env = gym.make("CartPole-v1")

    corrector = CerebellarCorrector(forward_model, correction_gain=0.3)
    hybrid = HybridController(
        forward_model, inverse_model, corrector,
        surprise_threshold_factor=2.0,
    )

    print()
    print("=" * 72)
    print("  TEST PHASE: CartPole — Comparing Control Strategies")
    print("=" * 72)

    results = {"bg": [], "predictive": [], "inverse": [], "hybrid": []}

    for trial in range(n_trials):
        trial_seed = seed + trial

        # --- Strategy 1: BG policy (Section 4.1) ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        basal_ganglia.reset()
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = basal_ganglia.policy(bg_state, explore=False)
            discrete = 1 if action[0] > 0 else 0
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)
        results["bg"].append(total_r)

        # --- Strategy 2: Predictive (Section 4.2.1) ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        basal_ganglia.reset()
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            _, discrete = _select_action_predictive(
                basal_ganglia, forward_model, cortex,
                raw_state, bg_state, explore=False,
            )
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)
        results["predictive"].append(total_r)

        # --- Strategy 3: Inverse model ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        target_state = np.zeros(4)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action = inverse_model.compute_action(raw_state, target_state)
            discrete = 1 if action[0] > 0 else 0
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
        results["inverse"].append(total_r)

        # --- Strategy 4: Hybrid (v2) ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        hybrid.reset()
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            target_raw = np.zeros(4)
            action, source, _ = hybrid.select_action(
                raw_state, bg_state, target_raw, basal_ganglia,
                observed_state=raw_state,
            )
            corrector.begin_step(raw_state, action)
            discrete = 1 if action[0] > 0 else 0
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)
        results["hybrid"].append(total_r)

    env.close()

    # Print results
    print()
    print(f"  {'Strategy':<45s} | {'Reward (steps)':>14s} | {'Max':>5s}")
    print("  " + "-" * 72)

    for name, label in [
        ("bg", "Basal Ganglia (4.1, Fig. 6)"),
        ("predictive", "Discrete model-based (4.2.1, Fig. 7)"),
        ("inverse", "Cerebellum inverse model (Fig. 11)"),
        ("hybrid", "Hybrid — correction + switch (v2)"),
    ]:
        r = results[name]
        print(
            f"  {label:<45s} | "
            f"{np.mean(r):6.1f} +/- {np.std(r):5.1f} | "
            f"{np.max(r):5.0f}"
        )

    print()
    print(f"  Hybrid: {hybrid.cerebellum_ratio*100:.1f}% cerebellar, "
          f"{(1 - hybrid.cerebellum_ratio)*100:.1f}% BG fallback")
    print()

    return results


# ======================================================================
# Training report (PNG)
# ======================================================================

def _smooth(data, window=50):
    """Simple moving average for plotting."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training_report(
    metrics: dict,
    mode: str,
    test_results: dict | None = None,
    filename: str = "cartpole_training_report.png",
):
    """Generate a multi-panel PNG summarizing training and test results.

    Panels:
      1. Episode reward (steps survived) with smoothed curve
      2. TD error magnitude (dopamine signal)
      3. Cortex reconstruction error
      4. Cerebellar forward & inverse model errors
      5. Test comparison bar chart (if test_results provided)
    """
    mode_label = {
        "reactive": "Section 4.1 — Reactive (Fig. 6)",
        "predictive": "Section 4.2.1 — Discrete model-based (Fig. 7)",
    }

    has_test = test_results is not None and len(test_results) > 0
    n_panels = 5 if has_test else 4

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.2 * n_panels))
    fig.suptitle(
        f"Doya (1999) — CartPole-v1 Training Report\n{mode_label.get(mode, mode)}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    episodes = np.arange(1, len(metrics["rewards"]) + 1)
    window = 50
    smooth_x = np.arange(window, len(metrics["rewards"]) + 1)

    # --- Panel 1: Episode reward ---
    ax = axes[0]
    ax.plot(episodes, metrics["rewards"], alpha=0.25, color="C0", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["rewards"], window), color="C0", linewidth=2,
            label=f"Avg (window={window})")
    ax.set_ylabel("Steps survived")
    ax.set_title("Episode Reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: TD magnitude (dopamine) ---
    ax = axes[1]
    ax.plot(episodes, metrics["td_magnitudes"], alpha=0.25, color="C1", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["td_magnitudes"], window), color="C1", linewidth=2)
    ax.set_ylabel("|δ| (DA signal)")
    ax.set_title("TD Error Magnitude — Dopamine Signal (Eq. 10)")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Cortex reconstruction error ---
    ax = axes[2]
    ax.plot(episodes, metrics["cortex_errors"], alpha=0.25, color="C2", linewidth=0.5)
    ax.plot(smooth_x, _smooth(metrics["cortex_errors"], window), color="C2", linewidth=2)
    ax.set_ylabel("||x - W'y||²")
    ax.set_title("Cortex Reconstruction Error — Unsupervised Learning (Eq. 17-18)")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Cerebellar errors ---
    ax = axes[3]
    ax.plot(episodes, metrics["forward_errors"], alpha=0.25, color="C3", linewidth=0.5)
    fwd_smooth = _smooth(metrics["forward_errors"], window)
    ax.plot(smooth_x, fwd_smooth, color="C3", linewidth=2, label="Forward model (Eq. 21)")
    ax.plot(episodes, metrics["inverse_errors"], alpha=0.25, color="C4", linewidth=0.5)
    inv_smooth = _smooth(metrics["inverse_errors"], window)
    ax.plot(smooth_x, inv_smooth, color="C4", linewidth=2, label="Inverse model (Fig. 11)")
    ax.set_ylabel("MSE")
    ax.set_title("Cerebellar Model Errors — Supervised Learning (Eq. 5)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Episode")

    # --- Panel 5: Test comparison (if available) ---
    if has_test:
        ax = axes[4]
        names = list(test_results.keys())
        means = [np.mean(v) for v in test_results.values()]
        stds = [np.std(v) for v in test_results.values()]
        colors = ["C0", "C3", "C4", "C5"][:len(names)]

        labels = {
            "bg": "BG\n(4.1)",
            "predictive": "Predictive\n(4.2.1)",
            "inverse": "Inverse\n(Fig. 11)",
            "hybrid": "Hybrid\n(v2)",
        }
        x_labels = [labels.get(n, n) for n in names]

        bars = ax.bar(x_labels, means, yerr=stds, color=colors, alpha=0.8,
                      capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Steps survived")
        ax.set_title("Test Phase — Strategy Comparison (50 trials)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=500, color="gray", linestyle="--", alpha=0.5, label="Max (500)")
        ax.legend(loc="upper left")

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("")

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training report saved: {filename}")


# ======================================================================
# Main — interactive architecture selection
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Doya (1999) architectures on CartPole-v1"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["reactive", "predictive"],
        default=None,
        help="Architecture: 'reactive' (Section 4.1) or 'predictive' (Section 4.2.1)",
    )
    parser.add_argument(
        "--episodes", "-n", type=int, default=1500,
        help="Number of training episodes (default: 1500)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip visual simulation after training",
    )
    args = parser.parse_args()

    # Interactive selection if no argument given
    if args.mode is None:
        print()
        print("=" * 56)
        print("  Doya (1999) — CartPole-v1")
        print("=" * 56)
        print()
        print("  Choose an architecture:")
        print()
        print("    [1] Section 4.1  — Reactive action selection (Fig. 6)")
        print("        Model-free actor-critic. The BG selects actions")
        print("        based solely on the learned value function.")
        print()
        print("    [2] Section 4.2.1 — Discrete model-based (Fig. 7)")
        print("        The cerebellar forward model predicts the outcome")
        print("        of each candidate action. The predictive TD error")
        print("        (Eq. 22) selects the best one.")
        print()

        while True:
            choice = input("  Enter 1 or 2: ").strip()
            if choice == "1":
                mode = "reactive"
                break
            elif choice == "2":
                mode = "predictive"
                break
            else:
                print("  Please enter 1 or 2.")
    else:
        mode = args.mode

    print()

    # Train
    results = train(mode=mode, n_episodes=args.episodes, print_every=100)
    cortex, bg, fwd, inv, rewards, lengths, metrics = results

    # Learning curve summary
    print()
    print("-" * 72)
    print("  Learning curve summary (avg steps survived):")
    n = len(rewards)
    bin_size = n // 4
    for i in range(4):
        s = i * bin_size
        e = (i + 1) * bin_size
        avg_r = np.mean(rewards[s:e])
        print(f"    Episodes {s+1:4d}-{e:4d}: {avg_r:7.1f} steps")
    print("-" * 72)

    # Statistical test (capture results for plot)
    test_results = run_test(cortex, bg, fwd, inv)

    # Generate training report PNG
    plot_training_report(metrics, mode, test_results=test_results)

    # Visual simulation
    if not args.no_render:
        print("  Visual simulation — press Ctrl+C to skip")
        print()

        strategies = ["bg", "predictive", "inverse", "hybrid"]
        for strat in strategies:
            try:
                simulate(cortex, bg, fwd, inv, strategy=strat, n_episodes=2)
            except KeyboardInterrupt:
                print("\n  Skipped.")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
