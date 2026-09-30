"""
Training & Testing V2 — Stabilized Doya (1999) Architectures on CartPole-v1

Extends train_cartpole.py with:

  1. _augment_state_v2: adds angle² and ang_vel² polynomial features
     (10 + repr_dim = 16 total) to give the linear BG quadratic
     expressivity without changing its architecture.

  2. BasalGangliaV2: stabilized Actor-Critic with delta clipping,
     Welford running normalization, and weight clipping.

  3. BasalGangliaMario adapter: the MLP Actor-Critic from the Mario
     branch (BasalGangliaMario) can be plugged into CartPole via
     bg_type="mario" in train_v2(). It uses discrete actions natively
     (n_actions=2) and Adam optimization.

Both bg_type="v2" and bg_type="mario" share _augment_state_v2,
the same cerebral cortex, and the same cerebellar models, so
differences in the benchmark isolate the BG contribution.

Usage
-----
    from train_cartpolev2 import train_v2, run_test_v2

    results = train_v2(mode="reactive", n_episodes=1500, bg_type="v2")
    cortex, bg, fwd, inv, rewards, lengths, metrics = results
    test_results = run_test_v2(cortex, bg, fwd, inv, bg_type="v2")
"""

import sys
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
from basal_ganglia_v2 import BasalGangliaV2
from basal_ganglia_mario import BasalGangliaMario
from basal_ganglia_mario_hebbian import BasalGangliaMarioHebbian
from basal_ganglia_actrace import BasalGangliaACTrace
from cerebellum import (
    ForwardModel,
    InverseModel,
    CerebellarCorrector,
    HybridController,
)
from train_cartpole import _augment_state, _action_for_cerebellum, _smooth


# ======================================================================
# V2 State augmentation — adds polynomial features
# ======================================================================

def _augment_state_v2(obs: np.ndarray, cortex_repr: np.ndarray) -> np.ndarray:
    """Augment CartPole observation with cortical features + polynomial terms.

    CartPole obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    Extends _augment_state (8 features) with quadratic terms that give
    the linear Actor-Critic the ability to represent non-linear
    interactions without changing its architecture:
      - angle²    : captures symmetric danger zones (large deviation
                    in either direction)
      - ang_vel²  : captures quadratic cost of angular velocity

    Note: angle * ang_vel (the falling indicator) already exists in the
    base features and is preserved here.

    Returns
    -------
    features : np.ndarray, shape (10 + repr_dim,)
        16-dimensional augmented state when repr_dim=6.
    """
    cart_pos, cart_vel, angle, ang_vel = obs[0], obs[1], obs[2], obs[3]

    features = np.concatenate([
        obs,                          # 4: raw observation
        [abs(angle)],                 # 1: deviation magnitude
        [angle * ang_vel],            # 1: falling indicator
        [np.sign(cart_pos)],          # 1: side of center
        [cart_vel * np.sign(angle)],  # 1: velocity alignment with tilt
        [angle ** 2],                 # 1: [NEW] quadratic deviation
        [ang_vel ** 2],               # 1: [NEW] quadratic angular velocity
        cortex_repr,                  # repr_dim: learned cortical features
    ])
    return features


def _augment_state_nonlinear(obs: np.ndarray, cortex_repr: np.ndarray) -> np.ndarray:
    """Augment CartPole observation with basic features + polynomial terms.

    CartPole obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]

    Extends the original 8 handcrafted features with quadratic polynomial
    terms (angle² and ang_vel²) to give the linear Actor-Critic the ability
    to represent non-linear interactions without changing its architecture.
    
    This variant uses the base 8 features (like _augment_state from train_cartpole.py)
    plus the 2 polynomial features, for a total of 10 + repr_dim = 16D state.

    Returns
    -------
    features : np.ndarray, shape (10 + repr_dim,)
        16-dimensional augmented state when repr_dim=6.
    """
    cart_pos, cart_vel, angle, ang_vel = obs[0], obs[1], obs[2], obs[3]

    features = np.concatenate([
        obs,                          # 4: raw observation
        [abs(angle)],                 # 1: deviation magnitude
        [angle * ang_vel],            # 1: falling indicator
        [np.sign(cart_pos)],          # 1: side of center
        [cart_vel * np.sign(angle)],  # 1: velocity alignment with tilt
        [angle ** 2],                 # 1: [NEW] quadratic deviation
        [ang_vel ** 2],               # 1: [NEW] quadratic angular velocity
        cortex_repr,                  # repr_dim: learned cortical features
    ])
    return features


# ======================================================================
# Adapters — handle BasalGangliaV2 (ndarray) and BasalGangliaMario (int)
# ======================================================================

def _select_action_reactive_v2(
    bg,
    bg_state: np.ndarray,
    explore: bool,
) -> tuple:
    """Reactive action selection for both BG types.

    BasalGangliaV2.policy() returns np.ndarray (continuous action).
    BasalGangliaMario.policy() returns int (discrete action index).

    Returns
    -------
    continuous : np.ndarray, shape (1,)
        Continuous action encoding for cerebellum learning.
    discrete : int
        Discrete action (0 = left, 1 = right) for gym.step().
    """
    if isinstance(bg, (BasalGangliaMario, BasalGangliaMarioHebbian, BasalGangliaACTrace)):
        discrete = int(bg.policy(bg_state, explore=explore))
        continuous = np.array([1.0 if discrete == 1 else -1.0])
    else:
        continuous = bg.policy(bg_state, explore=explore)
        discrete = 1 if continuous[0] > 0 else 0
    return continuous, discrete


def _select_action_predictive_v2(
    bg,
    forward_model: ForwardModel,
    cortex: CerebralCortex,
    raw_state: np.ndarray,
    bg_state: np.ndarray,
    explore: bool,
    n_candidates: int = 5,
    augment_fn=None,
) -> tuple:
    """Discrete model-based action selection using _augment_state_v2.

    Evaluates all discrete actions via the forward model's predicted
    next state, then selects the one with the highest predictive
    TD error (Eq. 22 of Doya 1999).

    For BasalGangliaMario: only the two deterministic actions {0, 1}
    are evaluated (softmax already provides stochastic exploration).
    For BasalGangliaV2: additional noise-perturbed candidates are
    generated as in the original _select_action_predictive.

    Returns
    -------
    continuous : np.ndarray, shape (1,)
    discrete : int
    """
    if augment_fn is None:
        augment_fn = _augment_state_v2

    current_value = bg.value(bg_state)
    gamma = bg.gamma

    best_delta = -1e10
    best_continuous = None
    best_discrete = 0

    # Evaluate both discrete actions
    for d_action in [0, 1]:
        action_enc = _action_for_cerebellum(d_action)
        pred_next_raw = forward_model.predict_next_state(raw_state, action_enc)
        pred_cortex = cortex.encode(pred_next_raw)
        pred_bg_state = augment_fn(pred_next_raw, pred_cortex)

        pred_value = bg.value(pred_bg_state)
        delta_star = gamma * pred_value - current_value

        if delta_star > best_delta:
            best_delta = delta_star
            best_discrete = d_action
            best_continuous = action_enc.copy()

    # Stochastic exploration for ACTrace: with temperature-dependent probability,
    # fall back to the actor's stochastic policy instead of the greedy planner.
    # This ensures the actor explores and the eligibility traces see diverse actions.
    if explore and isinstance(bg, BasalGangliaACTrace):
        temp = getattr(bg, 'temperature', 1.0)
        # Higher temperature → more exploration fallback
        # temp=2.0 → 50%, temp=0.5 → 12.5%
        explore_prob = min(0.5, temp / 4.0)
        if np.random.rand() < explore_prob:
            fallback_action = bg.policy(bg_state, explore=True)
            best_discrete = fallback_action
            best_continuous = _action_for_cerebellum(best_discrete)
            return best_continuous, best_discrete

    # For continuous BG (V2): add noise-perturbed candidates
    if explore and not isinstance(bg, (BasalGangliaMario, BasalGangliaMarioHebbian, BasalGangliaACTrace)):
        bg_action = bg.policy(bg_state, explore=True)
        for _ in range(n_candidates - 2):
            candidate = bg_action + np.random.randn(1) * bg.noise_std
            d_cand = 1 if candidate[0] > 0 else 0
            action_enc = _action_for_cerebellum(d_cand)
            pred_next_raw = forward_model.predict_next_state(raw_state, action_enc)
            pred_cortex = cortex.encode(pred_next_raw)
            pred_bg_state = augment_fn(pred_next_raw, pred_cortex)

            pred_value = bg.value(pred_bg_state)
            delta_star = gamma * pred_value - current_value

            if delta_star > best_delta:
                best_delta = delta_star
                best_discrete = d_cand
                best_continuous = action_enc.copy()

    if best_continuous is None:
        best_continuous = _action_for_cerebellum(best_discrete)

    return best_continuous, best_discrete


# ======================================================================
# Training loop V2
# ======================================================================

def train_v2(
    mode: str = "reactive",
    n_episodes: int = 1500,
    print_every: int = 100,
    seed: int = 42,
    bg_type: str = "v2",
    feature_variant: str = "standard",
):
    """Train the V2 integrated model on CartPole-v1.

    Parameters
    ----------
    mode : str
        "reactive" (Section 4.1) or "predictive" (Section 4.2.1).
    n_episodes : int
        Number of training episodes.
    print_every : int
        Logging interval.
    seed : int
        Random seed for reproducibility.
    bg_type : str
        "v2"    — BasalGangliaV2 (stabilized numpy Actor-Critic)
        "mario" — BasalGangliaMario (MLP Actor-Critic with Adam)
        "original" — BasalGanglia (basic numpy Actor-Critic)
    feature_variant : str
        "standard"   — original 8 handcrafted features (14D total)
        "nonlinear"  — 8 handcrafted + 2 polynomial features (16D total)

    Returns
    -------
    cortex, basal_ganglia, forward_model, inverse_model,
    episode_rewards, episode_lengths, metrics
    """
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = 4
    repr_dim = 6
    action_dim = 1

    # Feature augmentation: select function based on feature_variant
    if feature_variant == "standard":
        augment_fn = _augment_state       # 14D = 8 handcrafted + 6 cortex
        augmented_dim = 8 + repr_dim
    elif feature_variant == "nonlinear":
        augment_fn = _augment_state_nonlinear  # 16D = 10 handcrafted + 6 cortex
        augmented_dim = 10 + repr_dim
    else:
        raise ValueError(f"Unknown feature_variant: {feature_variant!r}. Use 'standard' or 'nonlinear'.")

    # --- Initialize cerebral cortex (shared across bg_type) ---
    cortex = CerebralCortex(
        input_dim=obs_dim,
        repr_dim=repr_dim,
        lr=0.005,
        sparsity=0.02,
        relaxation_steps=10,
        relaxation_dt=0.03,
    )

    # --- Initialize basal ganglia (depends on bg_type) ---
    if bg_type == "original":
        basal_ganglia = BasalGanglia(
            state_dim=augmented_dim,
            action_dim=action_dim,
            gamma=0.99,
            lr_critic=0.02,
            lr_actor=0.01,
            noise_std=0.5,
        )
    elif bg_type == "v2":
        basal_ganglia = BasalGangliaV2(
            state_dim=augmented_dim,
            action_dim=action_dim,
            gamma=0.99,
            lr_critic=0.02,
            lr_actor=0.01,
            noise_std=0.5,
            delta_clip=5.0,
            max_weight=10.0,
        )
    elif bg_type == "mario":
        basal_ganglia = BasalGangliaMario(
            state_dim=augmented_dim,
            n_actions=2,
            gamma=0.99,
            lr_critic=3e-4,
            lr_actor=3e-4,
            temperature=2.0,
            device="cpu",
        )
    elif bg_type == "hebbian":
        basal_ganglia = BasalGangliaMarioHebbian(
            state_dim=augmented_dim,
            n_actions=2,
            gamma=0.99,
            lr_critic=1e-4,
            lr_actor=1e-4,
            decay_lambda=0.9,
            temperature=2.0,
            device="cpu",
        )
    elif bg_type == "actrace":
        basal_ganglia = BasalGangliaACTrace(
            state_dim=augmented_dim,
            n_actions=2,
            gamma=0.99,
            lr_critic=1e-3,
            lr_actor=1e-3,
            decay_lambda=0.9,
            entropy_beta=0.01,
            temperature=2.0,
            device="cpu",
        )
    else:
        raise ValueError(f"Unknown bg_type: {bg_type!r}. Use 'original', 'v2', 'mario', 'hebbian', or 'actrace'.")

    # --- Initialize cerebellar models (shared) ---
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

    bg_label = {
        "original": "BasalGanglia (linear numpy)",
        "v2":       "BasalGangliaV2 (stabilized numpy)",
        "mario":    "BasalGangliaMario (MLP + Adam)",
        "hebbian":  "BasalGangliaMarioHebbian (Hebbian plasticity)",
        "actrace":  "BasalGangliaACTrace (TD(λ) eligibility traces)",
    }[bg_type]
    feature_label = {
        "standard":  "standard (8 features, 14D total)",
        "nonlinear": "nonlinear (10 features, 16D total)",
    }[feature_variant]
    mode_label = {
        "reactive":   "Section 4.1 — Reactive (Fig. 6)",
        "predictive": "Section 4.2.1 — Discrete model-based (Fig. 7)",
    }

    print("=" * 72)
    print(f"  Doya (1999) V2 — {mode_label[mode]}")
    print(f"  BG:            {bg_label}")
    print(f"  Features:      {feature_label}")
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
    initial_temperature = 2.0
    final_temperature = 0.5  # 0.1 caused catastrophic forgetting: Mario BG overexploited
                              # too early, specialized on poor states, then unlearned.

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        raw_state = np.array(obs, dtype=np.float64)
        basal_ganglia.reset()

        # Exploration decay schedule
        progress = episode / n_episodes
        if bg_type in ("v2", "original"):
            # These BG types use noise_std
            basal_ganglia.noise_std = (
                initial_noise * (1 - progress) + final_noise * progress
            )
        else:  # mario
            # BasalGangliaMario uses temperature
            basal_ganglia.temperature = (
                initial_temperature * (1 - progress) + final_temperature * progress
            )

        total_reward = 0.0
        ep_cx, ep_fwd, ep_inv, ep_td = [], [], [], []

        cortex_repr, cx_err = cortex.encode_and_learn(raw_state)
        ep_cx.append(cx_err)
        bg_state = augment_fn(raw_state, cortex_repr)

        # First action selection
        if mode == "reactive":
            # Single step() call: the stored _prev_action must match the
            # executed action exactly. A second policy() call would draw
            # different noise and corrupt the actor's learning signal.
            raw_action = basal_ganglia.step(bg_state, explore=True)
            if isinstance(raw_action, (int, np.integer)):
                discrete_action = int(raw_action)
                continuous_action = np.array([1.0 if discrete_action == 1 else -1.0])
            else:
                continuous_action = raw_action
                discrete_action = 1 if continuous_action[0] > 0 else 0
        else:
            # Predictive: For ACTrace, do bookkeeping explicitly to avoid storing
            # a false _prev_action. For other BG types, step() handles bookkeeping.
            if isinstance(basal_ganglia, BasalGangliaACTrace):
                basal_ganglia._prev_features = bg_state.copy()
                basal_ganglia._prev_value = basal_ganglia.value(bg_state)
                # _prev_action will be set via action_override in learn()
            else:
                basal_ganglia.step(bg_state, explore=True)
            continuous_action, discrete_action = _select_action_predictive_v2(
                basal_ganglia, forward_model, cortex,
                raw_state, bg_state, explore=True, augment_fn=augment_fn,
            )
        done = False
        truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, _ = env.step(discrete_action)
            raw_next = np.array(obs, dtype=np.float64)
            total_reward += reward

            next_cortex, cx_err = cortex.encode_and_learn(raw_next)
            ep_cx.append(cx_err)
            next_bg_state = augment_fn(raw_next, next_cortex)

            # BG learning
            terminal = done or truncated
            bg_reward = reward if not done else -10.0
            if mode == "predictive" and isinstance(basal_ganglia, BasalGangliaACTrace):
                delta = basal_ganglia.learn(bg_reward, next_bg_state, terminal, action_override=discrete_action)
            else:
                delta = basal_ganglia.learn(bg_reward, next_bg_state, terminal)
            ep_td.append(abs(delta))

            # Cerebellum: forward model
            action_enc = _action_for_cerebellum(discrete_action)
            fwd_err = forward_model.learn_transition(raw_state, action_enc, raw_next)
            ep_fwd.append(fwd_err)

            # Cerebellum: inverse model
            target_state = np.zeros(obs_dim)
            inv_err = inverse_model.learn_action(
                raw_state, target_state, continuous_action
            )
            ep_inv.append(inv_err)

            if not terminal:
                raw_state = raw_next
                bg_state = next_bg_state

                # Next action
                if mode == "reactive":
                    raw_action = basal_ganglia.step(bg_state, explore=True)
                    if isinstance(raw_action, (int, np.integer)):
                        discrete_action = int(raw_action)
                        continuous_action = np.array([1.0 if discrete_action == 1 else -1.0])
                    else:
                        continuous_action = raw_action
                        discrete_action = 1 if continuous_action[0] > 0 else 0
                else:
                    # Predictive: For ACTrace, do bookkeeping explicitly to avoid storing
                    # a false _prev_action. For other BG types, step() handles bookkeeping.
                    if isinstance(basal_ganglia, BasalGangliaACTrace):
                        basal_ganglia._prev_features = bg_state.copy()
                        basal_ganglia._prev_value = basal_ganglia.value(bg_state)
                        # _prev_action will be set via action_override in learn()
                    else:
                        basal_ganglia.step(bg_state, explore=True)
                    continuous_action, discrete_action = _select_action_predictive_v2(
                        basal_ganglia, forward_model, cortex,
                        raw_state, bg_state, explore=True, augment_fn=augment_fn,
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
        "rewards":        episode_rewards,
        "cortex_errors":  cortex_errors,
        "forward_errors": forward_errors,
        "inverse_errors": inverse_errors,
        "td_magnitudes":  td_magnitudes,
    }

    return (
        cortex, basal_ganglia, forward_model, inverse_model,
        episode_rewards, episode_lengths, metrics,
    )


# ======================================================================
# Test phase V2
# ======================================================================

def run_test_v2(
    cortex: CerebralCortex,
    bg,
    forward_model: ForwardModel,
    inverse_model: InverseModel,
    n_trials: int = 50,
    seed: int = 999,
    bg_type: str = "v2",
    feature_variant: str = "standard",
) -> dict:
    """Compare control strategies on CartPole (no rendering).

    Uses augmentation function based on feature_variant.

    Parameters
    ----------
    feature_variant : str
        "standard" or "nonlinear" — determines which augmentation function to use.

    Returns
    -------
    results : dict
        Keys: "bg", "predictive", "inverse", "hybrid".
        Values: list of episode rewards over n_trials.
    """
    np.random.seed(seed)
    env = gym.make("CartPole-v1")

    if feature_variant == "standard":
        augment_fn = _augment_state
    elif feature_variant == "nonlinear":
        augment_fn = _augment_state_nonlinear
    else:
        raise ValueError(f"Unknown feature_variant: {feature_variant!r}")

    corrector = CerebellarCorrector(forward_model, correction_gain=0.3)
    hybrid = HybridController(
        forward_model, inverse_model, corrector,
        surprise_threshold_factor=2.0,
    )

    print()
    print("=" * 72)
    print(f"  TEST PHASE V2: CartPole — Comparing Control Strategies")
    print("=" * 72)

    results = {"bg": [], "predictive": [], "inverse": [], "hybrid": []}

    for trial in range(n_trials):
        trial_seed = seed + trial

        # --- Strategy 1: BG policy ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        bg.reset()
        cortex_repr = cortex.encode(raw_state)
        bg_state = augment_fn(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            _, discrete = _select_action_reactive_v2(bg, bg_state, explore=False)
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = augment_fn(raw_state, cortex_repr)
        results["bg"].append(total_r)

        # --- Strategy 2: Predictive ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        bg.reset()
        cortex_repr = cortex.encode(raw_state)
        bg_state = augment_fn(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            _, discrete = _select_action_predictive_v2(
                bg, forward_model, cortex,
                raw_state, bg_state, explore=False, augment_fn=augment_fn,
            )
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = augment_fn(raw_state, cortex_repr)
        results["predictive"].append(total_r)

        # --- Strategy 3: Inverse model (BG-type agnostic) ---
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

        # --- Strategy 4: Hybrid (BG-type agnostic) ---
        obs, _ = env.reset(seed=trial_seed)
        raw_state = np.array(obs, dtype=np.float64)
        hybrid.reset()
        cortex_repr = cortex.encode(raw_state)
        bg_state = augment_fn(raw_state, cortex_repr)

        total_r = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            target_raw = np.zeros(4)
            action, source, _ = hybrid.select_action(
                raw_state, bg_state, target_raw, bg,
                observed_state=raw_state,
            )
            # BasalGangliaMario returns int; corrector.begin_step needs ndarray
            if isinstance(action, (int, np.integer)):
                action = np.array([1.0 if action == 1 else -1.0])
            corrector.begin_step(raw_state, action)
            discrete = 1 if action[0] > 0 else 0
            obs, r, done, truncated, _ = env.step(discrete)
            total_r += r
            raw_state = np.array(obs, dtype=np.float64)
            cortex_repr = cortex.encode(raw_state)
            bg_state = augment_fn(raw_state, cortex_repr)
        results["hybrid"].append(total_r)

    env.close()

    print()
    print(f"  {'Strategy':<45s} | {'Reward (steps)':>14s} | {'Max':>5s}")
    print("  " + "-" * 72)

    for name, label in [
        ("bg",         "BG V2"),
        ("predictive", "Predictive V2"),
        ("inverse",    "Cerebellum inverse model"),
        ("hybrid",     "Hybrid — correction + switch"),
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
