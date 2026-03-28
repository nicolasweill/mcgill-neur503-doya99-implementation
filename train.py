"""
Training & Testing — Integrated Cortex / Basal Ganglia / Cerebellum Model

Demonstrates Doya's (1999, 2000) framework of learning-oriented
specialization on a 2D arm-reaching task:

    - Cerebral cortex (unsupervised): learns a compact representation
      of the 6D sensory state (arm position, velocity, target position).
    - Basal ganglia (reinforcement learning): learns to select 2D
      motor commands that move the arm toward the target using TD
      learning with dopamine-like reward signals.
    - Cerebellum (supervised learning):
        * Forward model learns to predict the next state from
          current state and action (Eq. 21 of Doya 1999).
        * Inverse model learns to replicate the successful actions
          discovered by the basal ganglia (encapsulation, Fig. 11).

The task parallels the arm-reaching paradigm discussed in both papers
(Section "Arm reaching" of Doya 2000), where the cerebellum handles
visually guided movement (forward/inverse models) and the basal
ganglia handle reward-based action selection.

After training, we show the transition from basal ganglia-driven
exploration to cerebellar-driven automatic execution — the
encapsulation of learned skill described in Section 4.3.3.

Usage
-----
    python train.py
"""

import sys
import numpy as np

from cerebral_cortex import CerebralCortex
from basal_ganglia import BasalGanglia
from cerebellum import ForwardModel, InverseModel, CerebellarCorrector, HybridController


# ======================================================================
# Environment: 2D point-mass reaching task
# ======================================================================

class ReachingEnv:
    """Simple 2D reaching environment.

    An effector (point mass) must reach a target position.
    The state is [pos_x, pos_y, vel_x, vel_y, target_x, target_y].
    Actions are 2D forces clipped to [-1, 1].

    This models the essential aspects of the arm-reaching paradigm
    discussed in Doya (2000): coordinate transformation of visual
    input to motor output (cerebellum), and action selection based
    on reward prediction (basal ganglia).
    """

    def __init__(self, dt: float = 0.15, max_steps: int = 60):
        self.dt = dt
        self.max_steps = max_steps
        self.damping = 0.90  # velocity damping (like joint friction)
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset to a random start with a random target."""
        self.pos = np.random.uniform(-0.8, 0.8, size=2)
        self.vel = np.zeros(2)
        self.target = np.random.uniform(-0.8, 0.8, size=2)
        # Ensure a minimum distance so the task is non-trivial
        while np.linalg.norm(self.pos - self.target) < 0.3:
            self.target = np.random.uniform(-0.8, 0.8, size=2)
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """6D state: [pos, vel, target]."""
        return np.concatenate([self.pos, self.vel, self.target])

    def step(self, action: np.ndarray):
        """Apply action and advance the simulation.

        Returns (next_state, reward, done).
        """
        action = np.clip(action, -1, 1)

        prev_dist = np.linalg.norm(self.pos - self.target)

        # Simple dynamics: F = ma, with damping
        self.vel = self.damping * self.vel + self.dt * action
        self.pos = self.pos + self.dt * self.vel

        # Clip position to workspace
        self.pos = np.clip(self.pos, -2, 2)

        self.step_count += 1

        # Distance to target
        dist = np.linalg.norm(self.pos - self.target)

        # Reward shaping:
        # 1) Progress reward: positive when moving closer to target
        #    This creates a clear dopamine-like TD signal
        # 2) Proximity bonus: stronger reward when close
        # 3) Reach bonus: large reward for reaching the target
        # 4) Small action cost to encourage efficiency
        progress = prev_dist - dist
        reward = 5.0 * progress + 0.5 * max(0, 1.0 - dist) - 0.01 * np.sum(action ** 2)
        done = False

        if dist < 0.15:
            reward += 10.0
            done = True
        elif self.step_count >= self.max_steps:
            reward -= 2.0
            done = True

        return self._get_state(), reward, done


# ======================================================================
# State augmentation (cortical feature extraction)
# ======================================================================

def _augment_state(raw_state: np.ndarray, cortex_repr: np.ndarray) -> np.ndarray:
    """Create an augmented state for the BG with useful features.

    Includes raw state, cortical representation, and derived features
    (direction to target, distance, velocity alignment) that help the
    linear actor-critic learn. In Doya's framework, this corresponds to
    the cortex providing 'concise representation of sensory state,
    context, and action' (Table 1 of Doya 2000).

    The cortex extracts non-linear features from the raw input via
    unsupervised learning, complementing the hand-crafted geometric
    features. Together they form the common representational basis
    upon which the basal ganglia can operate.
    """
    pos = raw_state[:2]
    vel = raw_state[2:4]
    target = raw_state[4:6]

    direction = target - pos
    dist = np.linalg.norm(direction)
    if dist > 1e-8:
        unit_dir = direction / dist
    else:
        unit_dir = np.zeros(2)

    speed = np.linalg.norm(vel)
    vel_alignment = np.dot(vel, unit_dir) if dist > 1e-8 else 0.0

    features = np.concatenate([
        direction,          # 2: vector to target (most important signal)
        unit_dir,           # 2: normalized direction
        [dist],             # 1: scalar distance
        vel,                # 2: current velocity
        [speed],            # 1: scalar speed
        [vel_alignment],    # 1: alignment of velocity with target direction
        unit_dir * speed,   # 2: projected velocity components
        cortex_repr,        # repr_dim: learned nonlinear features
    ])
    return features


# ======================================================================
# Training loop
# ======================================================================

def train(
    n_episodes: int = 1000,
    print_every: int = 100,
    seed: int = 42,
):
    """Train the integrated model on the reaching task.

    Architecture follows Doya (1999) Fig. 6 (reactive action selection)
    combined with Fig. 11 (encapsulation of learned mapping):

        sensory input -> Cortex (unsupervised) -> state repr.
        [raw + cortex features] -> Basal Ganglia (RL) -> action selection
        (state, action, next_state) -> Cerebellum forward model (supervised)
        (state, target, action)     -> Cerebellum inverse model (supervised)
    """
    np.random.seed(seed)

    env = ReachingEnv(max_steps=60)
    raw_dim = 6   # [pos, vel, target]
    repr_dim = 8  # cortical representation dimension
    action_dim = 2

    # The augmented state dimension:
    # direction(2) + unit_dir(2) + dist(1) + vel(2) + speed(1) +
    # alignment(1) + proj_vel(2) + cortex_repr(repr_dim) = 11 + repr_dim
    augmented_dim = 11 + repr_dim

    # --- Initialize the three brain modules ---

    cortex = CerebralCortex(
        input_dim=raw_dim,
        repr_dim=repr_dim,
        lr=0.005,
        sparsity=0.02,
        relaxation_steps=10,
        relaxation_dt=0.03,
    )

    basal_ganglia = BasalGanglia(
        state_dim=augmented_dim,
        action_dim=action_dim,
        gamma=0.95,
        lr_critic=0.01,
        lr_actor=0.005,
        noise_std=0.5,
    )

    forward_model = ForwardModel(
        state_dim=raw_dim,
        action_dim=action_dim,
        n_granule=256,
        lr=0.01,
        grad_clip=2.0,
    )

    inverse_model = InverseModel(
        state_dim=raw_dim,
        action_dim=action_dim,
        n_granule=256,
        lr=0.01,
        grad_clip=2.0,
    )

    # --- Tracking metrics ---
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    cortex_errors = []
    forward_errors = []
    inverse_errors = []
    td_magnitudes = []

    print("=" * 72)
    print("  Doya (1999/2000) Integrated Brain Learning Model")
    print("=" * 72)
    print(f"  Cortex:        unsupervised, {raw_dim}D -> {repr_dim}D repr")
    print(f"  Basal Ganglia: actor-critic RL, {augmented_dim}D state -> {action_dim}D action")
    print(f"  Cerebellum:    forward model ({raw_dim}+{action_dim}D -> {raw_dim}D)")
    print(f"                 inverse model ({raw_dim}*2D -> {action_dim}D)")
    print(f"  Environment:   2D reaching, max {env.max_steps} steps")
    print("=" * 72)
    print()

    # Decay exploration noise over training (Doya 1999 Section 6.5:
    # the "temperature" parameter regulates exploration)
    initial_noise = 0.5
    final_noise = 0.05

    for episode in range(1, n_episodes + 1):
        raw_state = env.reset()
        basal_ganglia.reset()

        # Anneal exploration noise
        progress = episode / n_episodes
        basal_ganglia.noise_std = (
            initial_noise * (1 - progress) + final_noise * progress
        )

        total_reward = 0.0
        ep_cortex_err = []
        ep_fwd_err = []
        ep_inv_err = []
        ep_td = []

        # --- Cortex: encode initial state ---
        cortex_repr, cx_err = cortex.encode_and_learn(raw_state)
        ep_cortex_err.append(cx_err)

        # Augment state with cortical features for the BG
        bg_state = _augment_state(raw_state, cortex_repr)

        # --- BG: select first action ---
        action = basal_ganglia.step(bg_state, explore=True)

        done = False
        reached = False
        while not done:
            # --- Environment step ---
            raw_next_state, reward, done = env.step(action)
            total_reward += reward
            if reward > 5.0:
                reached = True

            # --- Cortex: learn representation of new state ---
            next_cortex_repr, cx_err = cortex.encode_and_learn(raw_next_state)
            ep_cortex_err.append(cx_err)
            next_bg_state = _augment_state(raw_next_state, next_cortex_repr)

            # --- Basal Ganglia: TD learning (dopamine signal) ---
            delta = basal_ganglia.learn(reward, next_bg_state, done)
            ep_td.append(abs(delta))

            # --- Cerebellum: forward model learns state transition ---
            # Uses raw state (mossy fiber input carries sensory afferents)
            fwd_err = forward_model.learn_transition(
                raw_state, action, raw_next_state
            )
            ep_fwd_err.append(fwd_err)

            # --- Cerebellum: inverse model learns action mapping ---
            # Teacher signal: the action chosen by the BG.
            # This implements "encapsulation" (Fig. 11 of Doya 1999):
            # the cerebellum learns to replicate the cortical feedback
            # pathway's input-output mapping.
            target_pos_state = np.concatenate([
                env.target, np.zeros(2), env.target
            ])
            inv_err = inverse_model.learn_action(
                raw_state, target_pos_state, action
            )
            ep_inv_err.append(inv_err)

            if not done:
                # Prepare next step
                raw_state = raw_next_state
                bg_state = next_bg_state
                action = basal_ganglia.step(bg_state, explore=True)

        # --- Log episode metrics ---
        episode_rewards.append(total_reward)
        episode_lengths.append(env.step_count)
        episode_successes.append(1 if reached else 0)
        cortex_errors.append(np.mean(ep_cortex_err))
        forward_errors.append(np.mean(ep_fwd_err) if ep_fwd_err else 0)
        inverse_errors.append(np.mean(ep_inv_err) if ep_inv_err else 0)
        td_magnitudes.append(np.mean(ep_td) if ep_td else 0)

        if episode % print_every == 0 or episode == 1:
            w = min(print_every, episode)
            avg_reward = np.mean(episode_rewards[-w:])
            avg_length = np.mean(episode_lengths[-w:])
            avg_cx = np.mean(cortex_errors[-w:])
            avg_fwd = np.mean(forward_errors[-w:])
            avg_inv = np.mean(inverse_errors[-w:])
            avg_td = np.mean(td_magnitudes[-w:])
            success_rate = np.mean(episode_successes[-w:]) * 100

            print(
                f"Ep {episode:4d} | "
                f"R {avg_reward:7.2f} | "
                f"Steps {avg_length:5.1f} | "
                f"Succ {success_rate:5.1f}% | "
                f"DA {avg_td:.3f} | "
                f"Cx {avg_cx:.3f} | "
                f"Fwd {avg_fwd:.4f} | "
                f"Inv {avg_inv:.4f}"
            )

    return (
        env, cortex, basal_ganglia, forward_model, inverse_model,
        episode_rewards, episode_lengths, episode_successes,
    )


# ======================================================================
# Testing: compare BG-driven vs cerebellum-driven execution
# ======================================================================

def test(
    env: ReachingEnv,
    cortex: CerebralCortex,
    basal_ganglia: BasalGanglia,
    forward_model: ForwardModel,
    inverse_model: InverseModel,
    n_trials: int = 50,
    seed: int = 123,
):
    """Compare performance of three control strategies:

    1. Basal ganglia (RL policy, no exploration)
    2. Cerebellum inverse model (encapsulated skill)
    3. Cerebellum forward model + BG value (predictive selection, Fig. 7)

    This demonstrates the transition from deliberate RL-based control
    to automatic cerebellar execution described in Doya (1999, 2000).
    """
    np.random.seed(seed)

    print()
    print("=" * 72)
    print("  TEST PHASE: Comparing Control Strategies")
    print("=" * 72)

    # Build the hybrid controller (v2 features)
    corrector = CerebellarCorrector(forward_model, correction_gain=0.5)
    hybrid = HybridController(
        forward_model, inverse_model, corrector,
        surprise_threshold_factor=2.0,
    )

    results = {"bg": [], "inverse": [], "predictive": [], "hybrid": []}

    for trial in range(n_trials):
        np.random.seed(seed + trial)

        # --- Strategy 1: Basal Ganglia policy (no noise) ---
        raw_state = env.reset()
        start_pos = env.pos.copy()
        target_pos = env.target.copy()
        basal_ganglia.reset()

        total_r = 0.0
        done = False
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)
        while not done:
            action = basal_ganglia.policy(bg_state, explore=False)
            raw_state, r, done = env.step(action)
            total_r += r
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)
        final_dist_bg = np.linalg.norm(env.pos - env.target)
        results["bg"].append((total_r, env.step_count, final_dist_bg))

        # --- Strategy 2: Cerebellar inverse model ---
        np.random.seed(seed + trial)
        env.reset()
        env.pos = start_pos.copy()
        env.vel = np.zeros(2)
        env.target = target_pos.copy()
        env.step_count = 0

        total_r = 0.0
        done = False
        raw_state = env._get_state()
        target_raw = np.concatenate([target_pos, np.zeros(2), target_pos])
        while not done:
            action = inverse_model.compute_action(raw_state, target_raw)
            action = np.clip(action, -1, 1)
            raw_state, r, done = env.step(action)
            total_r += r
            raw_state = env._get_state()
        final_dist_inv = np.linalg.norm(env.pos - env.target)
        results["inverse"].append((total_r, env.step_count, final_dist_inv))

        # --- Strategy 3: Predictive selection (forward model + BG value) ---
        # Implements Fig. 7 of Doya (1999): use the forward model to
        # predict outcomes of candidate actions, then select the action
        # whose predicted next state has the highest BG value.
        np.random.seed(seed + trial)
        env.reset()
        env.pos = start_pos.copy()
        env.vel = np.zeros(2)
        env.target = target_pos.copy()
        env.step_count = 0

        total_r = 0.0
        done = False
        raw_state = env._get_state()
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)
        n_candidates = 20

        while not done:
            base_action = basal_ganglia.policy(bg_state, explore=False)
            best_action = base_action.copy()
            best_value = -1e10

            for _ in range(n_candidates):
                candidate = base_action + np.random.randn(2) * 0.3
                candidate = np.clip(candidate, -1, 1)

                # Predict outcome using cerebellar forward model
                pred_next_raw = forward_model.predict_next_state(
                    raw_state, candidate
                )
                pred_cortex = cortex.encode(pred_next_raw)
                pred_bg_state = _augment_state(pred_next_raw, pred_cortex)

                # Evaluate with BG value function (Eq. 22 of Doya 1999)
                v = basal_ganglia.value(pred_bg_state)
                if v > best_value:
                    best_value = v
                    best_action = candidate

            raw_state, r, done = env.step(best_action)
            total_r += r
            raw_state = env._get_state()
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)
        final_dist_pred = np.linalg.norm(env.pos - env.target)
        results["predictive"].append((total_r, env.step_count, final_dist_pred))

        # --- Strategy 4: Hybrid controller (v2) ---
        # Cerebellum with online correction + automatic BG fallback
        # when forward model prediction error signals surprise.
        np.random.seed(seed + trial)
        env.reset()
        env.pos = start_pos.copy()
        env.vel = np.zeros(2)
        env.target = target_pos.copy()
        env.step_count = 0

        hybrid.reset()
        total_r = 0.0
        done = False
        raw_state = env._get_state()
        target_raw = np.concatenate([target_pos, np.zeros(2), target_pos])
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)
        prev_raw = None

        while not done:
            action, source, pred_err = hybrid.select_action(
                raw_state, bg_state, target_raw, basal_ganglia,
                observed_state=raw_state,
            )
            # Record for next step's correction
            corrector.begin_step(raw_state, action)

            raw_state, r, done = env.step(action)
            total_r += r
            raw_state = env._get_state()
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)

        final_dist_hyb = np.linalg.norm(env.pos - env.target)
        results["hybrid"].append((total_r, env.step_count, final_dist_hyb))

    # --- Print results ---
    print()
    print(f"  {'Strategy':<42s} | {'Reward':>12s} | {'Steps':>5s} | "
          f"{'Dist':>8s} | {'Reached':>7s}")
    print("  " + "-" * 86)

    for name, label in [
        ("bg", "Basal Ganglia (RL policy)"),
        ("inverse", "Cerebellum (inverse model)"),
        ("predictive", "Predictive (fwd model + BG value)"),
        ("hybrid", "Hybrid (correction + auto-switch)"),
    ]:
        rewards = [r for r, _, _ in results[name]]
        steps = [s for _, s, _ in results[name]]
        dists = [d for _, _, d in results[name]]
        successes = sum(1 for _, _, d in results[name] if d < 0.15)
        print(
            f"  {label:<42s} | "
            f"{np.mean(rewards):6.2f}+/-{np.std(rewards):4.2f} | "
            f"{np.mean(steps):5.1f} | "
            f"{np.mean(dists):8.3f} | "
            f"{successes:3d}/{n_trials}"
        )

    # Hybrid controller statistics
    print()
    print(f"  Hybrid controller: {hybrid.cerebellum_ratio*100:.1f}% cerebellar, "
          f"{(1 - hybrid.cerebellum_ratio)*100:.1f}% BG fallback")
    print(f"  Surprise threshold: {hybrid.surprise_threshold:.4f}")
    print()
    print("  Key observations (cf. Doya 1999, 2000):")
    print("  - The BG policy learns reward-based action selection via TD learning")
    print("  - The cerebellar inverse model encapsulates the BG's learned skill")
    print("    (Section 4.3.3, Fig. 11: 'encapsulation of learned mappings')")
    print("  - The predictive strategy combines the cerebellar forward model")
    print("    with the BG value function (Fig. 7: 'action selection with a")
    print("    forward model')")
    print("  - The hybrid controller uses the forward model as a surprise detector:")
    print("    cerebellar control for familiar states, automatic BG fallback")
    print("    when prediction error exceeds the adaptive threshold (v2)")
    print("  - Online correction uses the forward model prediction error to")
    print("    generate fine motor adjustments at each timestep (v2)")
    print()


# ======================================================================
# Main
# ======================================================================

def main():
    print()
    print("Training the integrated cortex-BG-cerebellum model...")
    print()

    results = train(n_episodes=1000, print_every=100)
    env, cortex, bg, fwd, inv, rewards, lengths, successes = results

    # Show learning curve summary
    print()
    print("-" * 72)
    print("  Learning curve summary:")
    bins = [(1, 250), (251, 500), (501, 750), (751, 1000)]
    for s, e in bins:
        if e <= len(rewards):
            sl = slice(s - 1, e)
            avg_r = np.mean(rewards[sl])
            avg_l = np.mean(lengths[sl])
            succ = np.mean(successes[sl]) * 100
            print(
                f"    Episodes {s:4d}-{e:4d}: "
                f"reward = {avg_r:7.2f}, "
                f"steps = {avg_l:5.1f}, "
                f"success = {succ:5.1f}%"
            )
    print("-" * 72)

    # Test the three control strategies
    test(env, cortex, bg, fwd, inv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
