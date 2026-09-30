"""
Mario Brain — Integration of Modular Learning Systems

Orchestrates three brain components for game playing:

  Visual Cortex: Extracts features from game images.

  Basal Ganglia: Selects actions based on current state and provides
  reward-based learning signals (dopamine).

  Cerebellum: Predicts action outcomes and learns internal models
  for skill development.

The dopamine signal from basal ganglia modulates learning in all
other modules, coordinating multi-system learning.
"""

import numpy as np
from collections import deque

from visual_cortex_cnn import VisualCortexCNN
from basal_ganglia_mario import BasalGangliaMario
from cerebellum_neural import ForwardModelNeural, InverseModelNeural, ICMModule


class MarioBrain:
    """Doya modular architecture for Super Mario Bros.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions (7 for SIMPLE_MOVEMENT).
    feature_dim : int
        Visual cortex output dimensionality.
    gamma : float
        Discount factor for BG.
    temperature_start : float
        Initial softmax temperature (high = exploratory).
    temperature_end : float
        Final softmax temperature (low = exploitative).
    temperature_anneal_epochs : int
        Number of epochs over which to anneal temperature.
    icm_weight : float
        Weight of ICM intrinsic reward.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_actions: int = 7,
        feature_dim: int = 256,
        gamma: float = 0.9,
        temperature_start: float = 2.5,
        temperature_end: float = 0.3,
        temperature_anneal_epochs: int = 800,
        icm_weight: float = 0.1,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.device = device
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_anneal_epochs = temperature_anneal_epochs

        # ── Visual Cortex: CNN backbone ──
        self.visual_cortex = VisualCortexCNN(
            n_channels=4,
            feature_dim=feature_dim,
            lr=3e-4,
            da_lr_scale=1.5,
            da_baseline_ema=0.05,
            reconstruction_weight=0.1,
            device=device,
        )

        # ── Basal Ganglia: actor-critic TD(λ) ──
        self.basal_ganglia = BasalGangliaMario(
            state_dim=feature_dim,
            n_actions=n_actions,
            gamma=gamma,
            lr_critic=1e-3,
            lr_actor=5e-4,
            decay_lambda=0.92,
            entropy_beta=0.02,
            temperature=temperature_start,
            delta_clip=10.0,
            max_weight=5.0,
            trace_clip=5.0,
            device=device,
        )

        # ── Cerebellum: forward + inverse models (ICM) ──
        self.forward_model = ForwardModelNeural(
            feature_dim=feature_dim,
            n_actions=n_actions,
            hidden_dim=256,
            lr=1e-3,
            da_lr_scale=2.0,
            da_baseline_ema=0.05,
            device=device,
        )

        self.inverse_model = InverseModelNeural(
            feature_dim=feature_dim,
            n_actions=n_actions,
            hidden_dim=256,
            lr=1e-3,
            da_lr_scale=2.0,
            da_baseline_ema=0.05,
            device=device,
        )

        self.icm = ICMModule(
            forward_model=self.forward_model,
            inverse_model=self.inverse_model,
            icm_weight=icm_weight,
            inverse_weight=0.2,
        )

        # ── State tracking ──
        self._prev_features = None
        self._prev_action = None
        self._prev_delta = None  # For delayed dopamine gating of cortex

        # ── Metrics tracking ──
        self._episode_deltas = []
        self._episode_recon_errors = []
        self._episode_forward_errors = []
        self._episode_inverse_errors = []
        self._episode_reward_breakdown = []
        self._episode_actions = []

        # ── Lifetime metrics ──
        self.total_steps = 0
        self.total_episodes = 0

    def reset_episode(self):
        """Reset all episode-level state."""
        self.basal_ganglia.reset()
        self._prev_features = None
        self._prev_action = None
        self._prev_delta = None
        self._episode_deltas = []
        self._episode_recon_errors = []
        self._episode_forward_errors = []
        self._episode_inverse_errors = []
        self._episode_reward_breakdown = []
        self._episode_actions = []

    def observe_and_act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Extract features from observation and select action.

        Parameters
        ----------
        obs : np.ndarray, shape (84, 84, 4) HWC uint8
            Stacked grayscale frames.
        explore : bool
            Whether to add exploration noise.

        Returns
        -------
        action : int
            Selected discrete action.
        """
        # Visual cortex: extract features with dopamine-gated learning
        features, recon_error = self.visual_cortex.learn(obs, delta=self._prev_delta)
        self._episode_recon_errors.append(recon_error)

        # BG: select action
        action = self.basal_ganglia.step(features, explore=explore)
        self._episode_actions.append(action)

        # Store for learning
        self._prev_features = features.copy()
        self._prev_action = action

        return action

    def learn_from_transition(
        self,
        next_obs: np.ndarray,
        extrinsic_reward: float,
        done: bool,
        info: dict,
    ) -> dict:
        """Learn from environment transition.

        Called AFTER observe_and_act() and environment step.

        Parameters
        ----------
        next_obs : np.ndarray, shape (84, 84, 4) HWC uint8
        extrinsic_reward : float
            Raw reward from environment.
        done : bool
            Whether episode terminated.
        info : dict
            Environment info (contains x_pos, life, etc.).

        Returns
        -------
        metrics : dict
            Learning metrics for logging.
        """
        if self._prev_features is None:
            return {}

        # Extract next features (no learning, just inference)
        next_features = self.visual_cortex.extract_features(next_obs)

        # ── ICM: compute intrinsic reward + train cerebellar models ──
        intrinsic_reward, fwd_loss, inv_loss = self.icm.compute_intrinsic_reward(
            self._prev_features,
            self._prev_action,
            next_features,
            delta=self._prev_delta,
        )
        self._episode_forward_errors.append(fwd_loss)
        self._episode_inverse_errors.append(inv_loss)

        # ── Reward shaping ──
        shaped_reward, breakdown = self.basal_ganglia.reward_shaper.shape_reward(
            extrinsic_reward, intrinsic_reward, info, done,
        )
        self._episode_reward_breakdown.append(breakdown)

        # ── BG: learn from shaped reward → δ (dopamine signal) ──
        delta = self.basal_ganglia.learn(shaped_reward, next_features, done)
        self._prev_delta = delta
        self._episode_deltas.append(abs(delta))

        self.total_steps += 1

        return {
            'delta': delta,
            'shaped_reward': shaped_reward,
            'intrinsic_reward': intrinsic_reward,
            'forward_error': fwd_loss,
            'inverse_error': inv_loss,
            'recon_error': self._episode_recon_errors[-1],
            'breakdown': breakdown,
        }

    def end_episode(self) -> dict:
        """Finalize episode and return summary metrics.

        Returns
        -------
        summary : dict
        """
        self.total_episodes += 1

        summary = {
            'mean_delta': np.mean(self._episode_deltas) if self._episode_deltas else 0.0,
            'mean_recon_error': np.mean(self._episode_recon_errors) if self._episode_recon_errors else 0.0,
            'mean_forward_error': np.mean(self._episode_forward_errors) if self._episode_forward_errors else 0.0,
            'mean_inverse_error': np.mean(self._episode_inverse_errors) if self._episode_inverse_errors else 0.0,
            'action_entropy': self._compute_action_entropy(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
        }

        # Reward breakdown averages
        if self._episode_reward_breakdown:
            keys = self._episode_reward_breakdown[0].keys()
            for key in keys:
                if key not in ('immobile_count', 'x_pos'):
                    vals = [b[key] for b in self._episode_reward_breakdown]
                    summary[f'reward_{key}'] = np.mean(vals)
            # Final x_pos
            summary['final_x_pos'] = self._episode_reward_breakdown[-1].get('x_pos', 0.0)

        return summary

    def _compute_action_entropy(self) -> float:
        """Compute empirical entropy of actions taken in this episode."""
        if not self._episode_actions:
            return 0.0
        counts = np.bincount(self._episode_actions, minlength=self.n_actions)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def set_temperature(self, epoch: int):
        """Anneal softmax temperature based on epoch.

        Linear annealing from temperature_start to temperature_end.
        """
        progress = min(1.0, epoch / max(1, self.temperature_anneal_epochs))
        temp = self.temperature_start + (self.temperature_end - self.temperature_start) * progress
        self.basal_ganglia.temperature = temp

    def get_temperature(self) -> float:
        return self.basal_ganglia.temperature

    def save(self, path_prefix: str):
        """Save all module weights."""
        self.visual_cortex.save(f"{path_prefix}_cortex.pt")
        self.basal_ganglia.save(f"{path_prefix}_bg.pt")
        self.forward_model.save(f"{path_prefix}_fwd.pt")
        self.inverse_model.save(f"{path_prefix}_inv.pt")

    def load(self, path_prefix: str):
        """Load all module weights."""
        self.visual_cortex.load(f"{path_prefix}_cortex.pt")
        self.basal_ganglia.load(f"{path_prefix}_bg.pt")
        self.forward_model.load(f"{path_prefix}_fwd.pt")
        self.inverse_model.load(f"{path_prefix}_inv.pt")
