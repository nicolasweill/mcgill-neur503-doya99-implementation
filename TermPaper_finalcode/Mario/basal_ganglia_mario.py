"""
Basal Ganglia Mario — Actor-Critic TD(λ) with Reward Shaping

Adapted from BasalGangliaACTrace for Super Mario Bros:
  - 7 discrete actions (SIMPLE_MOVEMENT)
  - Integrated reward shaping:
      r_shaped = r_ext + α×r_icm + β×Δx_pos - γ×penalty_immobility
  - Larger networks (256→128) for visual RL
  - Temperature annealing schedule

Three-factor learning rule (Frémaux & Gerstner 2016):
    ΔW = α × δ(t) × e(t)

where:
  - δ(t): TD error (dopamine from SNc)
  - e(t): eligibility trace (γλ × e(t-1) + ∇_W log π or ∇_W V)

Reward shaping components:
  1. Extrinsic reward: from the environment (score changes, flags, etc.)
  2. ICM intrinsic reward: forward model prediction error (curiosity)
  3. Progress bonus: Δx_pos scaled by progress_weight
  4. Immobility penalty: when x_pos doesn't change for N consecutive steps,
     apply an increasing penalty to force exploration.
  5. Death penalty: large negative reward on life loss.

References:
    Sutton & Barto (2018). TD(λ) actor-critic, Ch. 13.
    Frémaux & Gerstner (2016). Three-factor rules.
    Ng, Harada & Russell (1999). Policy invariance under reward transformations.
"""

import numpy as np
import torch
import torch.nn as nn


class _CriticNet(nn.Module):
    """Striosome pathway: V(s) — larger for Mario's visual complexity."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _ActorNet(nn.Module):
    """Matrix pathway: π(a|s) with SNr/GPi competition (softmax)."""

    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


class RewardShaper:
    """Reward shaping module for Mario with immobility penalty and progress bonus.

    Parameters
    ----------
    progress_weight : float
        Bonus per pixel of x_pos advancement.
    immobility_threshold : int
        Number of consecutive steps with Δx_pos ≈ 0 before penalty kicks in.
    immobility_penalty : float
        Base penalty per step of immobility (scales linearly).
    immobility_tolerance : float
        x_pos movement below this threshold counts as immobile.
    death_penalty : float
        Penalty applied on life loss.
    icm_weight : float
        Weight of ICM intrinsic reward (passed through from ICM module).
    max_shaped_reward : float
        Clip total shaped reward to prevent instability.
    """

    def __init__(
        self,
        progress_weight: float = 0.04,
        immobility_threshold: int = 30,
        immobility_penalty: float = 0.3,
        immobility_tolerance: float = 2.0,
        death_penalty: float = 50.0,
        max_shaped_reward: float = 15.0,
    ):
        self.progress_weight = progress_weight
        self.immobility_threshold = immobility_threshold
        self.immobility_penalty = immobility_penalty
        self.immobility_tolerance = immobility_tolerance
        self.death_penalty = death_penalty
        self.max_shaped_reward = max_shaped_reward

        # Tracking state
        self._prev_x_pos = 0.0
        self._prev_life = None
        self._immobile_count = 0
        self._total_progress = 0.0
        self._max_x_pos = 0.0  # Penalize going backwards too

    def reset(self):
        """Reset at episode start."""
        self._prev_x_pos = 0.0
        self._prev_life = None
        self._immobile_count = 0
        self._total_progress = 0.0
        self._max_x_pos = 0.0

    def shape_reward(
        self,
        extrinsic_reward: float,
        intrinsic_reward: float,
        info: dict,
        done: bool,
    ) -> tuple[float, dict]:
        """Compute shaped reward from environment info.

        Parameters
        ----------
        extrinsic_reward : float
            Raw reward from the environment.
        intrinsic_reward : float
            ICM curiosity reward.
        info : dict
            Mario environment info containing 'x_pos', 'life', etc.
        done : bool
            Whether episode terminated.

        Returns
        -------
        shaped_reward : float
        breakdown : dict
            Components for logging.
        """
        x_pos = info.get('x_pos', 0.0)
        life = info.get('life', None)

        # ── Progress bonus ──
        delta_x = x_pos - self._prev_x_pos
        progress_bonus = self.progress_weight * max(0.0, delta_x)

        # Extra bonus for reaching new max position
        if x_pos > self._max_x_pos:
            progress_bonus += self.progress_weight * 0.5 * (x_pos - self._max_x_pos)
            self._max_x_pos = x_pos

        self._total_progress += max(0.0, delta_x)

        # ── Immobility penalty ──
        if abs(delta_x) < self.immobility_tolerance:
            self._immobile_count += 1
        else:
            self._immobile_count = 0

        immobility_pen = 0.0
        if self._immobile_count > self.immobility_threshold:
            # Linearly increasing penalty
            excess = self._immobile_count - self.immobility_threshold
            immobility_pen = self.immobility_penalty * min(excess, 50) / 10.0

        # ── Death penalty ──
        death_pen = 0.0
        if self._prev_life is not None and life is not None:
            if life < self._prev_life:
                death_pen = self.death_penalty

        # ── Backward penalty ──
        backward_pen = 0.0
        if delta_x < -self.immobility_tolerance:
            backward_pen = self.progress_weight * abs(delta_x) * 0.5

        # ── Combine ──
        shaped = (
            extrinsic_reward
            + intrinsic_reward
            + progress_bonus
            - immobility_pen
            - death_pen
            - backward_pen
        )

        # Clip to prevent extreme values
        shaped = float(np.clip(shaped, -self.max_shaped_reward, self.max_shaped_reward))

        # Update state
        self._prev_x_pos = x_pos
        if life is not None:
            self._prev_life = life

        breakdown = {
            'extrinsic': extrinsic_reward,
            'intrinsic': intrinsic_reward,
            'progress': progress_bonus,
            'immobility': -immobility_pen,
            'death': -death_pen,
            'backward': -backward_pen,
            'shaped_total': shaped,
            'immobile_count': self._immobile_count,
            'x_pos': x_pos,
        }

        return shaped, breakdown


class BasalGangliaMario:
    """Actor-Critic TD(λ) for Mario with integrated reward shaping.

    Parameters
    ----------
    state_dim : int
        Feature dimensionality from visual cortex.
    n_actions : int
        Number of discrete actions (7 for SIMPLE_MOVEMENT).
    gamma : float
        Discount factor.
    lr_critic : float
        Critic learning rate.
    lr_actor : float
        Actor learning rate.
    decay_lambda : float
        Eligibility trace decay.
    entropy_beta : float
        Entropy regularization coefficient.
    temperature : float
        Softmax temperature (annealed during training).
    delta_clip : float
        Clip |δ| before normalization.
    max_weight : float
        Hard weight bound.
    trace_clip : float
        Hard trace bound.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        state_dim: int = 256,
        n_actions: int = 7,
        gamma: float = 0.9,
        lr_critic: float = 1e-3,
        lr_actor: float = 5e-4,
        decay_lambda: float = 0.92,
        entropy_beta: float = 0.02,
        temperature: float = 2.0,
        delta_clip: float = 10.0,
        max_weight: float = 5.0,
        trace_clip: float = 5.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.decay_lambda = decay_lambda
        self.entropy_beta = entropy_beta
        self.temperature = temperature
        self.delta_clip = delta_clip
        self.max_weight = max_weight
        self.trace_clip = trace_clip
        self.device = torch.device(device)

        self.critic = _CriticNet(state_dim).to(self.device)
        self.actor = _ActorNet(state_dim, n_actions).to(self.device)

        # Eligibility traces
        self._critic_traces = [
            torch.zeros_like(p) for p in self.critic.parameters()
        ]
        self._actor_traces = [
            torch.zeros_like(p) for p in self.actor.parameters()
        ]

        # Welford running normalization
        self._adv_count = 0
        self._adv_mean = 0.0
        self._adv_M2 = 0.0

        # Episode state
        self._prev_features = None
        self._prev_value = None
        self._prev_action = None

        # Reward shaper
        self.reward_shaper = RewardShaper()

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(features).float().unsqueeze(0).to(self.device)

    def _welford_update(self, delta: float) -> float:
        self._adv_count += 1
        d = delta - self._adv_mean
        self._adv_mean += d / self._adv_count
        d2 = delta - self._adv_mean
        self._adv_M2 += d * d2

        if self._adv_count < 50:  # More warmup for Mario's noisy rewards
            return delta

        variance = self._adv_M2 / (self._adv_count - 1)
        std = (variance ** 0.5) + 1e-8
        return (delta - self._adv_mean) / std

    def _accumulate_actor_trace(self, features: np.ndarray, action: int):
        t = self._to_tensor(features)
        self.actor.zero_grad()
        logits = self.actor(t)
        temp = max(self.temperature, 0.01)
        probs = torch.softmax(logits / temp, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_t = torch.tensor([action], device=self.device)
        log_prob = dist.log_prob(action_t)
        log_prob.backward()

        with torch.no_grad():
            for trace, param in zip(self._actor_traces, self.actor.parameters()):
                if param.grad is not None:
                    trace.mul_(self.gamma * self.decay_lambda)
                    trace.add_(param.grad.clone())
                    trace.clamp_(-self.trace_clip, self.trace_clip)

    def _accumulate_critic_trace(self, features: np.ndarray):
        t = self._to_tensor(features)
        self.critic.zero_grad()
        v = self.critic(t)
        v.backward()

        with torch.no_grad():
            for trace, param in zip(self._critic_traces, self.critic.parameters()):
                if param.grad is not None:
                    trace.mul_(self.gamma * self.decay_lambda)
                    trace.add_(param.grad.clone())
                    trace.clamp_(-self.trace_clip, self.trace_clip)

    def step(self, features: np.ndarray, explore: bool = True) -> int:
        """Select action and store state/value for learning."""
        t = self._to_tensor(features)

        with torch.no_grad():
            v = self.critic(t)
            logits = self.actor(t)

        temp = max(self.temperature, 0.01)
        probs = torch.softmax(logits / temp, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if explore:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        self._prev_features = features.copy()
        self._prev_value = v.item()
        self._prev_action = action.item()

        return action.item()

    def learn(
        self,
        reward: float,
        next_features: np.ndarray,
        done: bool = False,
    ) -> float:
        """Three-factor weight update: ΔW = α × δ_norm × e_trace.

        Parameters
        ----------
        reward : float
            Shaped reward (already includes ICM, progress, penalties).
        next_features : np.ndarray
            Features of the next state.
        done : bool
            Whether episode terminated.

        Returns
        -------
        delta : float
            Raw TD error (dopamine signal).
        """
        if self._prev_features is None:
            return 0.0

        with torch.no_grad():
            next_t = self._to_tensor(next_features)
            next_value = 0.0 if done else self.critic(next_t).item()

        delta = reward + self.gamma * next_value - self._prev_value
        delta_clipped = float(np.clip(delta, -self.delta_clip, self.delta_clip))
        delta_normed = self._welford_update(delta_clipped)

        self._accumulate_actor_trace(self._prev_features, self._prev_action)
        self._accumulate_critic_trace(self._prev_features)

        with torch.no_grad():
            for trace, param in zip(self._critic_traces, self.critic.parameters()):
                param.data.add_(self.lr_critic * delta_normed * trace)
                param.data.clamp_(-self.max_weight, self.max_weight)

            for trace, param in zip(self._actor_traces, self.actor.parameters()):
                param.data.add_(self.lr_actor * delta_normed * trace)
                param.data.clamp_(-self.max_weight, self.max_weight)

        # Entropy bonus
        t_prev = self._to_tensor(self._prev_features)
        self.actor.zero_grad()
        logits = self.actor(t_prev)
        temp = max(self.temperature, 0.01)
        probs = torch.softmax(logits / temp, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        entropy.backward()

        with torch.no_grad():
            for param in self.actor.parameters():
                if param.grad is not None:
                    param.data.add_(self.lr_actor * self.entropy_beta * param.grad)
                    param.data.clamp_(-self.max_weight, self.max_weight)

        return float(delta)

    def value(self, features: np.ndarray) -> float:
        with torch.no_grad():
            t = self._to_tensor(features)
            return self.critic(t).item()

    def policy(self, features: np.ndarray, explore: bool = True) -> int:
        with torch.no_grad():
            t = self._to_tensor(features)
            logits = self.actor(t)

        if explore:
            temp = max(self.temperature, 0.01)
            probs = torch.softmax(logits / temp, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()
        else:
            return logits.argmax(dim=-1).item()

    def reset(self):
        """Reset episode-specific state and traces."""
        self._prev_features = None
        self._prev_value = None
        self._prev_action = None
        self.reward_shaper.reset()

        for trace in self._critic_traces:
            trace.zero_()
        for trace in self._actor_traces:
            trace.zero_()

    def save(self, path: str):
        torch.save({
            'critic': self.critic.state_dict(),
            'actor': self.actor.state_dict(),
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'adv_count': self._adv_count,
            'adv_mean': self._adv_mean,
            'adv_M2': self._adv_M2,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.critic.load_state_dict(ckpt['critic'])
        self.actor.load_state_dict(ckpt['actor'])
        self._adv_count = ckpt.get('adv_count', 0)
        self._adv_mean = ckpt.get('adv_mean', 0.0)
        self._adv_M2 = ckpt.get('adv_M2', 0.0)
