
"""
Basal Ganglia with Eligibility Traces — Actor-Critic Learning

Implements actor-critic learning with eligibility traces, allowing the
network to learn from delayed rewards. Uses policy gradient methods with
proper credit assignment through trace accumulation.

Includes entropy regularization to prevent premature convergence and
normalization techniques to stabilize learning.
"""

import numpy as np
import torch
import torch.nn as nn


class _CriticNet(nn.Module):
    """Striosome pathway: V(s) with dendritic nonlinearity."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
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
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


class BasalGangliaACTrace:
    """Actor-Critic with TD(λ) REINFORCE eligibility traces.

    Biologically plausible: no backpropagation optimizer (Adam/SGD). Weights
    are updated manually via the three-factor rule ΔW = α × δ × e_trace.

    Parameters
    ----------
    state_dim : int
        Dimensionality of cortical features.
    n_actions : int
        Number of discrete actions.
    gamma : float
        Discount factor (Doya Eq. 10).
    lr_critic : float
        Learning rate for critic update.
    lr_actor : float
        Learning rate for actor update.
    decay_lambda : float
        Eligibility trace decay λ. γλ ≈ 0.9 → ~10-step credit window.
    entropy_beta : float
        Entropy regularization coefficient. Prevents premature policy collapse.
    temperature : float
        Softmax temperature (mutable — annealed by trainer).
    delta_clip : float
        Clamp |δ| before Welford normalization.
    max_weight : float
        Hard weight bound after each update.
    trace_clip : float
        Hard trace bound after each accumulation step.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        state_dim: int = 14,
        n_actions: int = 2,
        gamma: float = 0.99,
        lr_critic: float = 1e-3,
        lr_actor: float = 1e-3,
        decay_lambda: float = 0.9,
        entropy_beta: float = 0.01,
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

        # Eligibility traces — one tensor per parameter, same shape as param
        # Initialized to zero; reset at each episode boundary
        self._critic_traces = [
            torch.zeros_like(p) for p in self.critic.parameters()
        ]
        self._actor_traces = [
            torch.zeros_like(p) for p in self.actor.parameters()
        ]

        # Welford running normalization of TD advantage (NOT reset between episodes)
        self._adv_count = 0
        self._adv_mean = 0.0
        self._adv_M2 = 0.0

        # Episode state
        self._prev_features = None
        self._prev_value = None
        self._prev_action = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(features).float().unsqueeze(0).to(self.device)

    def _welford_update(self, delta: float) -> float:
        """Online Welford normalization of TD advantage."""
        self._adv_count += 1
        d = delta - self._adv_mean
        self._adv_mean += d / self._adv_count
        d2 = delta - self._adv_mean
        self._adv_M2 += d * d2

        if self._adv_count < 10:
            return delta

        variance = self._adv_M2 / (self._adv_count - 1)
        std = (variance ** 0.5) + 1e-8
        return (delta - self._adv_mean) / std

    def _accumulate_actor_trace(self, features: np.ndarray, action: int):
        """Accumulate actor eligibility traces via ∇_W log π(a|s)."""
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
        """Accumulate critic eligibility traces via ∇_W V(s)."""
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, features: np.ndarray, explore: bool = True) -> int:
        """Select action, store state/value for learning."""
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

    def learn(self, reward: float, next_features: np.ndarray, done: bool = False, action_override: int | None = None) -> float:
        """Three-factor weight update: ΔW = α × δ_normalized × e_trace."""
        if self._prev_features is None:
            return 0.0

        # SNc dopamine signal: TD error δ (Doya Eq. 10)
        with torch.no_grad():
            next_t = self._to_tensor(next_features)
            next_value = 0.0 if done else self.critic(next_t).item()

        delta = reward + self.gamma * next_value - self._prev_value
        delta_clipped = float(np.clip(delta, -self.delta_clip, self.delta_clip))
        delta_normed = self._welford_update(delta_clipped)

        # Accumulate traces for the action at time t
        actual_action = action_override if action_override is not None else self._prev_action
        self._accumulate_actor_trace(self._prev_features, actual_action)
        self._accumulate_critic_trace(self._prev_features)

        # Three-factor update: ΔW = α × δ × e
        with torch.no_grad():
            for trace, param in zip(self._critic_traces, self.critic.parameters()):
                param.data.add_(self.lr_critic * delta_normed * trace)
                param.data.clamp_(-self.max_weight, self.max_weight)

            for trace, param in zip(self._actor_traces, self.actor.parameters()):
                param.data.add_(self.lr_actor * delta_normed * trace)
                param.data.clamp_(-self.max_weight, self.max_weight)

        # Entropy bonus: β × ∇H(π) — pushes logits toward uniformity
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
        """Compute state value V(s)."""
        with torch.no_grad():
            t = self._to_tensor(features)
            return self.critic(t).item()

    def policy(self, features: np.ndarray, explore: bool = True) -> int:
        """Select action from current policy (no learning)."""
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
        """Reset episode-specific state and eligibility traces.

        Welford statistics are preserved across episodes.
        """
        self._prev_features = None
        self._prev_value = None
        self._prev_action = None

        for trace in self._critic_traces:
            trace.zero_()
        for trace in self._actor_traces:
            trace.zero_()

    def save(self, path: str):
        """Save weights and Welford statistics."""
        torch.save({
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "adv_count": self._adv_count,
            "adv_mean": self._adv_mean,
            "adv_M2": self._adv_M2,
        }, path)

    def load(self, path: str):
        """Load weights and Welford statistics."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.critic.load_state_dict(ckpt["critic"])
        self.actor.load_state_dict(ckpt["actor"])
        self._adv_count = ckpt.get("adv_count", 0)
        self._adv_mean = ckpt.get("adv_mean", 0.0)
        self._adv_M2 = ckpt.get("adv_M2", 0.0)
