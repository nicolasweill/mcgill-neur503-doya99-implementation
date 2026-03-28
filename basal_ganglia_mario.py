"""
Basal Ganglia Module — Reinforcement Learning (MLP, Discrete Actions)

Adapts Doya (1999, 2000) actor-critic architecture for high-dimensional
cortical features and discrete action spaces.

Physiological mapping:

    Component        Neural substrate     Justification
    ──────────────   ──────────────────   ──────────────────────────────────
    Critic MLP       Striosome MSNs       Dendritic trees with ~10,000 spines
                                          perform nonlinear NMDA-mediated
                                          integration (Wickens et al., 2007)
    Actor MLP        Matrix MSNs          Same dendritic nonlinearity
    ReLU             D1/D2 pathways       Direct (Go) / indirect (NoGo) gating
    Softmax          SNr/GPi competition  Mutual inhibition selects winning
                                          action channel (Eq. 13, Fig. 6)
    Temperature      Meta-parameter       Exploration control (Doya Sec. 6.5)
    TD error         SNc dopamine         δ = r + γV(s') - V(s) (Eq. 10)

The TD learning rules (Eq. 10-14) are preserved exactly:
  - Critic: minimize δ² via backprop = multi-layer generalization of Eq. 12
  - Actor: δ * log π(a|s) = REINFORCE = discrete generalization of Eq. 14
    where (u - ū) in continuous case becomes ∇log π in discrete case
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


class BasalGangliaMario:
    """MLP Actor-Critic for discrete actions on cortical features.

    Preserves the API of the original BasalGanglia:
        step(features, explore) -> action_index
        learn(reward, next_features, done) -> td_error
        value(features) -> V(s)
        policy(features, explore) -> action_index

    Parameters
    ----------
    state_dim : int
        Dimensionality of cortical features (default 64).
    n_actions : int
        Number of discrete actions.
    gamma : float
        Discount factor (Doya Eq. 10).
    lr_critic : float
        Learning rate for striosome (critic).
    lr_actor : float
        Learning rate for matrix (actor).
    temperature : float
        Softmax temperature for exploration (Doya Sec. 6.5).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 7,
        gamma: float = 0.99,
        lr_critic: float = 1e-4,
        lr_actor: float = 1e-4,
        temperature: float = 2.0,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.temperature = temperature
        self.device = torch.device(device)

        self.critic = _CriticNet(state_dim).to(self.device)
        self.actor = _ActorNet(state_dim, n_actions).to(self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Stored for TD learning
        self._prev_features = None
        self._prev_value = None
        self._prev_action = None
        self._prev_log_prob = None

    def _features_to_tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(features).float().unsqueeze(0).to(self.device)

    def value(self, features: np.ndarray) -> float:
        """Compute state value V(s) — striosome output (Eq. 11)."""
        with torch.no_grad():
            t = self._features_to_tensor(features)
            v = self.critic(t)
        return v.item()

    def policy(self, features: np.ndarray, explore: bool = True) -> int:
        """Select action — matrix output through SNr/GPi (Eq. 13).

        When explore=True, sample from softmax(logits/temperature).
        When explore=False, take argmax (greedy).
        """
        with torch.no_grad():
            t = self._features_to_tensor(features)
            logits = self.actor(t)

        if explore:
            # Temperature-scaled softmax (Doya Sec. 6.5)
            probs = torch.softmax(logits / max(self.temperature, 0.01), dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item()
        else:
            return logits.argmax(dim=-1).item()

    def step(self, features: np.ndarray, explore: bool = True) -> int:
        """Observe state, select action, store for learning.

        Returns
        -------
        action : int
            Selected discrete action index.
        """
        t = self._features_to_tensor(features)

        with torch.no_grad():
            v = self.critic(t)

        logits = self.actor(t)
        if explore:
            probs = torch.softmax(logits / max(self.temperature, 0.01), dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = logits.argmax(dim=-1)
            probs = torch.softmax(logits / max(self.temperature, 0.01), dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)

        self._prev_features = features.copy()
        self._prev_value = v.item()
        self._prev_action = action.item()
        self._prev_log_prob = log_prob

        return action.item()

    def learn(self, reward: float, next_features: np.ndarray, done: bool = False) -> float:
        """Update critic and actor using dopaminergic TD signal.

        Critic (Eq. 12 generalized): minimize δ² via backprop
        Actor (Eq. 14 generalized): δ * ∇log π(a|s)

        The TD error δ = r + γV(s') - V(s) is computed exactly as
        in Doya (1999) Eq. 10. The multi-layer backpropagation is
        the natural generalization to nonlinear function approximators.

        Returns
        -------
        delta : float
            TD error (dopamine signal magnitude).
        """
        if self._prev_features is None:
            return 0.0

        # --- SNc: compute TD error (Eq. 10) ---
        with torch.no_grad():
            next_t = self._features_to_tensor(next_features)
            next_value = 0.0 if done else self.critic(next_t).item()

        delta = reward + self.gamma * next_value - self._prev_value

        # --- Striosome: update critic (Eq. 12 generalized) ---
        prev_t = self._features_to_tensor(self._prev_features)
        pred_value = self.critic(prev_t)
        target_value = torch.tensor([reward + self.gamma * next_value],
                                     device=self.device)
        critic_loss = nn.functional.mse_loss(pred_value, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Matrix: update actor (Eq. 14 generalized) ---
        # REINFORCE: δ * log π(a|s) is the discrete generalization
        # of Doya's Δw = δ * (u - ū) * x
        if self._prev_log_prob is not None:
            actor_loss = -delta * self._prev_log_prob
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return float(delta)

    def learn_terminal(self, reward: float) -> float:
        """Learn from terminal state."""
        return self.learn(reward, np.zeros(self.state_dim), done=True)

    def reset(self):
        """Reset episode-specific state."""
        self._prev_features = None
        self._prev_value = None
        self._prev_action = None
        self._prev_log_prob = None
