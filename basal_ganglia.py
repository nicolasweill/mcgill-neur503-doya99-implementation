"""
Basal Ganglia Module — Reinforcement Learning

Based on Doya (1999, 2000): the basal ganglia are specialized for
reinforcement learning, guided by the reward signal encoded in
dopaminergic projections from the substantia nigra.

Architecture (Fig. 2 of Doya 2000, Fig. 6 of Doya 1999):

    Striosome → Critic : predicts state value V(x)
    Matrix    → Actor  : selects actions via policy G(x)
    SNc       → TD error δ(t) = r(t) + γV(t) - V(t-1)

The TD error δ serves a dual role (Doya 1999, Section 2.2):
  1. Error signal for value prediction learning   (Eq. 12)
  2. Reinforcement signal for policy improvement   (Eq. 14)

Dopamine modulates cortico-striatal synaptic plasticity:
  - Positive δ → LTP (reward better than expected)
  - Negative δ → LTD (reward worse than expected)
"""

import numpy as np


class BasalGanglia:
    """Actor-Critic reinforcement learning module.

    The striosome compartment implements the Critic (value function),
    and the matrix compartment implements the Actor (stochastic policy).
    The TD error is computed in the SNc dopamine neurons.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the cortical state representation.
    action_dim : int
        Number of continuous action dimensions.
    gamma : float
        Discount factor for future rewards (0 <= gamma <= 1).
    lr_critic : float
        Learning rate for the value function (striosome).
    lr_actor : float
        Learning rate for the policy (matrix).
    noise_std : float
        Standard deviation of exploration noise (temperature).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.95,
        lr_critic: float = 0.01,
        lr_actor: float = 0.005,
        noise_std: float = 0.3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.noise_std = noise_std

        # ------------------------------------------------------------------
        # Striosome (Critic): predicts future reward V(x) = sum v_j * x_j
        # Linear value function (Eq. 11 of Doya 1999)
        # ------------------------------------------------------------------
        self.v_weights = np.zeros(state_dim)
        self.v_bias = 0.0

        # ------------------------------------------------------------------
        # Matrix (Actor): policy u(x) = g(sum w_ij * x_j + noise)
        # Stochastic policy (Eq. 13 of Doya 1999)
        # ------------------------------------------------------------------
        self.w_weights = np.random.randn(action_dim, state_dim) * 0.01
        self.w_bias = np.zeros(action_dim)

        # Stored for learning
        self._prev_state = None
        self._prev_value = 0.0
        self._prev_action = None
        self._prev_action_mean = None

    def value(self, state: np.ndarray) -> float:
        """Compute state value V(x) — the striosome output.

        Implements Eq. 11 of Doya (1999): V(t) = sum v_j * x_j(t)

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
            Cortical state representation.

        Returns
        -------
        float
            Estimated cumulative future reward.
        """
        return float(self.v_weights @ state + self.v_bias)

    def policy(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select an action — the matrix output through SNr/GP.

        Implements Eq. 13 of Doya (1999):
            u_i(t) = g(sum w_ij * x_j(t) + noise)

        The competition among matrix outputs in SNr/GP selects the
        action with the highest expected future reward.

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
            Cortical state representation.
        explore : bool
            If True, add Gaussian noise for exploration.

        Returns
        -------
        action : np.ndarray, shape (action_dim,)
            Selected action, clipped to [-1, 1].
        """
        mean = self.w_weights @ state + self.w_bias

        if explore:
            noise = np.random.randn(self.action_dim) * self.noise_std
            action = mean + noise
        else:
            action = mean.copy()

        self._prev_action_mean = mean
        return np.clip(action, -1.0, 1.0)

    def td_error(self, reward: float, state: np.ndarray) -> float:
        """Compute the temporal difference error — the SNc dopamine signal.

        Implements Eq. 10 of Doya (1999):
            delta(t) = r(t) + gamma * V(x(t)) - V(x(t-1))

        This signal encodes the discrepancy between the predicted and
        the actual reward, exactly matching the response properties of
        midbrain dopamine neurons (Schultz et al., 1997).

        Parameters
        ----------
        reward : float
            Immediate reward r(t).
        state : np.ndarray, shape (state_dim,)
            Current state x(t).

        Returns
        -------
        delta : float
            TD error (dopamine signal).
        """
        current_value = self.value(state)
        delta = reward + self.gamma * current_value - self._prev_value
        return delta

    def step(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Observe state and select action (beginning of a time step).

        Parameters
        ----------
        state : np.ndarray, shape (state_dim,)
        explore : bool

        Returns
        -------
        action : np.ndarray, shape (action_dim,)
        """
        action = self.policy(state, explore=explore)
        self._prev_state = state.copy()
        self._prev_value = self.value(state)
        self._prev_action = action.copy()
        return action

    def learn(self, reward: float, next_state: np.ndarray, done: bool = False):
        """Update critic and actor using the dopaminergic TD signal.

        Critic update (Eq. 12 of Doya 1999):
            Delta_v_j = delta(t) * x_j(t-1)

        Actor update (Eq. 14 of Doya 1999):
            Delta_w_ij = delta(t) * (u_i(t-1) - u_bar_i) * x_j(t-1)

        Parameters
        ----------
        reward : float
            Reward obtained after taking the previous action.
        next_state : np.ndarray, shape (state_dim,)
            State reached after the action.
        done : bool
            Whether the episode has terminated.

        Returns
        -------
        delta : float
            The TD error (dopamine signal magnitude).
        """
        if self._prev_state is None:
            return 0.0

        # --- SNc: compute TD error (dopamine signal) ---
        next_value = 0.0 if done else self.value(next_state)
        delta = reward + self.gamma * next_value - self._prev_value

        # --- Striosome: update critic (Eq. 12) ---
        self.v_weights += self.lr_critic * delta * self._prev_state
        self.v_bias += self.lr_critic * delta

        # --- Matrix: update actor (Eq. 14) ---
        # (u - u_bar) approximated by (action - mean_action), i.e. the noise
        action_deviation = self._prev_action - self._prev_action_mean
        self.w_weights += self.lr_actor * delta * np.outer(
            action_deviation, self._prev_state
        )
        self.w_bias += self.lr_actor * delta * action_deviation

        return float(delta)

    def learn_terminal(self, reward: float):
        """Learn from a terminal state (no next state)."""
        return self.learn(reward, np.zeros(self.state_dim), done=True)

    def reset(self):
        """Reset episode-specific state between episodes."""
        self._prev_state = None
        self._prev_value = 0.0
        self._prev_action = None
        self._prev_action_mean = None
