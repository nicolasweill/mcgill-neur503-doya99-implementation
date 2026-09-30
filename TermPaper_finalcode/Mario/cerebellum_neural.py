"""
Cerebellum Neural — PyTorch Forward/Inverse Models with Dopamine Gating

Replaces the granule-cell random expansion of cerebellum_3f.py with learned
MLP networks trained via backpropagation, while preserving the dopamine-gated
learning rate from Cerebellum3F.

Forward Model (Eq. 21 Doya 1999):
    features(t+1) = F(features(t), action(t))
    Prediction error = ||features_actual - features_predicted||²
    → Used as ICM intrinsic reward (curiosity-driven exploration)

Inverse Model (Fig. 11 Doya 1999):
    action = F⁻¹(features(t), features(t+1))
    → Learns sensory-motor mapping; encapsulates learned skills
    → Also used in ICM: trains feature space to be action-relevant

Dopamine gating (from cerebellum_3f.py):
    η_eff = η_base × gate(δ)
    gate(δ) = 1 + scale × (sigmoid(δ_centered) - 0.5)

ICM (Intrinsic Curiosity Module, Pathak et al. 2017):
    r_intrinsic = η_icm × ||F(φ(s_t), a_t) - φ(s_{t+1})||²
    The inverse model loss ensures the feature space captures
    action-relevant information (not noise/stochasticity).

References:
    Pathak et al. (2017). Curiosity-driven Exploration by Self-Supervised
        Prediction. ICML.
    Doya (1999, 2000). What are the computations of the cerebellum, the
        basal ganglia and the cerebral cortex?
    Kostadinov et al. (2019). Predictive and reactive reward signals in
        climbing fiber inputs. Nature Neuroscience.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModelNeural(nn.Module):
    """Neural forward model: predicts next features from current features + action.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the feature vector from the visual cortex.
    n_actions : int
        Number of discrete actions (one-hot encoded).
    hidden_dim : int
        Hidden layer width.
    lr : float
        Base learning rate.
    da_lr_scale : float
        Dopamine modulation amplitude for learning rate.
    da_baseline_ema : float
        EMA coefficient for δ baseline centering.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_actions: int = 7,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        da_lr_scale: float = 2.0,
        da_baseline_ema: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.base_lr = lr
        self.da_lr_scale = da_lr_scale
        self.da_baseline_ema = da_baseline_ema
        self.device = torch.device(device)

        self._da_ema = 0.0

        # MLP: (features + one-hot action) → predicted next features
        input_dim = feature_dim + n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, features: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Predict next features.

        Parameters
        ----------
        features : torch.Tensor, shape (batch, feature_dim)
        action_onehot : torch.Tensor, shape (batch, n_actions)

        Returns
        -------
        predicted_next : torch.Tensor, shape (batch, feature_dim)
        """
        x = torch.cat([features, action_onehot], dim=-1)
        return self.net(x)

    def _dopamine_gate(self, delta: float) -> float:
        self._da_ema = (
            (1.0 - self.da_baseline_ema) * self._da_ema
            + self.da_baseline_ema * delta
        )
        delta_centered = delta - self._da_ema
        sig = 1.0 / (1.0 + np.exp(-np.clip(delta_centered, -10, 10)))
        gate = 1.0 + self.da_lr_scale * (sig - 0.5)
        return max(0.0, gate)

    def predict_next_features(
        self, features: np.ndarray, action: int
    ) -> np.ndarray:
        """Predict next features (no grad, inference only)."""
        with torch.no_grad():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            act_oh = torch.zeros(1, self.n_actions, device=self.device)
            act_oh[0, action] = 1.0
            pred = self.forward(feat_t, act_oh)
        return pred.cpu().numpy().flatten()

    def learn_transition(
        self,
        features: np.ndarray,
        action: int,
        next_features: np.ndarray,
        delta: float | None = None,
    ) -> float:
        """Learn from observed transition with dopamine-gated lr.

        Parameters
        ----------
        features : np.ndarray, shape (feature_dim,)
        action : int
        next_features : np.ndarray, shape (feature_dim,)
        delta : float or None
            TD error from BG for dopamine gating.

        Returns
        -------
        prediction_error : float
            MSE before update (used as ICM intrinsic reward).
        """
        # Set effective lr
        if delta is not None:
            gate = self._dopamine_gate(delta)
            effective_lr = self.base_lr * gate
        else:
            effective_lr = self.base_lr

        for pg in self.optimizer.param_groups:
            pg['lr'] = effective_lr

        feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        act_oh = torch.zeros(1, self.n_actions, device=self.device)
        act_oh[0, action] = 1.0
        target = torch.from_numpy(next_features).float().unsqueeze(0).to(self.device)

        predicted = self.forward(feat_t, act_oh)
        loss = F.mse_loss(predicted, target.detach())

        # Capture prediction error BEFORE update
        prediction_error = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return prediction_error

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'da_ema': self._da_ema,
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt['state_dict'])
        self._da_ema = ckpt.get('da_ema', 0.0)
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])


class InverseModelNeural(nn.Module):
    """Neural inverse model: predicts action from consecutive features.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the feature vector.
    n_actions : int
        Number of discrete actions.
    hidden_dim : int
        Hidden layer width.
    lr : float
        Base learning rate.
    da_lr_scale : float
        Dopamine modulation amplitude.
    da_baseline_ema : float
        EMA coefficient for δ baseline.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_actions: int = 7,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        da_lr_scale: float = 2.0,
        da_baseline_ema: float = 0.05,
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.base_lr = lr
        self.da_lr_scale = da_lr_scale
        self.da_baseline_ema = da_baseline_ema
        self.device = torch.device(device)

        self._da_ema = 0.0

        # MLP: (features_t, features_t+1) → action logits
        input_dim = feature_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, features: torch.Tensor, next_features: torch.Tensor) -> torch.Tensor:
        """Predict action logits from feature transition.

        Parameters
        ----------
        features : torch.Tensor, shape (batch, feature_dim)
        next_features : torch.Tensor, shape (batch, feature_dim)

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_actions)
        """
        x = torch.cat([features, next_features], dim=-1)
        return self.net(x)

    def _dopamine_gate(self, delta: float) -> float:
        self._da_ema = (
            (1.0 - self.da_baseline_ema) * self._da_ema
            + self.da_baseline_ema * delta
        )
        delta_centered = delta - self._da_ema
        sig = 1.0 / (1.0 + np.exp(-np.clip(delta_centered, -10, 10)))
        gate = 1.0 + self.da_lr_scale * (sig - 0.5)
        return max(0.0, gate)

    def compute_action(
        self, features: np.ndarray, next_features: np.ndarray
    ) -> int:
        """Predict action (no grad, inference only)."""
        with torch.no_grad():
            f_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            f_t1 = torch.from_numpy(next_features).float().unsqueeze(0).to(self.device)
            logits = self.forward(f_t, f_t1)
        return logits.argmax(dim=-1).item()

    def learn_action(
        self,
        features: np.ndarray,
        next_features: np.ndarray,
        correct_action: int,
        delta: float | None = None,
    ) -> float:
        """Learn from observed transition with dopamine-gated lr.

        Parameters
        ----------
        features : np.ndarray, shape (feature_dim,)
        next_features : np.ndarray, shape (feature_dim,)
        correct_action : int
            The action that was actually taken.
        delta : float or None
            TD error for dopamine gating.

        Returns
        -------
        loss : float
            Cross-entropy loss before update.
        """
        if delta is not None:
            gate = self._dopamine_gate(delta)
            effective_lr = self.base_lr * gate
        else:
            effective_lr = self.base_lr

        for pg in self.optimizer.param_groups:
            pg['lr'] = effective_lr

        f_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        f_t1 = torch.from_numpy(next_features).float().unsqueeze(0).to(self.device)
        target = torch.tensor([correct_action], dtype=torch.long, device=self.device)

        logits = self.forward(f_t, f_t1)
        loss = F.cross_entropy(logits, target)

        loss_val = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss_val

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'da_ema': self._da_ema,
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt['state_dict'])
        self._da_ema = ckpt.get('da_ema', 0.0)
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])


class ICMModule:
    """Intrinsic Curiosity Module combining forward and inverse models.

    Computes intrinsic reward as forward model prediction error,
    while the inverse model regularizes the feature space to only
    encode action-relevant information.

    Parameters
    ----------
    forward_model : ForwardModelNeural
    inverse_model : InverseModelNeural
    icm_weight : float
        Scaling factor for intrinsic reward.
    inverse_weight : float
        Weight of inverse model loss relative to forward model loss.
    """

    def __init__(
        self,
        forward_model: ForwardModelNeural,
        inverse_model: InverseModelNeural,
        icm_weight: float = 0.1,
        inverse_weight: float = 0.2,
    ):
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.icm_weight = icm_weight
        self.inverse_weight = inverse_weight

        # Running normalization of intrinsic reward
        self._reward_ema = 0.0
        self._reward_var_ema = 1.0
        self._n_updates = 0

    def compute_intrinsic_reward(
        self,
        features: np.ndarray,
        action: int,
        next_features: np.ndarray,
        delta: float | None = None,
    ) -> tuple[float, float, float]:
        """Compute intrinsic reward and train both models.

        Parameters
        ----------
        features : np.ndarray, shape (feature_dim,)
        action : int
        next_features : np.ndarray, shape (feature_dim,)
        delta : float or None
            TD error for dopamine gating of cerebellar learning.

        Returns
        -------
        intrinsic_reward : float
            Normalized forward model prediction error.
        forward_loss : float
            Raw forward model MSE.
        inverse_loss : float
            Raw inverse model cross-entropy.
        """
        # Forward model: learn and get prediction error
        forward_loss = self.forward_model.learn_transition(
            features, action, next_features, delta=delta
        )

        # Inverse model: learn action prediction
        inverse_loss = self.inverse_model.learn_action(
            features, next_features, action, delta=delta
        )

        # Intrinsic reward = normalized forward prediction error
        raw_reward = forward_loss

        # Running normalization
        self._n_updates += 1
        alpha = max(0.01, 1.0 / self._n_updates)
        self._reward_ema = (1 - alpha) * self._reward_ema + alpha * raw_reward
        self._reward_var_ema = (
            (1 - alpha) * self._reward_var_ema
            + alpha * (raw_reward - self._reward_ema) ** 2
        )
        std = max(np.sqrt(self._reward_var_ema), 1e-8)
        normalized = (raw_reward - self._reward_ema) / std

        # Clip and scale
        intrinsic_reward = self.icm_weight * float(np.clip(normalized, -3.0, 3.0))

        return intrinsic_reward, forward_loss, inverse_loss
