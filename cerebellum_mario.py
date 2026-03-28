"""
Cerebellum Module — Supervised Learning on Cortical Features

Adapts Doya (1999, 2000) cerebellar architecture for image-based tasks
operating on cortical feature vectors rather than raw pixel observations.

Physiological basis for operating on cortical features:
    The cerebellum receives mossy fiber input that has already been
    processed by the cerebral cortex. The pontine nuclei relay cortical
    output to the cerebellar granule cells (Doya 1999, Section 3.1).
    Thus the cerebellar forward/inverse models naturally operate on
    the cortical representation, not raw sensory input.

Architecture preserved from the original:
    - Granule cells: fixed random expansion (Doya Sec. 3.1.3)
    - Purkinje cells: linear readout
    - Learning: climbing fiber error signal (Eq. 5)
    - ForwardModel: predicts features(t+1) from (features(t), action)
    - InverseModel: computes action from (features(t), features_target)

Key adaptations for discrete actions:
    - Actions encoded as one-hot vectors (7 or 9 dimensions)
    - InverseModel outputs action logits (softmax for probabilities)
    - CerebellarCorrector evaluates all candidate actions exhaustively
      (Section 4.2.1, Fig. 7) instead of Jacobian finite differences
"""

import numpy as np
from cerebellum import Cerebellum


class ForwardModelMario(Cerebellum):
    """Cerebellar forward model on cortical features.

    Predicts features(t+1) = F(features(t), action(t))  (Eq. 21)

    Input: concat(features, one_hot_action) — mossy fiber afferents
    Output: predicted next features — deep cerebellar nuclei output
    """

    def __init__(self, feature_dim: int = 64, n_actions: int = 7, **kwargs):
        kwargs.setdefault("n_granule", 512)
        kwargs.setdefault("lr", 0.005)
        kwargs.setdefault("grad_clip", 2.0)
        super().__init__(
            input_dim=feature_dim + n_actions,
            output_dim=feature_dim,
            **kwargs,
        )
        self.feature_dim = feature_dim
        self.n_actions = n_actions

    def _action_to_onehot(self, action: int) -> np.ndarray:
        oh = np.zeros(self.n_actions)
        oh[action] = 1.0
        return oh

    def predict_next_features(
        self, features: np.ndarray, action: int
    ) -> np.ndarray:
        """Predict next cortical features given current features and action."""
        oh = self._action_to_onehot(action)
        x = np.concatenate([features, oh])
        return self.predict(x)

    def learn_transition(
        self, features: np.ndarray, action: int, next_features: np.ndarray
    ) -> float:
        """Learn from observed feature transition.

        The climbing fiber error is the difference between actual and
        predicted next features.
        """
        oh = self._action_to_onehot(action)
        x = np.concatenate([features, oh])
        _, error = self.learn(x, next_features)
        return error


class InverseModelMario(Cerebellum):
    """Cerebellar inverse model: computes action from features.

    Maps (features_current, features_target) -> action logits (Fig. 11)

    Used for encapsulating the BG's learned policy into a fast
    cerebellar reactive mapping (Doya 1999, Section 4.3.3).
    """

    def __init__(self, feature_dim: int = 64, n_actions: int = 7, **kwargs):
        kwargs.setdefault("n_granule", 512)
        kwargs.setdefault("lr", 0.005)
        kwargs.setdefault("grad_clip", 2.0)
        super().__init__(
            input_dim=feature_dim * 2,
            output_dim=n_actions,
            **kwargs,
        )
        self.feature_dim = feature_dim
        self.n_actions = n_actions

    def compute_action(
        self, features: np.ndarray, target_features: np.ndarray
    ) -> int:
        """Compute action to move from features toward target_features.

        Returns the argmax of the output logits (greedy selection).
        """
        x = np.concatenate([features, target_features])
        logits = self.predict(x)
        return int(np.argmax(logits))

    def compute_action_probs(
        self, features: np.ndarray, target_features: np.ndarray
    ) -> np.ndarray:
        """Compute action probability distribution (softmax of logits)."""
        x = np.concatenate([features, target_features])
        logits = self.predict(x)
        # Stable softmax
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)

    def learn_action(
        self,
        features: np.ndarray,
        target_features: np.ndarray,
        correct_action: int,
    ) -> float:
        """Learn from the action selected by the BG (encapsulation).

        Teacher signal: one-hot encoding of the BG's chosen action.
        This implements Fig. 11: the cerebellum learns to replicate
        the cortical feedback pathway's input-output mapping.
        """
        x = np.concatenate([features, target_features])
        target = np.zeros(self.n_actions)
        target[correct_action] = 1.0
        _, error = self.learn(x, target)
        return error


class CerebellarCorrectorMario:
    """Online correction via exhaustive action evaluation (Fig. 7).

    For discrete action spaces, the Jacobian-based correction from
    the original CerebellarCorrector is replaced by exhaustive
    evaluation: for each of the N possible actions, the forward model
    predicts the resulting features, and the action whose predicted
    outcome is closest to the desired state is selected.

    This is actually more faithful to Doya's Section 4.2.1 (Fig. 7):
    "consider a candidate action u*(t) one at a time, predict the
    resulting future state x*(t+1) and its value V(x*(t+1)), and
    accept it for execution if it is good enough."
    """

    def __init__(self, forward_model: ForwardModelMario):
        self.forward_model = forward_model
        self.n_actions = forward_model.n_actions
        self._prev_features = None
        self._prev_action = None
        self._prediction_error_history = []

    def begin_step(self, features: np.ndarray, action: int):
        """Record features and action before environment step."""
        self._prev_features = features.copy()
        self._prev_action = action

    def correct(
        self,
        observed_features: np.ndarray,
        base_action: int,
        desired_features: np.ndarray | None = None,
    ) -> tuple[int, float]:
        """Select best action using forward model predictions.

        If desired_features is provided, pick the action whose predicted
        next features are closest to it. Otherwise, just return base_action
        and the prediction error.

        Returns
        -------
        best_action : int
        prediction_error : float
        """
        prediction_error = 0.0

        if self._prev_features is not None and self._prev_action is not None:
            predicted = self.forward_model.predict_next_features(
                self._prev_features, self._prev_action
            )
            prediction_error = float(np.linalg.norm(observed_features - predicted))
            self._prediction_error_history.append(prediction_error)

        if desired_features is None:
            return base_action, prediction_error

        # Exhaustive evaluation: try all actions, pick best
        best_action = base_action
        best_dist = float("inf")

        for a in range(self.n_actions):
            pred = self.forward_model.predict_next_features(observed_features, a)
            dist = float(np.linalg.norm(pred - desired_features))
            if dist < best_dist:
                best_dist = dist
                best_action = a

        return best_action, prediction_error

    def reset(self):
        self._prev_features = None
        self._prev_action = None
        self._prediction_error_history = []

    @property
    def mean_prediction_error(self) -> float:
        if not self._prediction_error_history:
            return 0.0
        return float(np.mean(self._prediction_error_history))


class HybridControllerMario:
    """Automatic switching between cerebellum and BG for discrete actions.

    Same surprise-detection logic as HybridController:
    if forward model prediction error exceeds adaptive threshold,
    fall back to the BG for deliberate action selection.
    """

    def __init__(
        self,
        forward_model: ForwardModelMario,
        inverse_model: InverseModelMario,
        corrector: CerebellarCorrectorMario,
        surprise_threshold_factor: float = 2.0,
        ema_alpha: float = 0.05,
    ):
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.corrector = corrector
        self.surprise_factor = surprise_threshold_factor
        self.ema_alpha = ema_alpha

        self._ema_mean = 0.0
        self._ema_var = 0.01
        self._n_updates = 0
        self._cerebellum_steps = 0
        self._bg_steps = 0

    def _update_stats(self, prediction_error: float):
        self._n_updates += 1
        if self._n_updates == 1:
            self._ema_mean = prediction_error
            self._ema_var = prediction_error ** 2
        else:
            self._ema_mean = (
                (1 - self.ema_alpha) * self._ema_mean
                + self.ema_alpha * prediction_error
            )
            self._ema_var = (
                (1 - self.ema_alpha) * self._ema_var
                + self.ema_alpha * (prediction_error - self._ema_mean) ** 2
            )

    @property
    def surprise_threshold(self) -> float:
        return self._ema_mean + self.surprise_factor * np.sqrt(
            max(self._ema_var, 1e-8)
        )

    def select_action(
        self,
        features: np.ndarray,
        target_features: np.ndarray,
        basal_ganglia,
        observed_features: np.ndarray | None = None,
    ) -> tuple[int, str, float]:
        """Select action using cerebellum or BG based on confidence.

        Returns
        -------
        action : int
        source : str ("cerebellum" or "basal_ganglia")
        prediction_error : float
        """
        prediction_error = 0.0
        is_surprised = False

        if observed_features is not None and self.corrector._prev_features is not None:
            predicted = self.forward_model.predict_next_features(
                self.corrector._prev_features, self.corrector._prev_action
            )
            prediction_error = float(np.linalg.norm(observed_features - predicted))
            self._update_stats(prediction_error)

            if self._n_updates > 20:
                is_surprised = prediction_error > self.surprise_threshold

        if is_surprised:
            action = basal_ganglia.policy(features, explore=False)
            self._bg_steps += 1
            source = "basal_ganglia"
        else:
            action = self.inverse_model.compute_action(features, target_features)
            self._cerebellum_steps += 1
            source = "cerebellum"

        return action, source, prediction_error

    def reset(self):
        self.corrector.reset()

    @property
    def cerebellum_ratio(self) -> float:
        total = self._cerebellum_steps + self._bg_steps
        return self._cerebellum_steps / total if total > 0 else 0.0

    def reset_counters(self):
        self._cerebellum_steps = 0
        self._bg_steps = 0
