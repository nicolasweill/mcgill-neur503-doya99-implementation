"""
Cerebellum 3F — Dopamine-Modulated Forward and Inverse Models

Extends the cerebellar models with reward-based learning rate modulation.
The dopamine signal from the basal ganglia controls how strongly the
cerebellum learns from prediction errors.

This implements a form of gating where unexpected outcomes lead to
stronger learning than expected ones.
"""

import numpy as np
from cerebellum import Cerebellum, ForwardModel, InverseModel


class Cerebellum3F(Cerebellum):
    """Cerebellar module with dopamine-gated LTD.

    Extends Cerebellum.learn() to accept an optional TD error δ that
    modulates the effective learning rate via a sigmoid gate. When
    delta=None, degrades to standard supervised LTD (backward compatible).

    Parameters
    ----------
    input_dim, output_dim, n_granule, lr, granule_sparsity, grad_clip :
        Identical to Cerebellum.
    da_lr_scale : float
        Maximum amplification factor of the learning rate by dopamine.
        With da_lr_scale=2.0: lr can range from 0× (strong negative δ)
        to 2× (strong positive δ) of the nominal lr.
    da_baseline_ema : float
        EMA coefficient for the running baseline of δ.
        Used to center δ so that only surprises ABOVE the expected
        baseline are treated as "positive dopamine".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_granule: int = 200,
        lr: float = 0.005,
        granule_sparsity: float = 0.3,
        grad_clip: float = 1.0,
        da_lr_scale: float = 2.0,
        da_baseline_ema: float = 0.05,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            n_granule=n_granule,
            lr=lr,
            granule_sparsity=granule_sparsity,
            grad_clip=grad_clip,
        )
        self.da_lr_scale     = da_lr_scale
        self.da_baseline_ema = da_baseline_ema
        self._da_ema         = 0.0   # running baseline of δ; never reset

    def _dopamine_gate(self, delta: float) -> float:
        """Compute the dopamine-gated learning rate multiplier.

        Maps TD error δ to a smooth, bounded gain in [1 - scale/2, 1 + scale/2].

        Parameters
        ----------
        delta : float
            TD error from the BG (SNc dopamine signal).

        Returns
        -------
        gate : float
            Effective lr multiplier.
        """
        # Center around running baseline
        self._da_ema = (
            (1.0 - self.da_baseline_ema) * self._da_ema
            + self.da_baseline_ema * delta
        )
        delta_centered = delta - self._da_ema

        # Sigmoid gate: bounded, smooth, value=1.0 at delta_centered=0
        sig = 1.0 / (1.0 + np.exp(-delta_centered))
        gate = 1.0 + self.da_lr_scale * (sig - 0.5)

        # Hard lower bound at 0 (lr never negative)
        return max(0.0, gate)

    def learn(
        self, x: np.ndarray, target: np.ndarray, delta: float | None = None
    ) -> tuple[np.ndarray, float]:
        """Supervised LTD with optional dopamine-gated learning rate.

        Implements Doya (1999) Eq. 5 with dopamine modulation:
            ΔW = η_eff · (y_target - y_pred) · gc_activation

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Mossy fiber input.
        target : np.ndarray, shape (output_dim,)
            Teacher signal (climbing fiber).
        delta : float or None
            TD error from bg.learn(). If None, standard supervised LTD.

        Returns
        -------
        prediction : np.ndarray, shape (output_dim,)
        error : float
            Squared error ||target - prediction||^2.
        """
        effective_lr = self.lr * self._dopamine_gate(delta) if delta is not None else self.lr

        x_norm = self._normalize_input(x)
        gc = self._granule_activation(x_norm)

        prediction = self.pc_weights @ gc + self.pc_bias
        climbing_error = target - prediction

        error_norm = np.linalg.norm(climbing_error)
        if error_norm > self.grad_clip:
            climbing_error = climbing_error * (self.grad_clip / error_norm)

        self.pc_weights += effective_lr * np.outer(climbing_error, gc)
        self.pc_bias    += effective_lr * climbing_error

        squared_error = float(np.sum((target - prediction) ** 2))
        return prediction, squared_error


class ForwardModel3F(Cerebellum3F):
    """Dopamine-gated forward model: x(t+1) = F(x(t), u(t)).

    Same API as ForwardModel, with an optional delta parameter in
    learn_transition().
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_granule: int = 200,
        lr: float = 0.005,
        grad_clip: float = 1.0,
        da_lr_scale: float = 2.0,
        da_baseline_ema: float = 0.05,
    ):
        super().__init__(
            input_dim=state_dim + action_dim,
            output_dim=state_dim,
            n_granule=n_granule,
            lr=lr,
            grad_clip=grad_clip,
            da_lr_scale=da_lr_scale,
            da_baseline_ema=da_baseline_ema,
        )
        self.state_dim  = state_dim
        self.action_dim = action_dim

    def predict_next_state(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """Predict x(t+1) from x(t) and u(t). Identical to ForwardModel."""
        x = np.concatenate([state, action])
        return self.predict(x)

    def learn_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        delta: float | None = None,
    ) -> float:
        """Learn forward model transition with optional dopamine modulation.

        Parameters
        ----------
        state, action, next_state : np.ndarray
            As in ForwardModel.learn_transition().
        delta : float or None
            TD error from bg.learn(). Modulates effective lr.

        Returns
        -------
        error : float
            Prediction MSE before update.
        """
        x = np.concatenate([state, action])
        _, error = self.learn(x, next_state, delta=delta)
        return error


class InverseModel3F(Cerebellum3F):
    """Dopamine-gated inverse model: u = F^{-1}(x, x_desired).

    Same API as InverseModel, with an optional delta parameter in
    learn_action().
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_granule: int = 200,
        lr: float = 0.005,
        grad_clip: float = 1.0,
        da_lr_scale: float = 2.0,
        da_baseline_ema: float = 0.05,
    ):
        super().__init__(
            input_dim=state_dim * 2,
            output_dim=action_dim,
            n_granule=n_granule,
            lr=lr,
            grad_clip=grad_clip,
            da_lr_scale=da_lr_scale,
            da_baseline_ema=da_baseline_ema,
        )
        self.state_dim  = state_dim
        self.action_dim = action_dim

    def compute_action(
        self, state: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Compute the action that moves state toward target. No learning."""
        x = np.concatenate([state, target])
        return self.predict(x)

    def learn_action(
        self,
        state: np.ndarray,
        target: np.ndarray,
        correct_action: np.ndarray,
        delta: float | None = None,
    ) -> float:
        """Learn inverse model mapping with optional dopamine modulation.

        Parameters
        ----------
        state, target, correct_action : np.ndarray
            As in InverseModel.learn_action().
        delta : float or None
            TD error from bg.learn(). Modulates effective lr.

        Returns
        -------
        error : float
            Prediction MSE before update.
        """
        x = np.concatenate([state, target])
        _, error = self.learn(x, correct_action, delta=delta)
        return error
