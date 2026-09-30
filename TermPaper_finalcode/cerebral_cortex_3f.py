"""
Cerebral Cortex 3F — Reward-Modulated State Representation

Extends the basic cortex with dopamine modulation. The cortex now learns
to represent features that are relevant to reward, not just all statistical
structure in the input.

Reward signal from the basal ganglia gates whether Hebbian learning
occurs at each synapse.
"""

import numpy as np
from cerebral_cortex import CerebralCortex


class CerebralCortex3F(CerebralCortex):
    """Cortex with three-factor neuromodulated Hebbian learning.

    Extends CerebralCortex with optional dopaminergic gating of weight
    updates. When delta=None is passed, degrades gracefully to the
    standard two-factor rule (backward compatible).

    Parameters
    ----------
    input_dim, repr_dim, lr, sparsity, relaxation_steps, relaxation_dt :
        Identical to CerebralCortex.
    delta_clip : float
        Clips |δ_centered| before the weight update to prevent weight
        explosion during early training when TD errors are noisy.
    delta_baseline_ema : float
        EMA coefficient for the running baseline of δ. Centering δ around
        its running mean ensures that both above-average rewards (positive
        update) and below-average rewards (negative update) affect learning.
        Typical range: 0.05–0.2.
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int,
        lr: float = 0.005,
        sparsity: float = 0.02,
        relaxation_steps: int = 10,
        relaxation_dt: float = 0.03,
        delta_clip: float = 10.0,
        delta_baseline_ema: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            repr_dim=repr_dim,
            lr=lr,
            sparsity=sparsity,
            relaxation_steps=relaxation_steps,
            relaxation_dt=relaxation_dt,
        )
        self.delta_clip = delta_clip
        self.delta_baseline_ema = delta_baseline_ema

        # Synaptic tag: Hebbian update computed at t, applied at t+1
        # when δ_t becomes available.
        self._pending_dW = np.zeros_like(self.W)

        # Running mean of δ for baseline centering.
        # Not reset between episodes — accumulates across full training.
        self._delta_ema = 0.0

    def encode_and_learn(
        self, x: np.ndarray, delta: float | None = None
    ) -> tuple[np.ndarray, float]:
        """Encode input and update weights with optional dopamine modulation.

        Two-phase update implementing the synaptic tag model:
          Phase A — Apply the pending tag from t-1, scaled by δ_t-1 (= delta).
          Phase B — Compute and store the new Hebbian tag for t.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Raw sensory input at the current timestep.
        delta : float or None
            TD error δ from bg.learn() at the *previous* timestep.
            If None, falls back to standard two-factor Hebb (Doya 1999 Eq. 18).

        Returns
        -------
        y : np.ndarray, shape (repr_dim,)
            Cortical representation (from encode()).
        recon_error : float
            ||x - W'y||^2 reconstruction error.
        """
        x = np.asarray(x, dtype=np.float64)
        y = self.encode(x)

        reconstruction = self.W.T @ y
        error = x - reconstruction
        new_dW = np.outer(y, error)   # Hebbian tag for this timestep

        if delta is not None:
            # --- Three-factor update ---
            # Center δ around running mean: only surprises above baseline
            # (unexpected rewards) reinforce; surprises below baseline weaken.
            self._delta_ema = (
                (1.0 - self.delta_baseline_ema) * self._delta_ema
                + self.delta_baseline_ema * delta
            )
            delta_centered = float(np.clip(
                delta - self._delta_ema,
                -self.delta_clip, self.delta_clip
            ))
            # Apply PREVIOUS tag scaled by CURRENT (one-step-delayed) δ
            self.W += self.lr * delta_centered * self._pending_dW
        else:
            # --- Two-factor fallback (backward compatible) ---
            self.W += self.lr * new_dW

        self._normalize_weights()

        # Store current tag for next call
        self._pending_dW = new_dW

        recon_error = float(np.sum(error ** 2))
        return y, recon_error

    def reset_episode(self):
        """Clear the pending synaptic tag at episode boundaries.

        Called optionally at the start of each episode to avoid carrying
        a stale tag from the terminal state into the first step of the
        next episode. The δ EMA is preserved across episodes.
        """
        self._pending_dW[:] = 0.0
