"""
Cerebral Cortex Improved — Multi-Frame Temporal Representation

Extends the basic cortex with temporal context by stacking multiple
frames together. This allows the cortex to perceive velocity and motion,
not just static features.

Also includes a fixed non-linear feature space before sparse coding,
which helps discover more useful representations.

Key methods:
  - encode(x): Extract features from observation without modifying buffer
  - encode_and_learn(x): Extract features and update weights
  - reset_episode(): Clear buffer at episode start
"""

import numpy as np
from collections import deque
from cerebral_cortex import CerebralCortex


class CerebralCortexImproved:
    """Frame-stacking cortex with fixed non-linear projection and Hebbian sparse coding.

    Parameters
    ----------
    input_dim : int
        Dimensionality of a single observation (4 for CartPole).
    n_frames : int
        Number of frames to stack (default 4).
    repr_dim : int
        Dimensionality of the learned sparse representation (output).
    hidden_dim : int
        Dimensionality of the hidden projection layer (fixed random tanh).
    lr : float
        Hebbian learning rate for sparse coding weights.
    sparsity : float
        L1 sparsity penalty strength in relaxation dynamics.
    relaxation_steps : int
        Number of integration steps to find sparse code y.
    relaxation_dt : float
        Integration step size for relaxation.
    """

    def __init__(
        self,
        input_dim: int = 4,
        n_frames: int = 4,
        repr_dim: int = 8,
        hidden_dim: int = 16,
        lr: float = 0.005,
        sparsity: float = 0.1,
        relaxation_steps: int = 20,
        relaxation_dt: float = 0.05,
    ):
        self.input_dim = input_dim
        self.n_frames = n_frames
        self.repr_dim = repr_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.sparsity = sparsity
        self.relaxation_steps = relaxation_steps
        self.relaxation_dt = relaxation_dt

        # Frame buffer: holds up to n_frames past observations
        self.frame_buffer: deque = deque(maxlen=n_frames)

        stacked_dim = input_dim * n_frames  # 16D for n_frames=4, input_dim=4

        # Fixed random non-linear projection (not trained)
        # Layer 1: stacked_dim → 32D
        scale1 = 1.0 / np.sqrt(stacked_dim)
        self._proj_W1 = np.random.randn(32, stacked_dim) * scale1
        self._proj_b1 = np.zeros(32)
        # Layer 2: 32D → hidden_dim
        scale2 = 1.0 / np.sqrt(32)
        self._proj_W2 = np.random.randn(hidden_dim, 32) * scale2
        self._proj_b2 = np.zeros(hidden_dim)

        # Hebbian sparse coding: hidden_dim → repr_dim
        self.W = np.random.randn(repr_dim, hidden_dim) * 0.1
        self._normalize_weights()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_weights(self):
        """Keep each row of W on the unit sphere for stable Hebbian learning."""
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        self.W /= np.maximum(norms, 1e-8)

    def _project(self, stacked: np.ndarray) -> np.ndarray:
        """Fixed non-linear projection: stacked_dim → hidden_dim (tanh)."""
        h1 = np.tanh(self._proj_W1 @ stacked + self._proj_b1)
        h2 = np.tanh(self._proj_W2 @ h1 + self._proj_b2)
        return h2

    def _sparse_encode(self, h: np.ndarray) -> np.ndarray:
        """Sparse coding relaxation on projected input h (Eq. 17 Doya).

        ẏ = Wh - WW'y - sparsity·sign(y)
        """
        y = np.zeros(self.repr_dim)
        Wh = self.W @ h
        for _ in range(self.relaxation_steps):
            reconstruction = self.W.T @ y
            recurrent = self.W @ reconstruction
            dy = Wh - recurrent - self.sparsity * np.sign(y)
            y = y + self.relaxation_dt * dy
        return y

    def _stacked_peek(self, x: np.ndarray) -> np.ndarray:
        """Return stacked input with x as current frame WITHOUT pushing to buffer.

        Takes the (n_frames-1) most recent frames from the buffer and appends x,
        padding with zeros if the buffer has fewer than (n_frames-1) entries.
        """
        buf = list(self.frame_buffer)
        n_prev = self.n_frames - 1
        if len(buf) >= n_prev:
            prev = buf[-n_prev:] if n_prev > 0 else []
        else:
            padding = [np.zeros(self.input_dim)] * (n_prev - len(buf))
            prev = padding + buf
        return np.concatenate(prev + [x])

    def _stacked_push(self, x: np.ndarray) -> np.ndarray:
        """Push x to buffer and return the full n_frames stacked input."""
        self.frame_buffer.append(x.copy())
        buf = list(self.frame_buffer)
        if len(buf) < self.n_frames:
            padding = [np.zeros(self.input_dim)] * (self.n_frames - len(buf))
            buf = padding + buf
        return np.concatenate(buf)

    # ------------------------------------------------------------------
    # Public interface (mirrors CerebralCortex API)
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode x without modifying the frame buffer.

        Used for value estimation (next-state lookups) where we want to
        preview a future frame without committing it to the buffer.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)

        Returns
        -------
        y : np.ndarray, shape (repr_dim,)
        """
        x = np.asarray(x, dtype=np.float64)
        stacked = self._stacked_peek(x)
        h = self._project(stacked)
        return self._sparse_encode(h)

    def encode_and_learn(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Push x to buffer, encode, and update Hebbian weights (Eq. 18 Doya).

        ΔW = η · y · (h - W'y)'   (h = non-linear projection of stacked frames)

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)

        Returns
        -------
        y : np.ndarray, shape (repr_dim,)
        recon_error : float
            ||h - W'y||^2 reconstruction error.
        """
        x = np.asarray(x, dtype=np.float64)
        stacked = self._stacked_push(x)
        h = self._project(stacked)
        y = self._sparse_encode(h)

        reconstruction = self.W.T @ y
        error = h - reconstruction
        dW = np.outer(y, error)
        self.W += self.lr * dW
        self._normalize_weights()

        recon_error = float(np.sum(error ** 2))
        return y, recon_error

    def reset_episode(self):
        """Clear the frame buffer at the start of a new episode."""
        self.frame_buffer.clear()


# ======================================================================
# Three-Factor variant: dopaminergic gating via synaptic tag model
# ======================================================================

class CerebralCortexImproved3F(CerebralCortexImproved):
    """Improved cortex with three-factor neuromodulated Hebbian learning.

    Extends CerebralCortexImproved with the same synaptic tag mechanism
    as CerebralCortex3F (Frey & Morris 1997):

        ΔW = η · δ_centered(t) · tag(t-1)

    where tag(t) = outer(y_t, error_t) is the Hebbian coincidence from the
    previous step, and δ_centered is the TD error centered around a running
    mean to ensure bidirectional modulation.

    Parameters
    ----------
    delta_clip : float
        Clips |δ_centered| to prevent weight explosion.
    delta_baseline_ema : float
        EMA coefficient for centering δ (persists across episodes).
    """

    def __init__(
        self,
        input_dim: int = 4,
        n_frames: int = 4,
        repr_dim: int = 8,
        hidden_dim: int = 16,
        lr: float = 0.005,
        sparsity: float = 0.1,
        relaxation_steps: int = 20,
        relaxation_dt: float = 0.05,
        delta_clip: float = 10.0,
        delta_baseline_ema: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            n_frames=n_frames,
            repr_dim=repr_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            sparsity=sparsity,
            relaxation_steps=relaxation_steps,
            relaxation_dt=relaxation_dt,
        )
        self.delta_clip = delta_clip
        self.delta_baseline_ema = delta_baseline_ema

        # Synaptic tag: Hebbian coincidence at t, applied with δ at t+1
        self._pending_dW = np.zeros_like(self.W)

        # Running mean of δ for baseline centering (persists across episodes)
        self._delta_ema = 0.0

    def encode_and_learn(
        self, x: np.ndarray, delta: float | None = None
    ) -> tuple[np.ndarray, float]:
        """Push x to buffer, encode, and apply three-factor update.

        Phase A: apply the pending tag from t-1, scaled by δ_centered.
        Phase B: compute and store the new Hebbian tag for t.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
        delta : float or None
            TD error from bg.learn() at the PREVIOUS timestep.
            If None, falls back to standard two-factor Hebb.

        Returns
        -------
        y : np.ndarray, shape (repr_dim,)
        recon_error : float
        """
        x = np.asarray(x, dtype=np.float64)
        stacked = self._stacked_push(x)
        h = self._project(stacked)
        y = self._sparse_encode(h)

        reconstruction = self.W.T @ y
        error = h - reconstruction
        new_dW = np.outer(y, error)  # Hebbian tag for this timestep

        if delta is not None:
            # Center δ around running mean
            self._delta_ema = (
                (1.0 - self.delta_baseline_ema) * self._delta_ema
                + self.delta_baseline_ema * delta
            )
            delta_centered = float(np.clip(
                delta - self._delta_ema,
                -self.delta_clip, self.delta_clip,
            ))
            # Apply PREVIOUS tag scaled by CURRENT (one-step-delayed) δ
            self.W += self.lr * delta_centered * self._pending_dW
        else:
            # Two-factor fallback (backward compatible)
            self.W += self.lr * new_dW

        self._normalize_weights()
        self._pending_dW = new_dW  # store for next call

        recon_error = float(np.sum(error ** 2))
        return y, recon_error

    def reset_episode(self):
        """Clear frame buffer and pending tag at episode boundaries.

        The δ EMA is preserved across episodes (accumulates over full training).
        """
        super().reset_episode()      # clears frame buffer
        self._pending_dW[:] = 0.0   # stale tag from terminal state is discarded
