"""
Cerebral Cortex — State Representation Learning

Extracts compressed, sparse representations of sensory input through
unsupervised learning. Uses Hebbian plasticity to identify statistically
efficient features that capture the structure of the input.

These learned representations are then used by the basal ganglia and
cerebellum for their respective learning tasks.
"""

import numpy as np


class CerebralCortex:
    """Unsupervised learning module for state representation.

    Learns a compressed, sparse representation of high-dimensional
    sensory input through Hebbian learning with sparseness constraints.

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw sensory input (n).
    repr_dim : int
        Dimensionality of the learned representation (m < n).
    lr : float
        Learning rate for weight updates.
    sparsity : float
        Strength of the L1 sparseness penalty.
    relaxation_steps : int
        Number of relaxation iterations to find the output y.
    relaxation_dt : float
        Step size for the relaxation dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        repr_dim: int,
        lr: float = 0.005,
        sparsity: float = 0.1,
        relaxation_steps: int = 20,
        relaxation_dt: float = 0.05,
    ):
        self.input_dim = input_dim
        self.repr_dim = repr_dim
        self.lr = lr
        self.sparsity = sparsity
        self.relaxation_steps = relaxation_steps
        self.relaxation_dt = relaxation_dt

        # Weight matrix W: repr_dim × input_dim
        # Initialized with small random values, then normalized per row
        self.W = np.random.randn(repr_dim, input_dim) * 0.1
        self._normalize_weights()

    def _normalize_weights(self):
        """Keep weight vectors on the unit sphere for stability."""
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.W = self.W / norms

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Compute the cortical representation via relaxation dynamics.

        Implements Eq. 17 from Doya (1999):
            ẏ = Wx - WW'y - sparsity * sign(y)

        The dynamics settle into a fixed point that minimizes the
        reconstruction error while enforcing sparsity.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Raw sensory input.

        Returns
        -------
        y : np.ndarray, shape (repr_dim,)
            Cortical state representation.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros(self.repr_dim)

        Wx = self.W @ x  # precompute once

        for _ in range(self.relaxation_steps):
            # Recurrent inhibition: WW'y
            reconstruction = self.W.T @ y
            recurrent = self.W @ reconstruction

            # Relaxation dynamics (Eq. 17)
            dy = Wx - recurrent - self.sparsity * np.sign(y)
            y = y + self.relaxation_dt * dy

        return y

    def learn(self, x: np.ndarray, y: np.ndarray | None = None) -> float:
        """Update weights with Hebbian learning rule.

        Implements Eq. 18 from Doya (1999):
            ΔW = y·x' - y·y'·W

        This is a Hebbian potentiation (y·x') combined with an
        activity-dependent synaptic decay (y·y'·W).

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Sensory input.
        y : np.ndarray or None
            Pre-computed representation. If None, encode(x) is called.

        Returns
        -------
        recon_error : float
            Reconstruction error ||x - W'y||^2.
        """
        x = np.asarray(x, dtype=np.float64)
        if y is None:
            y = self.encode(x)

        reconstruction = self.W.T @ y
        error = x - reconstruction

        # Hebbian update: ΔW = y·(x - W'y)' = y·error'  (Eq. 18)
        dW = np.outer(y, error)
        self.W += self.lr * dW
        self._normalize_weights()

        recon_error = float(np.sum(error ** 2))
        return recon_error

    def encode_and_learn(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Encode input and update weights in one step.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)

        Returns
        -------
        y : np.ndarray, shape (repr_dim,)
            Cortical representation.
        recon_error : float
            Reconstruction error.
        """
        y = self.encode(x)
        err = self.learn(x, y)
        return y, err
