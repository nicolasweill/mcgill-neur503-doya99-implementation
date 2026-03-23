"""
Cerebellum Module — Supervised Learning

Based on Doya (1999, 2000): the cerebellum is specialized for
supervised learning, guided by the error signal encoded in climbing
fiber input from the inferior olive.

Architecture (Fig. 3 of Doya 2000, Fig. 3 of Doya 1999):

    Mossy fibers  -> Granule cells -> Parallel fibers -> Purkinje cells
    Climbing fibers (error signal from inferior olive) -> Purkinje cells
    Purkinje cells -> Deep cerebellar nuclei -> Output

Learning mechanism:
    Coincident activation of parallel fibers and climbing fibers
    induces LTD of the parallel fiber-Purkinje cell synapse
    (Ito, Sakurai & Tongroach, 1982).

This implements both:
  - A forward model: predicts x(t+1) = F(x(t), u(t))     (Eq. 21)
  - An inverse model: computes u = F^{-1}(x, x_desired)   (Fig. 11)

The supervised learning rule (Eq. 5 of Doya 1999):
    Delta_w_ij proportional to (y_hat_i - y_i) * x_j

where (y_hat - y) is the error signal from the climbing fibers
and x_j is the parallel fiber input from the granule cells.
"""

import numpy as np


class Cerebellum:
    """Supervised learning module implementing cerebellar internal models.

    The granule cell layer expands the input into a high-dimensional
    representation (basis function expansion). The Purkinje cells
    linearly combine these expanded inputs. The climbing fiber error
    signal drives LTD of the parallel fiber-Purkinje synapses.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the combined input (state + action for
        forward model, or state + target for inverse model).
    output_dim : int
        Dimensionality of the prediction output.
    n_granule : int
        Number of granule cells (expansion layer). Doya notes that
        the massive number of granule cells work as expansion encoders
        of the mossy fiber input signal (Section 3.1.3).
    lr : float
        Learning rate (controls LTD magnitude).
    granule_sparsity : float
        Fraction of granule cells active for each input (sparse coding).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_granule: int = 200,
        lr: float = 0.005,
        granule_sparsity: float = 0.3,
        grad_clip: float = 1.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_granule = n_granule
        self.lr = lr
        self.granule_sparsity = granule_sparsity
        self.grad_clip = grad_clip

        # ------------------------------------------------------------------
        # Granule cells: random expansion of mossy fiber input
        # Each granule cell combines different mossy fiber inputs.
        # Weights are fixed (not learned) — only the PC weights change.
        # ------------------------------------------------------------------
        self.gc_weights = np.random.randn(n_granule, input_dim)
        # Normalize rows so each granule cell has unit-norm receptive field
        norms = np.linalg.norm(self.gc_weights, axis=1, keepdims=True)
        self.gc_weights /= np.maximum(norms, 1e-8)
        self.gc_bias = np.random.uniform(-1, 1, size=n_granule)

        # ------------------------------------------------------------------
        # Purkinje cells: linear readout of parallel fiber (granule) input
        # Weights W such that output y = W @ gc_activation
        # These are the parallel fiber -> Purkinje cell synapses subject to LTD
        # ------------------------------------------------------------------
        self.pc_weights = np.zeros((output_dim, n_granule))
        self.pc_bias = np.zeros(output_dim)

        # Running statistics for input normalization
        self._input_mean = np.zeros(input_dim)
        self._input_var = np.ones(input_dim)
        self._n_samples = 0

    def _granule_activation(self, x: np.ndarray) -> np.ndarray:
        """Compute granule cell activation (parallel fiber signals).

        The granule cells provide a sparse, high-dimensional expansion
        of the mossy fiber input, similar to a radial basis function
        or random feature expansion.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Mossy fiber input (combined state and action).

        Returns
        -------
        gc : np.ndarray, shape (n_granule,)
            Granule cell activations (parallel fiber signals).
        """
        pre_activation = self.gc_weights @ x + self.gc_bias

        # ReLU activation with sparsity threshold
        gc = np.maximum(0, pre_activation)

        # Enforce sparsity: only keep top-k activations
        if self.granule_sparsity < 1.0:
            k = max(1, int(self.n_granule * self.granule_sparsity))
            if np.any(gc > 0):
                threshold = np.partition(gc, -k)[-k]
                gc = gc * (gc >= threshold)

        # Normalize to prevent exploding activations
        gc_norm = np.linalg.norm(gc)
        if gc_norm > 1e-8:
            gc = gc / gc_norm

        return gc

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Running normalization of input for training stability."""
        self._n_samples += 1
        alpha = 1.0 / self._n_samples
        self._input_mean = (1 - alpha) * self._input_mean + alpha * x
        self._input_var = (1 - alpha) * self._input_var + alpha * (
            x - self._input_mean
        ) ** 2
        std = np.sqrt(self._input_var + 1e-8)
        return (x - self._input_mean) / std

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the cerebellar circuit.

        Mossy fibers -> Granule cells -> Parallel fibers -> Purkinje cells
        -> Cerebellar nuclei -> Output

        Implements Eq. 4 of Doya (1999):
            y_i(t) = sum_j w_ij * x_j(t)

        where x_j are the parallel fiber (granule cell) activations.

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Mossy fiber input.

        Returns
        -------
        y : np.ndarray, shape (output_dim,)
            Predicted output (from cerebellar nuclei).
        """
        x_norm = self._normalize_input(x)
        gc = self._granule_activation(x_norm)
        y = self.pc_weights @ gc + self.pc_bias
        return y

    def learn(
        self, x: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """One step of supervised learning driven by climbing fiber error.

        The climbing fiber from the inferior olive carries the error
        signal (target - prediction). Coincident activation of climbing
        fibers and parallel fibers induces LTD of the parallel
        fiber-Purkinje synapse (Ito et al., 1982).

        Implements Eq. 5 of Doya (1999):
            Delta_w_ij proportional to (y_hat_i - y_i) * x_j

        Parameters
        ----------
        x : np.ndarray, shape (input_dim,)
            Mossy fiber input.
        target : np.ndarray, shape (output_dim,)
            Desired output (teacher signal, y_hat).

        Returns
        -------
        prediction : np.ndarray, shape (output_dim,)
            Current prediction before weight update.
        error : float
            Squared error ||target - prediction||^2.
        """
        x_norm = self._normalize_input(x)
        gc = self._granule_activation(x_norm)

        # Purkinje cell prediction
        prediction = self.pc_weights @ gc + self.pc_bias

        # Climbing fiber error signal (from inferior olive)
        climbing_error = target - prediction

        # Clip the error to prevent exploding gradients
        error_norm = np.linalg.norm(climbing_error)
        if error_norm > self.grad_clip:
            climbing_error = climbing_error * (self.grad_clip / error_norm)

        # LTD of parallel fiber-Purkinje synapses (Eq. 5)
        self.pc_weights += self.lr * np.outer(climbing_error, gc)
        self.pc_bias += self.lr * climbing_error

        squared_error = float(np.sum((target - prediction) ** 2))
        return prediction, squared_error


class ForwardModel(Cerebellum):
    """Cerebellar forward model: predicts next state from state and action.

    Implements x(t+1) = F(x(t), u(t))  (Eq. 21 of Doya 1999)

    Used for:
      - State estimation / sensory delay compensation (Fig. 9)
      - Simulation in virtual environment (Fig. 10)
      - Predictive action selection (Fig. 7)
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(
            input_dim=state_dim + action_dim,
            output_dim=state_dim,
            **kwargs,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def predict_next_state(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """Predict the next state given current state and action."""
        x = np.concatenate([state, action])
        return self.predict(x)

    def learn_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """Learn from an observed state transition.

        The climbing fiber error is the difference between the actual
        next state and the predicted next state.
        """
        x = np.concatenate([state, action])
        _, error = self.learn(x, next_state)
        return error


class InverseModel(Cerebellum):
    """Cerebellar inverse model: computes action from state and target.

    Implements u = F^{-1}(x, x_desired)  (Fig. 11 of Doya 1999)

    Used for encapsulating learned sensory-motor mappings for robust
    execution and quick reaction (Section 4.3.3 of Doya 1999).
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(
            input_dim=state_dim * 2,
            output_dim=action_dim,
            **kwargs,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def compute_action(
        self, state: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Compute the action to move from state toward target."""
        x = np.concatenate([state, target])
        return self.predict(x)

    def learn_action(
        self,
        state: np.ndarray,
        target: np.ndarray,
        correct_action: np.ndarray,
    ) -> float:
        """Learn from the action that was actually successful.

        The teacher signal comes from the cortical feedback pathway
        (Doya 1999, Section 4.3.3): the cerebellum learns to replicate
        the input-output mapping acquired elsewhere in the brain.
        """
        x = np.concatenate([state, target])
        _, error = self.learn(x, correct_action)
        return error
