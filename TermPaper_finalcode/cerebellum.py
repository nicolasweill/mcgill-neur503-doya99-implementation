"""
Cerebellum — Forward and Inverse Models

Implements learning of action-outcome relationships using supervised learning.
Includes both a forward model (predicts next state from current state and action)
and an inverse model (predicts action from current and desired states).

Learning is driven by prediction error, training the internal models
to capture the dynamics of the controlled system.
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

    def save(self, path: str):
        """Save cerebellar weights (granule + Purkinje + normalization stats)."""
        np.savez(
            path,
            gc_weights=self.gc_weights,
            gc_bias=self.gc_bias,
            pc_weights=self.pc_weights,
            pc_bias=self.pc_bias,
            input_mean=self._input_mean,
            input_var=self._input_var,
            n_samples=np.array([self._n_samples]),
        )

    def load(self, path: str):
        """Load cerebellar weights."""
        data = np.load(path)
        self.gc_weights = data["gc_weights"]
        self.gc_bias = data["gc_bias"]
        self.pc_weights = data["pc_weights"]
        self.pc_bias = data["pc_bias"]
        self._input_mean = data["input_mean"]
        self._input_var = data["input_var"]
        self._n_samples = int(data["n_samples"][0])


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


class CerebellarCorrector:
    """Real-time online correction using forward model prediction error.

    Implements the cerebellar feedback error learning scheme described
    in Doya (1999, Section 4.3.2, Fig. 10): the forward model predicts
    the sensory consequences of an action, and the discrepancy between
    prediction and observation generates a corrective motor signal.

    This is the "fine correction in real time" that complements the
    inverse model's role as skill encapsulator. Biologically, the
    climbing fiber error from the inferior olive drives both:
      1. Long-term learning (LTD of parallel fiber–Purkinje synapses)
      2. Online correction (immediate adjustment of motor output)

    The correction is computed as:
        u_corrected = u_base + gain * J^T * (x_observed - x_predicted)

    where J^T approximates the inverse Jacobian of the forward model,
    mapping state-space error back into action-space corrections.

    Parameters
    ----------
    forward_model : ForwardModel
        Learned forward model for prediction.
    correction_gain : float
        Scaling factor for the corrective signal (default 0.5).
    """

    def __init__(self, forward_model: ForwardModel, correction_gain: float = 0.5):
        self.forward_model = forward_model
        self.correction_gain = correction_gain
        self._prev_state = None
        self._prev_action = None
        self._prediction_error_history = []

    def begin_step(self, state: np.ndarray, action: np.ndarray):
        """Record state and action before environment step."""
        self._prev_state = state.copy()
        self._prev_action = action.copy()

    def correct(
        self,
        observed_state: np.ndarray,
        base_action: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute corrective signal from forward model prediction error.

        If a previous (state, action) pair was recorded, the forward model's
        prediction is compared to the actual observed state. The prediction
        error in state space is projected back into action space via a
        finite-difference Jacobian approximation, producing an additive
        correction to the current action.

        Parameters
        ----------
        observed_state : np.ndarray
            Actual state observed after the previous action.
        base_action : np.ndarray
            Action proposed by the inverse model or BG for the current step.

        Returns
        -------
        corrected_action : np.ndarray
            base_action + correction.
        prediction_error : float
            Norm of the state-space prediction error (0 if no history).
        """
        if self._prev_state is None or self._prev_action is None:
            return base_action.copy(), 0.0

        # Forward model predicts what state we should be in
        predicted_state = self.forward_model.predict_next_state(
            self._prev_state, self._prev_action
        )

        # Climbing fiber error: discrepancy between prediction and reality
        state_error = observed_state - predicted_state
        prediction_error = float(np.linalg.norm(state_error))
        self._prediction_error_history.append(prediction_error)

        # Approximate inverse Jacobian via finite differences:
        # J_ij ≈ ∂F_i/∂u_j estimated by perturbing each action dimension
        action_dim = len(base_action)
        state_dim = len(observed_state)
        jacobian = np.zeros((state_dim, action_dim))
        eps = 1e-3

        for j in range(action_dim):
            perturbed = self._prev_action.copy()
            perturbed[j] += eps
            pred_plus = self.forward_model.predict_next_state(
                self._prev_state, perturbed
            )
            jacobian[:, j] = (pred_plus - predicted_state) / eps

        # Correction: J^T @ state_error maps state error to action space
        correction = self.correction_gain * (jacobian.T @ state_error)

        corrected_action = base_action + correction
        return corrected_action, prediction_error

    def reset(self):
        """Reset between episodes."""
        self._prev_state = None
        self._prev_action = None
        self._prediction_error_history = []

    @property
    def mean_prediction_error(self) -> float:
        if not self._prediction_error_history:
            return 0.0
        return float(np.mean(self._prediction_error_history))


class HybridController:
    """Automatic switching between cerebellar and BG control circuits.

    Implements the confidence-gated arbitration described conceptually
    in Doya (1999, 2000): the cerebellum handles well-learned situations
    quickly via the inverse model, but when the forward model's prediction
    error exceeds a threshold — indicating a novel or out-of-distribution
    situation — control reverts to the slower but more flexible basal
    ganglia circuit.

    The forward model acts as a surprise detector:
        if ||x_observed - x_predicted|| > threshold → use BG (deliberate)
        otherwise → use cerebellum inverse model (automatic)

    The threshold adapts online as an exponential moving average of
    recent prediction errors, allowing the system to calibrate its own
    "comfort zone".

    Parameters
    ----------
    forward_model : ForwardModel
        Cerebellar forward model (surprise detector).
    inverse_model : InverseModel
        Cerebellar inverse model (fast automatic control).
    corrector : CerebellarCorrector
        Online correction module.
    surprise_threshold_factor : float
        How many standard deviations above the running mean prediction
        error triggers a switch to BG. Default 2.0.
    ema_alpha : float
        Smoothing factor for the running statistics of prediction error.
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        inverse_model: InverseModel,
        corrector: CerebellarCorrector,
        surprise_threshold_factor: float = 2.0,
        ema_alpha: float = 0.05,
    ):
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.corrector = corrector
        self.surprise_factor = surprise_threshold_factor
        self.ema_alpha = ema_alpha

        # Running statistics for adaptive threshold
        self._ema_mean = 0.0
        self._ema_var = 0.01
        self._n_updates = 0

        # Tracking
        self._cerebellum_steps = 0
        self._bg_steps = 0

    def _update_stats(self, prediction_error: float):
        """Update exponential moving average of prediction error."""
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
        """Current adaptive threshold for switching to BG."""
        return self._ema_mean + self.surprise_factor * np.sqrt(
            max(self._ema_var, 1e-8)
        )

    def select_action(
        self,
        raw_state: np.ndarray,
        bg_state: np.ndarray,
        target_raw: np.ndarray,
        basal_ganglia,
        observed_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, float]:
        """Select action using cerebellum or BG based on confidence.

        Parameters
        ----------
        raw_state : np.ndarray
            Current raw sensory state (for cerebellar models).
        bg_state : np.ndarray
            Augmented state (for BG policy).
        target_raw : np.ndarray
            Target state for the inverse model.
        basal_ganglia : BasalGanglia
            The BG module (fallback).
        observed_state : np.ndarray or None
            If provided, used to compute prediction error from previous step.

        Returns
        -------
        action : np.ndarray
            Selected action.
        source : str
            "cerebellum" or "basal_ganglia" — which circuit produced the action.
        prediction_error : float
            Forward model prediction error (0 on first step).
        """
        # Compute prediction error from previous step
        prediction_error = 0.0
        is_surprised = False

        if observed_state is not None and self.corrector._prev_state is not None:
            predicted = self.forward_model.predict_next_state(
                self.corrector._prev_state, self.corrector._prev_action
            )
            prediction_error = float(np.linalg.norm(observed_state - predicted))
            self._update_stats(prediction_error)

            # Check if we're surprised (out of distribution)
            if self._n_updates > 10:  # need enough history
                is_surprised = prediction_error > self.surprise_threshold

        if is_surprised:
            # Fall back to slow deliberate BG circuit
            action = basal_ganglia.policy(bg_state, explore=False)
            self._bg_steps += 1
            source = "basal_ganglia"
        else:
            # Use fast cerebellar inverse model
            base_action = self.inverse_model.compute_action(raw_state, target_raw)
            # Apply online correction from forward model error
            action, _ = self.corrector.correct(observed_state if observed_state is not None else raw_state, base_action)
            action = np.clip(action, -1, 1)
            self._cerebellum_steps += 1
            source = "cerebellum"

        return action, source, prediction_error

    def reset(self):
        """Reset episode-level state (keep running stats)."""
        self.corrector.reset()

    @property
    def cerebellum_ratio(self) -> float:
        """Fraction of steps handled by the cerebellum."""
        total = self._cerebellum_steps + self._bg_steps
        if total == 0:
            return 0.0
        return self._cerebellum_steps / total

    def reset_counters(self):
        """Reset step counters (e.g. between test phases)."""
        self._cerebellum_steps = 0
        self._bg_steps = 0
