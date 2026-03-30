"""
Cerebral Cortex Module — Unsupervised Visual Learning (CNN)

Adapts Doya (1999, 2000) cortical unsupervised learning for image
observations using a convolutional sparse autoencoder.

Physiological mapping of the architecture:

    Layer               Cortical area    Justification
    ─────────────────   ──────────────   ──────────────────────────────────
    Conv(4,32,8,s=4)    V1               Simple/complex cells with 8x8
                                         receptive fields (Hubel & Wiesel, 1962)
    Conv(32,64,4,s=2)   V2               Intermediate features, texture
    Conv(64,64,3,s=1)   V4               Shape-selective neurons
    FC(3136,256)        IT cortex        Object-level representations
    FC(256,64)          Cortical output  Compact state for downstream modules

The autoencoder loss preserves Doya's energy function (Eq. 17):

    E = ||x - W'y||^2 + lambda * sum|y_i|

where the first term is reconstruction error (decoder) and the second
is L1 sparsity on the cortical representation, matching the
`sparsity * sign(y)` term in the relaxation dynamics.

The decoder corresponds to top-down generative connections in the visual
cortex (Rao & Ballard, 1999 predictive coding framework).

The cortex trains unsupervised: `features.detach()` ensures that
dopaminergic TD error from the basal ganglia does NOT backpropagate
into cortical weights. This is biologically correct — dopamine modulates
corticostriatal synapses (in the BG), not intracortical synapses.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class _Encoder(nn.Module):
    """Visual hierarchy: V1 -> V2 -> V4 -> IT -> cortical output."""

    def __init__(self, in_channels: int = 4, repr_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            # V1: simple/complex cells, 8x8 receptive fields
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # V2: intermediate feature detectors
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # V4: shape-selective neurons
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # IT cortex: object-level representation
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, repr_dim),
        )

    def forward(self, x):
        # x: (batch, channels, H, W)
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class _Decoder(nn.Module):
    """Top-down generative connections (Rao & Ballard, 1999)."""

    def __init__(self, repr_dim: int = 64, out_channels: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(repr_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=8, stride=4),
            nn.Sigmoid(),  # output in [0,1] to match normalized input
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.reshape(h.size(0), 64, 7, 7)
        return self.deconv(h)


class CerebralCortexMario:
    """CNN-based unsupervised cortical module for image observations.

    Preserves the API of the original CerebralCortex:
        encode(frame)           -> features (numpy, 64D)
        encode_and_learn(frame) -> (features, reconstruction_error)

    Parameters
    ----------
    in_channels : int
        Number of input channels (frame stack depth, default 4).
    repr_dim : int
        Cortical representation dimensionality (default 64).
    lr : float
        Learning rate for autoencoder (Adam optimizer).
    sparsity_lambda : float
        Weight of L1 sparsity penalty on representations.
        Corresponds to Doya's sparsity parameter in Eq. 17.
    buffer_size : int
        Replay buffer capacity for asynchronous cortex training.
    batch_size : int
        Mini-batch size for autoencoder updates.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        in_channels: int = 4,
        repr_dim: int = 64,
        lr: float = 1e-4,
        sparsity_lambda: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.repr_dim = repr_dim
        self.sparsity_lambda = sparsity_lambda
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.encoder = _Encoder(in_channels, repr_dim).to(self.device)
        self.decoder = _Decoder(repr_dim, in_channels).to(self.device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )

        # Replay buffer for asynchronous unsupervised training
        self._buffer = deque(maxlen=buffer_size)
        self._train_steps = 0

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert (H, W, C) numpy frame to (1, C, H, W) tensor."""
        # frame is (84, 84, 4) float32 in [0, 1]
        t = torch.from_numpy(frame).float()
        t = t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return t.to(self.device)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Compute cortical representation (no learning).

        Parameters
        ----------
        frame : np.ndarray, shape (84, 84, 4)
            Preprocessed observation (grayscale, stacked, normalized).

        Returns
        -------
        features : np.ndarray, shape (repr_dim,)
            Cortical feature vector. Detached from computation graph
            so downstream modules (BG) cannot backpropagate into cortex.
        """
        with torch.no_grad():
            t = self._frame_to_tensor(frame)
            z = self.encoder(t)
        return z.squeeze(0).cpu().numpy()

    def store_frame(self, frame: np.ndarray):
        """Add frame to replay buffer for asynchronous training."""
        self._buffer.append(frame.copy())

    def train_autoencoder(self, n_steps: int = 1) -> float:
        """Train the autoencoder on a mini-batch from the replay buffer.

        Loss = MSE(reconstruction) + lambda * L1(features)

        This is the direct implementation of Doya's cortical energy
        function E = ||x - W'y||^2 + sum|y_i| (Eq. 17).

        Returns
        -------
        loss : float
            Total loss value.
        """
        if len(self._buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        for _ in range(n_steps):
            # Sample mini-batch
            indices = np.random.choice(len(self._buffer), self.batch_size, replace=False)
            batch = np.stack([self._buffer[i] for i in indices])
            # (batch, H, W, C) -> (batch, C, H, W)
            x = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(self.device)

            # Forward pass
            z = self.encoder(x)
            x_recon = self.decoder(z)

            # Doya Eq. 17: E = ||x - W'y||^2 + lambda * sum|y_i|
            recon_loss = nn.functional.mse_loss(x_recon, x)
            sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
            loss = recon_loss + sparsity_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self._train_steps += 1

        return total_loss / n_steps

    def encode_and_learn(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Encode frame, store in buffer, and optionally train.

        Combines encoding with unsupervised learning, matching the
        API of the original CerebralCortex.encode_and_learn().

        Returns
        -------
        features : np.ndarray, shape (repr_dim,)
        recon_error : float
        """
        features = self.encode(frame)
        self.store_frame(frame)

        # Train every 4 calls if buffer has enough data
        error = self.train_autoencoder(n_steps=1)

        return features, error

    def warmup(self, frames: list[np.ndarray], n_epochs: int = 10) -> float:
        """Pre-train autoencoder on collected frames before RL starts.

        Parameters
        ----------
        frames : list of np.ndarray
            Collection of preprocessed frames.
        n_epochs : int
            Number of passes through the data.

        Returns
        -------
        final_loss : float
        """
        for f in frames:
            self._buffer.append(f.copy())

        print(f"    Cortex warmup: {len(frames)} frames, {n_epochs} epochs...")
        losses = []
        steps_per_epoch = max(1, len(frames) // self.batch_size)
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for _ in range(steps_per_epoch):
                epoch_loss += self.train_autoencoder(n_steps=1)
            avg = epoch_loss / steps_per_epoch
            losses.append(avg)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:3d}: loss = {avg:.4f}")

        return losses[-1] if losses else 0.0

    def save(self, path: str):
        """Save encoder and decoder weights."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "repr_dim": self.repr_dim,
            "sparsity_lambda": self.sparsity_lambda,
        }, path)

    def load(self, path: str):
        """Load encoder and decoder weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
