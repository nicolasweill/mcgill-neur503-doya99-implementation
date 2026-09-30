"""
Visual Cortex CNN — Feature Extraction with Reward Modulation

CNN-based feature extractor for game images. Learning rate is modulated
by the reward signal from the basal ganglia, ensuring that task-relevant
features are strengthened and irrelevant ones fade.

Optionally includes an unsupervised reconstruction loss that improves
feature quality without task-specific labels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualCortexCNN(nn.Module):
    """NatureCNN feature extractor with dopamine-gated learning rate.

    Parameters
    ----------
    n_channels : int
        Number of input channels (stacked frames, default 4).
    feature_dim : int
        Output feature dimensionality (default 256).
    lr : float
        Base learning rate for the optimizer.
    da_lr_scale : float
        Dopamine modulation amplitude. With scale=1.5:
        η ranges from 0.25× (strong negative δ) to 1.75× (strong positive δ).
    da_baseline_ema : float
        EMA coefficient for centering δ around its running mean.
    reconstruction_weight : float
        Weight of the reconstruction loss (unsupervised cortical learning).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_channels: int = 4,
        feature_dim: int = 256,
        lr: float = 3e-4,
        da_lr_scale: float = 1.5,
        da_baseline_ema: float = 0.05,
        reconstruction_weight: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.base_lr = lr
        self.da_lr_scale = da_lr_scale
        self.da_baseline_ema = da_baseline_ema
        self.reconstruction_weight = reconstruction_weight
        self.device = torch.device(device)

        # Running baseline for δ centering (persists across episodes)
        self._da_ema = 0.0

        # ── Encoder (NatureCNN) ──
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, 84, 84)
            flat_size = self.encoder(dummy).shape[1]  # Should be 3136

        self.fc_encoder = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

        # ── Decoder (reconstruction head for unsupervised loss) ──
        self.fc_decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.ReLU(),
        )
        self._flat_size = flat_size

        # Lightweight decoder: just predict a downscaled version (21×21)
        # to reduce compute. The conv encoder's final spatial size is 7×7×64.
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),   # 7→9
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2),   # 9→19
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_channels, kernel_size=4, stride=2),  # 19→40
            # We'll just use MSE on the downscaled version
        )

        self.to(self.device)

        # Optimizer with base lr (will be scaled by dopamine gate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input frames.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, 84, 84)
            Preprocessed stacked frames, values in [0, 1].

        Returns
        -------
        features : torch.Tensor, shape (batch, feature_dim)
        """
        conv_out = self.encoder(x)
        features = self.fc_encoder(conv_out)
        return features

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with reconstruction for unsupervised loss.

        Returns
        -------
        features : torch.Tensor, shape (batch, feature_dim)
        recon_loss : torch.Tensor, scalar
        """
        conv_out = self.encoder(x)
        features = self.fc_encoder(conv_out)

        # Decode: features → flat → reshape → deconv
        decoded_flat = self.fc_decoder(features)
        decoded_spatial = decoded_flat.view(-1, 64, 7, 7)
        reconstructed = self.decoder_conv(decoded_spatial)

        # Downsample input to match decoder output size
        target = F.adaptive_avg_pool2d(x, reconstructed.shape[-2:])
        recon_loss = F.mse_loss(reconstructed, target)

        return features, recon_loss

    def _dopamine_gate(self, delta: float) -> float:
        """Compute dopamine-gated learning rate multiplier.

        Maps TD error δ to a smooth, bounded gain.

        Parameters
        ----------
        delta : float
            TD error from the BG (SNc dopamine signal).

        Returns
        -------
        gate : float
            lr multiplier in [1 - scale/2, 1 + scale/2].
        """
        self._da_ema = (
            (1.0 - self.da_baseline_ema) * self._da_ema
            + self.da_baseline_ema * delta
        )
        delta_centered = delta - self._da_ema

        # Sigmoid gate
        sig = 1.0 / (1.0 + np.exp(-np.clip(delta_centered, -10, 10)))
        gate = 1.0 + self.da_lr_scale * (sig - 0.5)
        return max(0.0, gate)

    def extract_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract features from a single observation (no grad).

        Parameters
        ----------
        obs : np.ndarray, shape (84, 84, 4) HWC uint8
            Stacked grayscale frames.

        Returns
        -------
        features : np.ndarray, shape (feature_dim,)
        """
        with torch.no_grad():
            # HWC uint8 → CHW float [0,1]
            x = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            x = x.to(self.device)
            features = self.forward(x)
        return features.cpu().numpy().flatten()

    def learn(self, obs: np.ndarray, delta: float | None = None) -> tuple[np.ndarray, float]:
        """Extract features and update weights with dopamine-gated reconstruction loss.

        Parameters
        ----------
        obs : np.ndarray, shape (84, 84, 4) HWC uint8
        delta : float or None
            TD error from BG. Modulates effective learning rate.

        Returns
        -------
        features : np.ndarray, shape (feature_dim,)
        recon_loss : float
        """
        # Set effective learning rate
        if delta is not None:
            gate = self._dopamine_gate(delta)
            effective_lr = self.base_lr * gate
        else:
            effective_lr = self.base_lr

        for pg in self.optimizer.param_groups:
            pg['lr'] = effective_lr

        # HWC uint8 → CHW float [0,1]
        x = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(self.device)

        features, recon_loss = self.reconstruct(x)

        # Scale reconstruction loss
        loss = self.reconstruction_weight * recon_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return features.detach().cpu().numpy().flatten(), recon_loss.item()

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
