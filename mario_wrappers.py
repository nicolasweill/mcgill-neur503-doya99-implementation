"""
Environment Wrappers — Retinal Preprocessing Pipeline

Physiological mapping of image preprocessing to early visual processing:

    SkipFrame     : Motor command persistence (~240ms between re-evaluations).
                    The basal ganglia maintain an action for several frames
                    before the next deliberative cycle completes.

    GrayScaleResize : Rod-dominated luminance processing in the retina.
                      Spatial downsampling by retinal ganglion cells with
                      large receptive fields (magnocellular pathway).

    FrameStack    : Temporal integration in V1 direction-selective simple
                    cells. Stacking 4 consecutive frames provides velocity
                    information that a single static frame cannot convey,
                    analogous to motion-energy computation (Adelson & Bergen,
                    1985).

These wrappers produce a 84x84x4 float32 tensor suitable for the cortical
CNN encoder.

Supports both:
  - Super Mario Bros (gym-super-mario-bros + nes-py)
  - Atari (ale-py + gymnasium) as fallback
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import cv2


class SkipFrame(gym.Wrapper):
    """Repeat the chosen action for `skip` frames, accumulate reward.

    Physiological basis: motor commands persist for ~100-250ms before
    cortical re-evaluation. At 60fps, 4-frame skip = ~67ms per frame
    x 4 = ~267ms, matching the ballpark of voluntary movement initiation
    latency (Georgopoulos et al., 1982).
    """

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info


class GrayScaleResize(gym.ObservationWrapper):
    """Convert RGB to grayscale and resize to `size x size`.

    Physiological basis:
      - Grayscale: luminance channel extracted by retinal rod photoreceptors
        and magnocellular ganglion cells. Color (parvocellular) is less
        critical for spatial navigation and motor control.
      - Resize: spatial pooling by retinal ganglion cells with large
        receptive fields. The foveal region provides high acuity but the
        overall spatial resolution available to higher areas is limited.
    """

    def __init__(self, env, size: int = 84):
        super().__init__(env)
        self._size = size
        # New observation space: single-channel grayscale
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(size, size, 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # Convert to grayscale using luminance formula
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize with area interpolation (anti-aliased downsampling)
        resized = cv2.resize(gray, (self._size, self._size), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class FrameStack(gym.Wrapper):
    """Stack the last `n_frames` observations along the channel axis.

    Physiological basis: V1 direction-selective neurons integrate temporal
    information over ~50-100ms windows (Adelson & Bergen, 1985). Stacking
    4 consecutive (post-skip) frames provides the equivalent of a
    motion-energy representation, enabling the cortical encoder to
    extract velocity and acceleration information.

    Output shape: (H, W, n_frames) with float32 in [0, 1].
    """

    def __init__(self, env, n_frames: int = 4):
        super().__init__(env)
        self._n_frames = n_frames
        self._frames = []

        # Update observation space
        old_space = env.observation_space
        h, w = old_space.shape[0], old_space.shape[1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(h, w, n_frames),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Squeeze single channel if present
        if obs.ndim == 3 and obs.shape[2] == 1:
            obs = obs[:, :, 0]
        # Normalize to [0, 1]
        frame = obs.astype(np.float32) / 255.0
        self._frames = [frame] * self._n_frames
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if obs.ndim == 3 and obs.shape[2] == 1:
            obs = obs[:, :, 0]
        frame = obs.astype(np.float32) / 255.0
        self._frames.append(frame)
        if len(self._frames) > self._n_frames:
            self._frames.pop(0)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        # Stack along channel axis: (H, W, n_frames)
        return np.stack(self._frames, axis=-1)


# ======================================================================
# Environment factory
# ======================================================================

def make_mario_env(render_mode=None):
    """Create Super Mario Bros environment with retinal preprocessing.

    Falls back to Atari MsPacman if gym-super-mario-bros is not available
    (requires nes-py C++ compilation).
    """
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

        env = gym_super_mario_bros.make(
            'SuperMarioBros-v0',
            apply_api_compatibility=True,
            render_mode=render_mode,
        )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        n_actions = len(SIMPLE_MOVEMENT)  # 7
        env_name = "SuperMarioBros-v0"
    except (ImportError, Exception):
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/MsPacman-v5", render_mode=render_mode)
        n_actions = env.action_space.n  # 9
        env_name = "ALE/MsPacman-v5 (fallback)"

    env = SkipFrame(env, skip=4)
    env = GrayScaleResize(env, size=84)
    env = FrameStack(env, n_frames=4)

    return env, n_actions, env_name
