#!/usr/bin/env python3
"""
Mario PPO – High-performance PPO agent for Super Mario Bros.

Install dependencies:
    pip install gym-super-mario-bros stable-baselines3[extra] shimmy gymnasium torch opencv-python

Usage:
    python mario_ppo.py --train                                          # Train from scratch
    python mario_ppo.py --train --resume checkpoints/latest_model.zip   # Resume training
    python mario_ppo.py --play checkpoints/best_model.zip               # Watch the agent play
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import gymnasium
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import get_linear_fn

# ─── Config ───────────────────────────────────────────────────────────────────

TOTAL_EPOCHS      = 1000
STEPS_PER_EPOCH   = 4096        # one "epoch" = 4 096 env steps
FRAME_SKIP        = 4           # repeat each action for N frames
N_STACK           = 4           # stack N consecutive grayscale frames
OBS_SIZE          = 84          # resize obs to OBS_SIZE × OBS_SIZE

CHECKPOINT_DIR    = Path("checkpoints")
BEST_MODEL_PATH   = CHECKPOINT_DIR / "best_model"      # .zip appended by SB3
LATEST_MODEL_PATH = CHECKPOINT_DIR / "latest_model"
METRICS_LOG       = CHECKPOINT_DIR / "metrics.csv"


# ─── Wrappers ─────────────────────────────────────────────────────────────────

# Global frame storage for display
_display_frame = None

def set_display_frame(frame):
    """Store the latest RGB frame for display."""
    global _display_frame
    _display_frame = frame

def get_display_frame():
    """Retrieve the latest RGB frame."""
    return _display_frame


class SkipFrame(gymnasium.Wrapper):
    """Repeat the same action for `skip` frames, accumulating rewards."""

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward, done = 0.0, False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GrayscaleWrapper(gymnasium.Wrapper):
    """Convert RGB observation to grayscale."""

    def __init__(self, env):
        super().__init__(env)
        import cv2
        self.cv2 = cv2
        h, w, _ = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_gray(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._to_gray(obs), info

    def _to_gray(self, obs):
        # Handle case where obs is a tuple (from shimmy)
        if isinstance(obs, tuple):
            obs = obs[0]
        gray = self.cv2.cvtColor(obs, self.cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, axis=-1)


class ResizeWrapper(gymnasium.Wrapper):
    """Resize observation to a smaller size."""

    def __init__(self, env, size: int = 84):
        super().__init__(env)
        import cv2
        self.cv2 = cv2
        self.size = size
        h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(size, size, c), dtype=np.uint8
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._resize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._resize(obs), info

    def _resize(self, obs):
        resized = self.cv2.resize(obs, (self.size, self.size), interpolation=self.cv2.INTER_AREA)
        # cv2.resize drops the channel dimension for single-channel images, so add it back
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        return resized


class FrameStackGym(gymnasium.Wrapper):
    """
    Stack `n` grayscale frames along the channel axis.
    Input obs shape : (H, W, 1)   (one grayscale frame)
    Output obs shape: (H, W, n)   (n stacked frames, HWC format)
    """

    def __init__(self, env, n: int = 4):
        super().__init__(env)
        self._n = n
        self._buf: deque = deque(maxlen=n)
        h, w, _ = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(h, w, n), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._n):
            self._buf.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buf.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> np.ndarray:
        # Concatenate along channel axis → (H, W, n_stack)
        return np.concatenate(list(self._buf), axis=-1)


class FrameCapturerWrapper:
    """Captures the raw RGB frame before any processing for display (works with old gym wrappers)."""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.call_count = 0
    
    def step(self, action):
        result = self.env.step(action)
        # Capture the RGB frame from the environment
        self._capture_frame()
        return result
    
    def reset(self):
        obs = self.env.reset()
        self._capture_frame()
        return obs
    
    def _capture_frame(self):
        """Extract RGB frame from base environment."""
        self.call_count += 1
        try:
            env = self.env
            # Traverse down to find the base gym environment
            depth = 0
            while hasattr(env, 'env') and depth < 50:
                env = env.env
                depth += 1
            
            # Get the raw RGB frame
            if hasattr(env, 'render'):
                try:
                    # render() returns the frame as numpy array when mode='rgb_array'
                    frame = env.render(mode='rgb_array')
                    if frame is not None and len(frame.shape) == 3:
                        # Store globally so play() can access it
                        set_display_frame(frame)
                except Exception as e:
                    pass
        except Exception as e:
            pass
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


class ManualRenderWrapper(gymnasium.Wrapper):
    """Wrapper to enable manual rendering without automatic shimmy calls."""
    
    def __init__(self, env, render_mode: str | None = None):
        super().__init__(env)
        self._manual_render_mode = render_mode
        self._viewer = None
        self._render_initialized = False
    
    def render(self):
        """Render the environment."""
        if self._manual_render_mode == "human":
            try:
                # Find the base gym environment for rendering
                env = self.env
                depth = 0
                while hasattr(env, 'env') and depth < 20:
                    env = env.env
                    depth += 1
                
                # Try to render
                if hasattr(env, 'render'):
                    env.render(mode='human')
                else:
                    if not self._render_initialized:
                        print(f"Warning: No render method found on base env (depth: {depth})")
                        self._render_initialized = True
            except Exception as e:
                if not self._render_initialized:
                    print(f"Warning: Render failed: {type(e).__name__}: {str(e)[:100]}")
                    self._render_initialized = True
        return None
    
    def close(self):
        """Close the environment and cleanup."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        super().close()


# ─── Environment factory ──────────────────────────────────────────────────────

class OldGymToNewConverter:
    """Convert old gym API (4-value step) to new gym API (5-value step)."""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, 'metadata', {})
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            # Old gym API: (obs, reward, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        else:
            # Already new format (shouldn't happen with unwrapped env)
            return result
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Old gym SuperMarioBrosEnv returns just obs (numpy array)
        if isinstance(result, np.ndarray):
            return result, {}  # Convert to (obs, info)
        elif isinstance(result, tuple) and len(result) == 2:
            return result  # Already (obs, info)
        else:
            return result, {}
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


def make_env(render_mode: str | None = None):
    """
    Returns a zero-argument callable that builds one wrapped Mario env.
    Pass render_mode="human" to enable the game window (play mode).

    Wrapper chain:  
        gym_super_mario_bros.make → creates base environment
        OldGymToNewConverter → converts old 4-value step to new 5-value format
        JoypadSpace       → restrict to SIMPLE_MOVEMENT actions
        GymV21CompatibilityV0 → convert to gymnasium API
        SkipFrame         → repeat action 4 frames, sum rewards
        GrayscaleWrapper  → RGB (240,256,3) → grayscale (240,256,1)
        ResizeWrapper     → (240,256,1) → (84,84,1)
        FrameStackGym     → (84,84,1) × 4 → (84,84,4)  HWC

    SB3's PPO detects HWC obs and auto-applies VecTransposeImage → (4,84,84) CHW
    which is the format expected by NatureCNN.
    """
    def _init():
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        # Unwrap to remove the incompatible time_limit wrapper
        base_env = env.unwrapped
        
        # JoypadSpace works with old gym API
        env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        
        # Now we need to have our wrappers work with the old gym API (4-value step)
        # Create a wrapper that handles the old gym (4-value) API but pretends to be new (5-value)
        # but only for the wrappers we apply BEFORE shimmy
        
        # Create simplified wrappers that work with old gym API
        class SkipFrameOld:
            def __init__(self, env, skip=4):
                self.env = env
                self._skip = skip
                self.observation_space = env.observation_space
                self.action_space = env.action_space
            
            def step(self, action):
                total_reward, done = 0.0, False
                for _ in range(self._skip):
                    obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
                return obs, total_reward, done, info
            
            def reset(self):
                return self.env.reset()
            
            def render(self, mode=None):
                if hasattr(self.env, 'render'):
                    return self.env.render(mode=mode)
            
            def close(self):
                self.env.close()
        
        class GrayscaleWrapperOld:
            def __init__(self, env):
                import cv2
                import gym
                self.env = env
                self.cv2 = cv2
                h, w, _ = env.observation_space.shape
                self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
                self.action_space = env.action_space
            
            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                return self._to_gray(obs), reward, done, info
            
            def reset(self):
                obs = self.env.reset()
                return self._to_gray(obs)
            
            def _to_gray(self, obs):
                if isinstance(obs, tuple):
                    obs = obs[0]
                gray = self.cv2.cvtColor(obs, self.cv2.COLOR_RGB2GRAY)
                return np.expand_dims(gray, axis=-1)
            
            def render(self, mode=None):
                if hasattr(self.env, 'render'):
                    return self.env.render(mode=mode)
            
            def close(self):
                self.env.close()
        
        class ResizeWrapperOld:
            def __init__(self, env, size=84):
                import cv2
                import gym
                self.env = env
                self.cv2 = cv2
                self.size = size
                h, w, c = env.observation_space.shape
                self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size, size, c), dtype=np.uint8)
                self.action_space = env.action_space
            
            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                return self._resize(obs), reward, done, info
            
            def reset(self):
                obs = self.env.reset()
                return self._resize(obs)
            
            def _resize(self, obs):
                resized = self.cv2.resize(obs, (self.size, self.size), interpolation=self.cv2.INTER_AREA)
                if len(resized.shape) == 2:
                    resized = np.expand_dims(resized, axis=-1)
                return resized
            
            def render(self, mode=None):
                if hasattr(self.env, 'render'):
                    return self.env.render(mode=mode)
            
            def close(self):
                self.env.close()
        
        # Apply old-gym-style wrappers before shimmy conversion
        env = SkipFrameOld(env, skip=FRAME_SKIP)
        env = GrayscaleWrapperOld(env)
        env = ResizeWrapperOld(env, size=OBS_SIZE)
        
        # Add frame capturer BEFORE shimmy to get RGB frames
        env = FrameCapturerWrapper(env)
        
        # Convert to gymnasium using shimmy (without render_mode to avoid auto-rendering)
        env = GymV21CompatibilityV0(env=env, render_mode=None)
        
        # Apply gymnasium wrappers after shimmy conversion
        env = FrameStackGym(env, n=N_STACK)
        
        # Add manual render wrapper to handle rendering without pyglet issues
        env = ManualRenderWrapper(env, render_mode=render_mode)
        
        return env
    return _init


# ─── Metrics callback ─────────────────────────────────────────────────────────

class MarioCallback(BaseCallback):
    """
    Tracks per-episode stats, prints every 10 epochs, saves checkpoints.
    Epoch boundaries are detected by counting env steps inside _on_step.
    Optionally displays frames in real-time during training.
    """

    def __init__(self, steps_per_epoch: int, total_epochs: int, start_epoch: int = 0, 
                 display_frames: bool = False):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs    = total_epochs
        self.current_epoch   = start_epoch   # correct value restored on resume
        self._step_in_epoch  = 0
        self.display_frames  = display_frames
        self._stop_training  = False

        self.ep_rewards = deque(maxlen=20)
        self.ep_lengths = deque(maxlen=20)
        self.ep_x_pos   = deque(maxlen=20)

        self.best_reward = -np.inf
        self._epoch_t0   = time.perf_counter()

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── SB3 hook: called once per env step ───────────────────────────────────

    def _on_step(self) -> bool:
        self._step_in_epoch += 1

        # VecMonitor injects info["episode"] when an episode ends
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(ep["r"])
                self.ep_lengths.append(ep["l"])
                if "x_pos" in info:
                    self.ep_x_pos.append(info["x_pos"])

        # Display frame in real-time if enabled
        if self.display_frames:
            import cv2
            frame = get_display_frame()
            if frame is not None and len(frame.shape) >= 2:
                h, w = frame.shape[:2]
                # Upscale for visibility
                scale = max(1, 480 // h)
                frame_display = cv2.resize(frame, (w * scale, h * scale), 
                                          interpolation=cv2.INTER_LINEAR)
                
                # Add reward overlay only
                font = cv2.FONT_HERSHEY_SIMPLEX
                mean_r = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0
                text = f"Reward: {mean_r:.1f}"
                cv2.putText(frame_display, text, (10, 30), font, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Mario PPO Training", frame_display)
                
                # Check for ESC key to stop training (33ms ≈ 30 FPS)
                key = cv2.waitKey(33) & 0xFF
                if key == 27:  # ESC key
                    print("\n⚠ Training interrupted by user (ESC key)")
                    self._stop_training = True
                    return False  # Stop training

        if self._step_in_epoch >= self.steps_per_epoch:
            self.current_epoch += 1
            self._on_epoch_end()
            self._step_in_epoch = 0

        return True  # return False to abort training

    # ── End-of-epoch logic ────────────────────────────────────────────────────

    def _on_epoch_end(self):
        elapsed = time.perf_counter() - self._epoch_t0
        fps = self.steps_per_epoch / elapsed if elapsed > 0 else 0.0
        self._epoch_t0 = time.perf_counter()

        mean_r = float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0
        mean_l = float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0
        mean_x = float(np.mean(self.ep_x_pos))   if self.ep_x_pos   else 0.0
        epoch  = self.current_epoch

        # Always save the latest model so training can be resumed
        self.model.save(str(LATEST_MODEL_PATH))

        # Save best model whenever mean reward improves
        new_best = mean_r > self.best_reward
        if new_best:
            self.best_reward = mean_r
            self.model.save(str(BEST_MODEL_PATH))

        # Print a metrics row every 10 epochs
        if epoch % 10 == 0:
            star = " ★" if new_best else ""
            print(
                f"[{epoch:4d}/{self.total_epochs}]"
                f"  reward={mean_r:8.2f}"
                f"  ep_len={mean_l:7.1f}"
                f"  x_pos={mean_x:5.0f}"
                f"  fps={fps:5.0f}"
                f"  best={self.best_reward:8.2f}{star}"
            )

        # Append to CSV
        write_header = not METRICS_LOG.exists()
        with METRICS_LOG.open("a") as f:
            if write_header:
                f.write("epoch,timesteps,mean_reward,mean_ep_len,mean_x_pos,fps,best_reward\n")
            f.write(
                f"{epoch},{self.model.num_timesteps},"
                f"{mean_r:.4f},{mean_l:.1f},{mean_x:.1f},{fps:.1f},{self.best_reward:.4f}\n"
            )
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        import cv2
        cv2.destroyAllWindows()


# ─── Train ────────────────────────────────────────────────────────────────────

def train(resume_path: str | None = None, display_frames: bool = False) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecMonitor(vec_env)

    total_timesteps = TOTAL_EPOCHS * STEPS_PER_EPOCH

    # Linear LR schedule: 3e-4 → 1e-6 over the full training run.
    # When resuming with reset_num_timesteps=False, SB3 computes progress
    # relative to num_timesteps already done so the schedule continues
    # seamlessly from where it left off.
    lr_schedule = get_linear_fn(start=3e-4, end=1e-6, end_fraction=1.0)

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=vec_env)
        model.learning_rate = lr_schedule
        done_steps  = model.num_timesteps
        start_epoch = done_steps // STEPS_PER_EPOCH
        remaining   = total_timesteps - done_steps
        if remaining <= 0:
            print("Training already complete (no remaining steps).")
            vec_env.close()
            return
        print(f"  → resuming at epoch {start_epoch}/{TOTAL_EPOCHS}  "
              f"({done_steps:,} / {total_timesteps:,} steps done)")
    else:
        print("Starting fresh training…")
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate=lr_schedule,
            n_steps=512,          # rollout buffer length per update
            batch_size=64,        # mini-batch size for gradient updates
            n_epochs=10,          # PPO epochs per rollout
            gamma=0.9,            # discount factor
            gae_lambda=0.95,      # GAE lambda for advantage estimation
            clip_range=0.2,       # PPO clipping parameter
            ent_coef=0.01,        # entropy bonus (encourages exploration)
            vf_coef=0.5,          # value function loss weight
            max_grad_norm=0.5,    # gradient clipping
            verbose=0,
        )
        done_steps  = 0
        start_epoch = 0
        remaining   = total_timesteps

    callback = MarioCallback(
        steps_per_epoch=STEPS_PER_EPOCH,
        total_epochs=TOTAL_EPOCHS,
        start_epoch=start_epoch,
        display_frames=display_frames,
    )

    print(f"\n{'─' * 74}")
    print(f"  PPO · CnnPolicy (NatureCNN)  |  epochs={TOTAL_EPOCHS}  "
          f"steps/epoch={STEPS_PER_EPOCH:,}")
    print(f"  Total steps: {total_timesteps:,}  |  checkpoints → {CHECKPOINT_DIR}/")
    if display_frames:
        print(f"  Display: ON (press ESC in window to stop)")
    print(f"{'─' * 74}")
    print(f"{'epoch':>12}  {'reward':>8}  {'ep_len':>7}  "
          f"{'x_pos':>6}  {'fps':>5}  {'best':>8}")
    print(f"{'─' * 74}")

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=False,   # preserve global step counter for resume
            progress_bar=False,
        )
    finally:
        import cv2
        cv2.destroyAllWindows()

    print(f"\nTraining complete!  Best reward: {callback.best_reward:.2f}")
    print(f"  Best model   → {BEST_MODEL_PATH}.zip")
    print(f"  Latest model → {LATEST_MODEL_PATH}.zip")
    print(f"  Metrics CSV  → {METRICS_LOG}")
    vec_env.close()


# ─── Play ─────────────────────────────────────────────────────────────────────

def play(model_path: str) -> None:
    """Load a saved model and watch it play with a human-rendered window."""
    import cv2
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Build a single (non-VecEnv) environment WITHOUT render_mode
    # We'll display frames manually with cv2.imshow instead
    env = make_env(render_mode=None)()   # gymnasium env, obs (84,84,4) HWC

    print("Watching agent play… Press ESC to quit.\n")
    episode = 0

    try:
        while True:
            obs, _ = env.reset()
            done, total_r, steps = False, 0.0, 0
            episode += 1

            while not done:
                # Training used VecTransposeImage (HWC→CHW); replicate manually
                obs_chw = np.transpose(np.array(obs, dtype=np.uint8), (2, 0, 1))
                action, _ = model.predict(obs_chw, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                total_r += float(reward)
                steps += 1
                
                # Display the frame using cv2
                frame_to_display = None
                
                # Try to get RGB frame from global storage
                frame_to_display = get_display_frame()
                
                # Fallback to grayscale if RGB not available
                if frame_to_display is None:
                    frame_to_display = obs[:, :, -1]  # Last frame in the stack
                    # Convert grayscale to RGB for better display
                    frame_to_display = cv2.cvtColor(frame_to_display, cv2.COLOR_GRAY2BGR)
                
                # Safely check frame size and display
                if frame_to_display is not None and len(frame_to_display.shape) >= 2:
                    h, w = frame_to_display.shape[:2]
                    # Upscale significantly for better visibility
                    scale = max(1, 480 // h)  # Scale to ~480px height
                    frame_display = cv2.resize(frame_to_display, (w * scale, h * scale), 
                                              interpolation=cv2.INTER_LINEAR)
                    
                    # Add some info text to the display
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"Ep: {episode} | Step: {steps} | Reward: {total_r:.1f}"
                    cv2.putText(frame_display, text, (10, 30), font, 0.7, (0, 255, 0), 2)
                    
                    # Display with cv2
                    cv2.imshow("Mario PPO Agent", frame_display)
                
                # ESC key = 27 (33ms ≈ 30 FPS)
                key = cv2.waitKey(33) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt

            flag = "  FLAG!" if info.get("flag_get") else ""
            print(
                f"Ep {episode:3d}  |  reward={total_r:8.2f}"
                f"  steps={steps:5d}  x_pos={info.get('x_pos', '?')}{flag}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cv2.destroyAllWindows()
        env.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mario PPO – train or watch an RL agent play Super Mario Bros.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train", action="store_true",
        help="Run 500-epoch PPO training"
    )
    mode.add_argument(
        "--play", type=str, metavar="MODEL",
        help="Watch a saved model play (human render window)"
    )
    parser.add_argument(
        "--resume", type=str, metavar="MODEL",
        help="Path to a checkpoint to resume --train from"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Display the game window during training (real-time visualization)"
    )
    args = parser.parse_args()

    if args.train:
        train(resume_path=args.resume, display_frames=args.display)
    else:
        play(args.play)
