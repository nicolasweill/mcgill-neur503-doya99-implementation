#!/usr/bin/env python3
"""
Mario Doya — Modular Brain Architecture for Super Mario Bros

Trains a Doya (1999) modular architecture on Super Mario Bros:
  - Visual Cortex CNN: NatureCNN feature extractor with dopamine-gated lr
  - Basal Ganglia: Actor-Critic TD(λ) with reward shaping + ICM
  - Cerebellum: Neural forward/inverse models with dopamine gating

Preprocessing pipeline:
    Raw frames → SkipFrame(4) → Grayscale → Resize(84) → FrameStack(4) → CNN

Unlike mario_ppo.py (PPO with rollout buffers), this uses online TD(λ)
learning — each step updates weights immediately via eligibility traces.

Install dependencies:
    pip install gym-super-mario-bros shimmy gymnasium torch opencv-python numpy

Usage:
    python mario_doya.py --train                                        # Train from scratch
    python mario_doya.py --train --resume checkpoints_doya/best         # Resume training
    python mario_doya.py --play checkpoints_doya/best                   # Watch agent play
    python mario_doya.py --train --display                              # Train with live window
"""

import argparse
import sys
import time
import csv
from collections import deque
from pathlib import Path

# Unbuffer stdout to see prints immediately
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np

try:
    import gymnasium
except ImportError:
    print("Error: gymnasium is required. Install with: pip install gymnasium")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from shimmy import GymV21CompatibilityV0
except ImportError:
    print("Error: Mario dependencies required.")
    print("Install with: pip install gym-super-mario-bros shimmy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None

from mario_brain import MarioBrain


# ─── Config ───────────────────────────────────────────────────────────────────

TOTAL_EPOCHS       = 1000
STEPS_PER_EPOCH    = 4096
FRAME_SKIP         = 4
N_STACK            = 4
OBS_SIZE           = 84
N_ACTIONS          = len(SIMPLE_MOVEMENT)  # 7

CHECKPOINT_DIR     = Path("checkpoints_doya")
BEST_MODEL_PREFIX  = CHECKPOINT_DIR / "best"
LATEST_MODEL_PREFIX = CHECKPOINT_DIR / "latest"
METRICS_CSV        = CHECKPOINT_DIR / "metrics.csv"

# Global frame storage for display
_display_frame = None

def set_display_frame(frame):
    global _display_frame
    _display_frame = frame

def get_display_frame():
    return _display_frame


# ─── Environment wrappers (reused from mario_ppo.py) ─────────────────────────

class FrameCapturerWrapper:
    """Captures the raw RGB frame before any processing for display."""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        result = self.env.step(action)
        self._capture_frame()
        return result

    def reset(self):
        obs = self.env.reset()
        self._capture_frame()
        return obs

    def _capture_frame(self):
        try:
            env = self.env
            depth = 0
            while hasattr(env, 'env') and depth < 50:
                env = env.env
                depth += 1
            if hasattr(env, 'render'):
                frame = env.render(mode='rgb_array')
                if frame is not None and len(frame.shape) == 3:
                    set_display_frame(frame)
        except Exception:
            pass

    def render(self, mode=None):
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


class FrameStackGym(gymnasium.Wrapper):
    """Stack n grayscale frames along channel axis → (H, W, n)."""

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
        return np.concatenate(list(self._buf), axis=-1)


def make_env():
    """Build wrapped Mario environment.

    Pipeline: SkipFrame(4) → Grayscale → Resize(84) → FrameCapture → shimmy → FrameStack(4)
    """

    class SkipFrameOld:
        def __init__(self, env, skip=4):
            self.env = env
            self._skip = skip
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        def step(self, action):
            total_reward, done = 0.0, False
            info = {}
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
            import cv2 as _cv2
            import gym
            self.env = env
            self.cv2 = _cv2
            h, w, _ = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(h, w, 1), dtype=np.uint8
            )
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
            import cv2 as _cv2
            import gym
            self.env = env
            self.cv2 = _cv2
            self.size = size
            h, w, c = env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(size, size, c), dtype=np.uint8
            )
            self.action_space = env.action_space

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            return self._resize(obs), reward, done, info

        def reset(self):
            obs = self.env.reset()
            return self._resize(obs)

        def _resize(self, obs):
            resized = self.cv2.resize(
                obs, (self.size, self.size), interpolation=self.cv2.INTER_AREA
            )
            if len(resized.shape) == 2:
                resized = np.expand_dims(resized, axis=-1)
            return resized

        def render(self, mode=None):
            if hasattr(self.env, 'render'):
                return self.env.render(mode=mode)

        def close(self):
            self.env.close()

    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    print("[DEBUG] Mario env created", flush=True)
    base_env = env.unwrapped
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    print("[DEBUG] JoypadSpace wrapped", flush=True)

    env = SkipFrameOld(env, skip=FRAME_SKIP)
    print("[DEBUG] SkipFrame wrapped", flush=True)
    env = GrayscaleWrapperOld(env)
    print("[DEBUG] Grayscale wrapped", flush=True)
    env = ResizeWrapperOld(env, size=OBS_SIZE)
    print("[DEBUG] Resize wrapped", flush=True)
    env = FrameCapturerWrapper(env)
    print("[DEBUG] FrameCapturer wrapped", flush=True)

    # Convert to gymnasium API (render_mode=None to avoid pyglet issues on Windows)
    env = GymV21CompatibilityV0(env=env, render_mode=None)
    print(f"[DEBUG] Gymnasium API applied (render_mode=None, using OpenCV display)", flush=True)

    # Stack frames
    env = FrameStackGym(env, n=N_STACK)
    print("[DEBUG] FrameStack wrapped", flush=True)

    return env


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    resume_path: str | None = None,
    display_frames: bool = False,
    total_epochs: int = TOTAL_EPOCHS,
):
    """Train the Doya modular brain on Super Mario Bros.

    Online TD(λ) learning — each step updates weights via eligibility traces.
    
    Args:
        resume_path: Path to checkpoint to resume from
        display_frames: Display game with OpenCV window
        total_epochs: Total training epochs
    """
    print("[DEBUG] Starting train()...", flush=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ── Build environment ──
    print("[DEBUG] Building environment...", flush=True)
    env = make_env()
    print("[DEBUG] Environment created!", flush=True)

    # ── Build brain ──
    brain = MarioBrain(
        n_actions=N_ACTIONS,
        feature_dim=256,
        gamma=0.9,
        temperature_start=2.5,
        temperature_end=0.3,
        temperature_anneal_epochs=800,
        icm_weight=0.1,
        device=device,
    )

    # ── Resume if requested ──
    start_epoch = 0
    best_reward = -np.inf
    if resume_path:
        try:
            brain.load(resume_path)
            print(f"Loaded checkpoint: {resume_path}")
            # Try to read last epoch from metrics CSV
            if METRICS_CSV.exists():
                with open(METRICS_CSV, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        start_epoch = int(rows[-1]['epoch'])
                        best_reward = float(rows[-1].get('best_reward', -np.inf))
                        print(f"  → resuming at epoch {start_epoch}, best_reward={best_reward:.1f}")
        except Exception as e:
            print(f"Warning: could not load checkpoint: {e}")
            print("Starting fresh.")

    total_timesteps = total_epochs * STEPS_PER_EPOCH

    # ── Print header ──
    print()
    print("─" * 80)
    print(f"  Doya (1999) Modular Brain — Super Mario Bros")
    print(f"  Visual Cortex CNN (256D) → BG TD(λ) + Cerebellum ICM")
    print(f"  Actions: {N_ACTIONS} (SIMPLE_MOVEMENT)")
    print(f"  Epochs: {total_epochs}  Steps/epoch: {STEPS_PER_EPOCH:,}")
    print(f"  Total steps: {total_timesteps:,}")
    if display_frames:
        print(f"  Display: ON (press ESC to stop)")
    print("─" * 80)
    print(
        f"{'epoch':>6}  {'reward':>8}  {'x_pos':>7}  "
        f"{'DA':>6}  {'fwd':>7}  {'inv':>7}  "
        f"{'recon':>7}  {'temp':>5}  {'fps':>5}  {'best':>8}"
    )
    print("─" * 80)

    # ── Metrics CSV header ──
    csv_header_written = METRICS_CSV.exists() and start_epoch > 0
    if not csv_header_written:
        with open(METRICS_CSV, 'w') as f:
            f.write(
                "epoch,timesteps,mean_reward,mean_x_pos,mean_delta,"
                "mean_forward,mean_inverse,mean_recon,temperature,"
                "fps,best_reward,action_entropy\n"
            )

    # ── Training loop ──
    ep_rewards = deque(maxlen=20)
    ep_x_pos = deque(maxlen=20)
    ep_lengths = deque(maxlen=20)
    ep_metrics = deque(maxlen=20)

    global_step = start_epoch * STEPS_PER_EPOCH
    current_epoch = start_epoch
    epoch_t0 = time.perf_counter()
    steps_in_epoch = 0

    try:
        while current_epoch < total_epochs:
            # ── Episode loop ──
            obs, info = env.reset()
            brain.reset_episode()

            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                # Select action
                action = brain.observe_and_act(obs, explore=True)

                # Execute in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Learn from transition
                step_metrics = brain.learn_from_transition(
                    next_obs, reward, done, info
                )

                episode_reward += reward
                episode_steps += 1
                global_step += 1
                steps_in_epoch += 1

                obs = next_obs

                # Display
                if display_frames and cv2 is not None:
                    frame = get_display_frame()
                    if frame is not None and len(frame.shape) >= 2:
                        h, w = frame.shape[:2]
                        scale = max(1, 480 // h)
                        frame_display = cv2.resize(
                            frame, (w * scale, h * scale),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                        text = f"Reward: {mean_r:.1f} | Epoch: {current_epoch}"
                        cv2.putText(frame_display, text, (10, 30),
                                    font, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Mario Doya Training", frame_display)
                        cv2.waitKey(1)  # No key checking during training - no slowdown

                # ── Epoch boundary ──
                if steps_in_epoch >= STEPS_PER_EPOCH:
                    current_epoch += 1
                    brain.set_temperature(current_epoch)

                    elapsed = time.perf_counter() - epoch_t0
                    fps = steps_in_epoch / elapsed if elapsed > 0 else 0.0
                    epoch_t0 = time.perf_counter()
                    steps_in_epoch = 0

                    # Aggregate metrics
                    mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
                    mean_x = float(np.mean(ep_x_pos)) if ep_x_pos else 0.0

                    # Module metrics from recent episodes
                    mean_da = 0.0
                    mean_fwd = 0.0
                    mean_inv = 0.0
                    mean_recon = 0.0
                    mean_entropy = 0.0
                    if ep_metrics:
                        mean_da = float(np.mean([m['mean_delta'] for m in ep_metrics]))
                        mean_fwd = float(np.mean([m['mean_forward_error'] for m in ep_metrics]))
                        mean_inv = float(np.mean([m['mean_inverse_error'] for m in ep_metrics]))
                        mean_recon = float(np.mean([m['mean_recon_error'] for m in ep_metrics]))
                        mean_entropy = float(np.mean([m['action_entropy'] for m in ep_metrics]))

                    temp = brain.get_temperature()

                    # Best model
                    new_best = mean_r > best_reward and len(ep_rewards) >= 5
                    if new_best:
                        best_reward = mean_r
                        brain.save(str(BEST_MODEL_PREFIX))

                    # Always save latest
                    brain.save(str(LATEST_MODEL_PREFIX))

                    # Print every epoch (for testing)
                    star = " ★" if new_best else ""
                    print(
                        f"{current_epoch:6d}  {mean_r:8.1f}  {mean_x:7.0f}  "
                        f"{mean_da:6.3f}  {mean_fwd:7.4f}  {mean_inv:7.4f}  "
                        f"{mean_recon:7.4f}  {temp:5.2f}  {fps:5.0f}  "
                        f"{best_reward:8.1f}{star}",
                        flush=True
                    )

                    # CSV
                    with open(METRICS_CSV, 'a') as f:
                        f.write(
                            f"{current_epoch},{global_step},{mean_r:.4f},"
                            f"{mean_x:.1f},{mean_da:.6f},{mean_fwd:.6f},"
                            f"{mean_inv:.6f},{mean_recon:.6f},{temp:.3f},"
                            f"{fps:.1f},{best_reward:.4f},{mean_entropy:.4f}\n"
                        )

                    if current_epoch >= total_epochs:
                        break

            # ── Episode ended ──
            ep_summary = brain.end_episode()
            ep_rewards.append(episode_reward)
            ep_lengths.append(episode_steps)
            ep_x_pos.append(ep_summary.get('final_x_pos', 0.0))
            ep_metrics.append(ep_summary)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        brain.save(str(LATEST_MODEL_PREFIX))

    finally:
        if display_frames and cv2 is not None:
            cv2.destroyAllWindows()
        env.close()

    print()
    print(f"Training complete!  Best reward: {best_reward:.1f}")
    print(f"  Best model   → {BEST_MODEL_PREFIX}_*.pt")
    print(f"  Latest model → {LATEST_MODEL_PREFIX}_*.pt")
    print(f"  Metrics CSV  → {METRICS_CSV}")


# ─── Play ─────────────────────────────────────────────────────────────────────

def play(model_path: str):
    """Load a saved model and watch it play."""
    if cv2 is None:
        print("Error: opencv-python is required for play mode.")
        print("Install with: pip install opencv-python")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    brain = MarioBrain(
        n_actions=N_ACTIONS,
        feature_dim=256,
        device=device,
    )
    brain.load(model_path)
    print(f"Loaded model: {model_path}")

    env = make_env()

    print("Watching agent play… Press ESC to quit.\n")
    episode = 0

    try:
        while True:
            obs, info = env.reset()
            brain.reset_episode()
            done = False
            total_r = 0.0
            steps = 0
            episode += 1

            while not done:
                action = brain.observe_and_act(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_r += reward
                steps += 1

                # Display
                frame = get_display_frame()
                if frame is not None and len(frame.shape) >= 2:
                    h, w = frame.shape[:2]
                    scale = max(1, 480 // h)
                    frame_display = cv2.resize(
                        frame, (w * scale, h * scale),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"Ep: {episode} | Step: {steps} | Reward: {total_r:.1f}"
                    cv2.putText(frame_display, text, (10, 30),
                                font, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Mario Doya Agent", frame_display)
                    key = cv2.waitKey(33) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt

            flag = "  FLAG!" if info.get("flag_get") else ""
            print(
                f"Ep {episode:3d}  |  reward={total_r:8.1f}"
                f"  steps={steps:5d}  x_pos={info.get('x_pos', '?')}{flag}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cv2.destroyAllWindows()
        env.close()


# ─── Benchmark: compare Doya vs PPO ──────────────────────────────────────────

def benchmark_strategies(model_path: str, n_trials: int = 20):
    """Test different action selection strategies with the loaded model.

    Strategies:
      1. BG policy (greedy) — basal ganglia actor, no exploration
      2. BG policy (stochastic) — with low temperature
      3. Inverse model — cerebellar skill encapsulation
      4. Forward model lookahead — 1-step predictive selection
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    brain = MarioBrain(
        n_actions=N_ACTIONS,
        feature_dim=256,
        device=device,
    )
    brain.load(model_path)
    print(f"Loaded model: {model_path}")

    env = make_env()

    strategies = {
        'bg_greedy': lambda feat: brain.basal_ganglia.policy(feat, explore=False),
        'bg_stochastic': lambda feat: brain.basal_ganglia.policy(feat, explore=True),
    }

    print()
    print("=" * 60)
    print("  Strategy Benchmark")
    print("=" * 60)

    for name, policy_fn in strategies.items():
        rewards = []
        x_positions = []

        for trial in range(n_trials):
            obs, info = env.reset()
            brain.reset_episode()
            features = brain.visual_cortex.extract_features(obs)

            total_r = 0.0
            done = False
            while not done:
                action = policy_fn(features)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_r += reward
                features = brain.visual_cortex.extract_features(obs)

            rewards.append(total_r)
            x_positions.append(info.get('x_pos', 0))

        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        mean_x = np.mean(x_positions)
        print(
            f"  {name:<20s}  reward={mean_r:7.1f} ± {std_r:5.1f}"
            f"  x_pos={mean_x:6.0f}  max_x={np.max(x_positions):.0f}"
        )

    env.close()
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mario Doya — Modular brain architecture for Super Mario Bros.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train", action="store_true",
        help="Train the modular brain",
    )
    mode.add_argument(
        "--play", type=str, metavar="MODEL_PREFIX",
        help="Watch a saved model play (e.g. checkpoints_doya/best)",
    )
    mode.add_argument(
        "--benchmark", type=str, metavar="MODEL_PREFIX",
        help="Benchmark different strategies with a saved model",
    )
    parser.add_argument(
        "--resume", type=str, metavar="MODEL_PREFIX",
        help="Resume training from a checkpoint prefix",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Display game window during training",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable human-mode rendering (shows actual game window)",
    )
    parser.add_argument(
        "--epochs", type=int, default=TOTAL_EPOCHS,
        help=f"Number of training epochs (default: {TOTAL_EPOCHS})",
    )
    args = parser.parse_args()

    if args.train:
        train(
            resume_path=args.resume,
            display_frames=args.display or args.render,  # --render enables display
            total_epochs=args.epochs,
        )
    elif args.play:
        play(args.play)
    elif args.benchmark:
        benchmark_strategies(args.benchmark)
