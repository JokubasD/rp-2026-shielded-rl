"""Sequential sweep of pure-RL configurations on the rented GPU box.

Each run writes:
  runs/<label>_seed<N>_<ts>/
    config.txt          - all hyperparameters used
    train.log           - captured stdout (parseable for graphs)
    ppo_model.zip       - trained policy
    vecnormalize.pkl    - normalization stats for replay

Designed for hands-off operation inside tmux. Continues to next run on
any failure so a single bad config doesn't waste the whole queue.

Imports primitives from src.rl only; does not modify them.
"""
import gc
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, VecFrameStack, VecNormalize,
)

from src.constants import MapConfig
from src.rl.env import SaREnv, N_STACK
from src.rl.reward import RewardWeights
from src.rl.train import GridCNN


class Tee:
    """Mirror writes to multiple streams (live stdout + per-run log file)."""
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


EASY_CONFIG = MapConfig(
    num_rooms=3, num_victims=2, num_agents=0,
    min_room_width=4, max_room_width=6,
    min_room_length=4, max_room_length=6,
    max_tunnel_thickness=1,
    initial_fire_points=0, fire_spread_rate=0.0,
    room_vulnerability_probability=0.0,
    tunnel_vulnerability_probability=0.0,
)

MEDIUM_CONFIG = MapConfig(
    num_rooms=5, num_victims=3, num_agents=0,
    min_room_width=5, max_room_width=8,
    min_room_length=5, max_room_length=8,
    max_tunnel_thickness=2,
    initial_fire_points=1, fire_spread_rate=0.05, fire_duration=8,
    room_vulnerability_probability=0.15, room_vulnerability_severity=0.3,
    tunnel_vulnerability_probability=0.1, tunnel_vulnerability_severity=0.3,
)


def run_one(label, seed, total_timesteps=2_000_000, n_envs=32,
            n_stack=N_STACK, ent_coef=0.05,
            reward_weights=None, env_kwargs_override=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"{label}_seed{seed}_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "label": label, "seed": seed, "total_timesteps": total_timesteps,
        "n_envs": n_envs, "n_stack": n_stack, "ent_coef": ent_coef,
        "reward_weights": asdict(reward_weights or RewardWeights()),
        "env_kwargs_override": str(env_kwargs_override or {}),
    }
    (out / "config.txt").write_text(repr(cfg))

    log_file = open(out / "train.log", "w", buffering=1)
    orig = sys.stdout
    sys.stdout = Tee(orig, log_file)
    try:
        print(f"[sweep] config: {cfg}", flush=True)

        env_kwargs = {}
        if reward_weights is not None:
            env_kwargs["reward_weights"] = reward_weights
        if env_kwargs_override:
            env_kwargs.update(env_kwargs_override)

        vec_env = make_vec_env(SaREnv, n_envs=n_envs,
                               vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
        if n_stack > 1:
            vec_env = VecFrameStack(vec_env, n_stack=n_stack,
                                    channels_order="first")

        model = PPO(
            "MlpPolicy", vec_env,
            policy_kwargs=dict(
                features_extractor_class=GridCNN,
                features_extractor_kwargs=dict(features_dim=256),
                normalize_images=False,
            ),
            n_steps=1024, batch_size=64,
            ent_coef=ent_coef, verbose=1, seed=seed, device="auto",
        )

        t0 = time.time()
        try:
            model.learn(total_timesteps=total_timesteps)
            model.save(out / "ppo_model")
            vec_env.save(out / "vecnormalize.pkl")
            print(f"[sweep] {label} seed={seed} done in "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        finally:
            try:
                vec_env.close()
            except Exception:
                pass
            del model
            del vec_env
            gc.collect()
    finally:
        sys.stdout = orig
        log_file.close()


# === The sweep =====================================================
# Trimmed to 4 essential runs (~6h, ~$4): difficulty ladder + one main
# + one ablation. Any run that fails just gets skipped.
SWEEP = [
    # 1. Easy 20x20 - small env, no hazards. Validates the algorithm.
    ("easy_20x20", 0, {
        "total_timesteps": 500_000,
        "env_kwargs_override": {
            "width": 20, "height": 20, "max_episode_steps": 150,
            "config": EASY_CONFIG,
        },
    }),
    # 2. Medium 30x30 - stepping stone with light hazards.
    ("medium_30x30", 0, {
        "total_timesteps": 1_500_000,
        "env_kwargs_override": {
            "width": 30, "height": 30, "max_episode_steps": 200,
            "config": MEDIUM_CONFIG,
        },
    }),
    # 3. Main 40x40 - headline result.
    ("main_40x40", 0, {}),
    # 4. Ablation: novelty bonus off. Shows that intrinsic motivation matters.
    ("ablate_no_novelty", 0, {
        "reward_weights": RewardWeights(w_novelty=0.0),
    }),
]


def main():
    total = len(SWEEP)
    t_total = time.time()
    print(f"[sweep] {total} runs queued", flush=True)
    for i, (label, seed, kwargs) in enumerate(SWEEP, 1):
        print(f"\n{'='*70}\n[sweep] run {i}/{total}: {label} seed={seed}\n"
              f"{'='*70}", flush=True)
        try:
            run_one(label, seed, **kwargs)
        except Exception:
            import traceback
            print(f"[sweep] {label} seed={seed} FAILED:", flush=True)
            traceback.print_exc()
            print("[sweep] continuing with next run", flush=True)
    print(f"\n[sweep] ALL DONE in {(time.time()-t_total)/3600:.2f} h",
          flush=True)


if __name__ == "__main__":
    main()
