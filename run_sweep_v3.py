"""v3 sweep: REWARD-ONLY fix to escape the negative-reward trap observed in v2.

v2 -> v3 changes (literature-justified):
  w_tout:    15.0 -> 0.0    (Skalse 2022 -- removes "trying = bad" trap)
  w_novelty:  0.3 -> 0.5    (Bellemare 2016 -- match intrinsic-extrinsic magnitude)
  w_succ:    50.0 -> 100.0  (stronger completion-success gradient pull)

ALL PPO hyperparameters identical to v2. Reward-only ablation: any improvement
is attributable solely to the reward fix.

Each run writes to runs/v3_<label>_seed<N>_<ts>/ with config.txt, train.log,
ppo_model.zip, vecnormalize.pkl.
"""
import gc
import sys
import time
from dataclasses import asdict, replace
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
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# === v3 REWARD WEIGHTS ============================================
# Reward-only change from v2. Same task-only structure, but:
#   - timeout penalty zeroed (was 15.0)            -> Skalse 2022
#   - novelty bumped 0.3 -> 0.5                    -> Bellemare 2016
#   - success bonus doubled 50.0 -> 100.0           -> stronger task gradient
# All hazard terms stay zeroed (the shield's job in the shielded variant).
V3_REWARD = RewardWeights(
    w_v=30.0,
    w_e=15.0,
    w_s=0.005,
    w_c_t=0.1,
    w_c_v=1.0,
    w_h_v=0.0,
    w_h_f_flam=0.0,
    w_h_f_burn=0.0,
    w_succ=100.0,    # bumped from 50.0
    w_tout=0.0,      # zeroed from 15.0  <-- key fix
    w_novelty=0.5,   # bumped from 0.3
)


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


# === The v3 sweep ====================================================
SWEEP = [
    # 1. v3 easy 20x20 - direct A/B comparison vs v2_easy_20x20.
    ("v3_easy_20x20", 0, {
        "total_timesteps": 500_000,
        "reward_weights": V3_REWARD,
        "env_kwargs_override": {
            "width": 20, "height": 20, "max_episode_steps": 150,
            "config": EASY_CONFIG,
        },
    }),
    # 2. v3 medium 30x30 - this is the one that collapsed under v2 (-2.8 reward).
    #    Under v3, expect positive baseline + gradient toward completion.
    ("v3_medium_30x30", 0, {
        "total_timesteps": 1_500_000,
        "reward_weights": V3_REWARD,
        "env_kwargs_override": {
            "width": 30, "height": 30, "max_episode_steps": 200,
            "config": MEDIUM_CONFIG,
        },
    }),
    # 3. v3 main 40x40 - the headline result on the full task.
    ("v3_main_40x40", 0, {
        "total_timesteps": 2_000_000,
        "reward_weights": V3_REWARD,
    }),
    # 4. v3 main ablation: novelty bonus disabled. Should be visibly worse than
    #    v3_main_40x40 -- demonstrates the count-based bonus matters.
    ("v3_main_no_novelty", 0, {
        "total_timesteps": 1_000_000,
        "reward_weights": replace(V3_REWARD, w_novelty=0.0),
    }),
]


def main():
    total = len(SWEEP)
    t_total = time.time()
    print(f"[sweep v3] {total} runs queued", flush=True)
    for i, (label, seed, kwargs) in enumerate(SWEEP, 1):
        print(f"\n{'='*70}\n[sweep v3] run {i}/{total}: {label} seed={seed}\n"
              f"{'='*70}", flush=True)
        try:
            run_one(label, seed, **kwargs)
        except Exception:
            import traceback
            print(f"[sweep v3] {label} seed={seed} FAILED:", flush=True)
            traceback.print_exc()
            print("[sweep v3] continuing with next run", flush=True)
    print(f"\n[sweep v3] ALL DONE in {(time.time()-t_total)/3600:.2f} h",
          flush=True)


if __name__ == "__main__":
    main()
