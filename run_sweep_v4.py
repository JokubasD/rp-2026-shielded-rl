"""v4 sweep: make the pure-RL agent actually COMPLETE rescues.

Three changes over v3 (run_sweep_v3.py), all literature- or measurement-justified:
  1. Longer episode horizon. An empirical audit showed that on the 40x40 map a
     realistic blind searcher needs a median ~266 / p90 ~423 steps to scan all 3
     victims, so the old 300-step cap made ~50% of maps unsolvable. Horizons are
     raised to easy=300, medium=500, main=750.
  2. Directed exploration via potential-based frontier shaping (w_phi=10.0,
     Ng-Harada-Russell 1999). Provably optimality-preserving, so the baseline
     stays "pure". Computed from the agent's own belief (src/rl/frontier.py).
  3. Success logging (SuccessRateCallback + Monitor info_keywords) so we can SEE
     rescue success rate / victims-found fraction, not just reward.

The last run is a w_phi=0 ablation: identical to v4_main except shaping off, to
isolate the effect of the frontier term.

Each run writes runs/v4_<label>_seed<N>_<ts>/ with config.txt, train.log,
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
from src.rl.callbacks import SuccessRateCallback, EntCoefAnneal
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


# === v4 REWARD WEIGHTS =============================================
# v3 weights + the new directed-exploration term w_phi=10.0. Everything else is
# identical to v3, so v4_main vs v4_main_no_shaping isolates the shaping effect.
V4_REWARD = RewardWeights(
    w_v=30.0,
    w_e=15.0,
    w_s=0.005,
    w_c_t=0.1,
    w_c_v=1.0,
    w_h_v=0.0,
    w_h_f_flam=0.0,
    w_h_f_burn=0.0,
    w_succ=100.0,
    w_tout=0.0,
    w_novelty=0.5,
    w_phi=10.0,      # <-- NEW: potential-based frontier shaping
)

# Surface the task-progress metrics through the Monitor so the callback can log them.
MONITOR_KWARGS = dict(info_keywords=("outcome", "victims_found", "total_victims"))


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
        print(f"[sweep v4] config: {cfg}", flush=True)

        env_kwargs = {}
        if reward_weights is not None:
            env_kwargs["reward_weights"] = reward_weights
        if env_kwargs_override:
            env_kwargs.update(env_kwargs_override)

        vec_env = make_vec_env(SaREnv, n_envs=n_envs,
                               vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs,
                               monitor_kwargs=MONITOR_KWARGS)
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
        vnorm = vec_env  # handle for saving normalization stats
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

        callbacks = CallbackList([
            # Logs success_rate / victims_found_frac AND saves best_model by
            # SUCCESS (reward and success diverge, so reward-based selection is wrong).
            SuccessRateCallback(best_save_dir=out, vecnormalize=vnorm, verbose=1),
            # Explore early, commit late -> reduces the late-training regression.
            EntCoefAnneal(start=ent_coef, end=0.01, total_timesteps=total_timesteps),
            # Periodic safety checkpoints (every ~250k steps).
            CheckpointCallback(save_freq=max(1, 250_000 // n_envs),
                               save_path=str(out / "checkpoints"), name_prefix="ppo"),
        ])

        t0 = time.time()
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
            model.save(out / "ppo_model")             # final model
            vnorm.save(str(out / "vecnormalize.pkl"))  # final normalization stats
            print(f"[sweep v4] {label} seed={seed} done in "
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


# === The v4 sweep ====================================================
# Horizons raised per the audit; total_timesteps scaled up because longer
# episodes mean fewer terminations per update. Tune the step counts to budget;
# the main run is the headline, the ablation isolates the shaping term.
SWEEP = [
    # 1. v4 easy 20x20 - sanity that shaping does not hurt the already-working case.
    ("v4_easy_20x20", 0, {
        "total_timesteps": 750_000,
        "reward_weights": V4_REWARD,
        "env_kwargs_override": {
            "width": 20, "height": 20, "max_episode_steps": 300,
            "config": EASY_CONFIG,
        },
    }),
    # 2. v4 medium 30x30 - longer horizon + shaping.
    ("v4_medium_30x30", 0, {
        "total_timesteps": 2_000_000,
        "reward_weights": V4_REWARD,
        "env_kwargs_override": {
            "width": 30, "height": 30, "max_episode_steps": 500,
            "config": MEDIUM_CONFIG,
        },
    }),
    # 3. v4 main 40x40 - HEADLINE. 750-step horizon, shaping on.
    #    Consider 5-6M steps for the final paper run if budget allows.
    ("v4_main_40x40", 0, {
        "total_timesteps": 3_000_000,
        "reward_weights": V4_REWARD,
        "env_kwargs_override": {"max_episode_steps": 750},
    }),
    # 4. v4 main ABLATION: shaping off (w_phi=0), else identical to run 3.
    #    Isolates the contribution of the frontier potential.
    ("v4_main_no_shaping", 0, {
        "total_timesteps": 1_500_000,
        "reward_weights": replace(V4_REWARD, w_phi=0.0),
        "env_kwargs_override": {"max_episode_steps": 750},
    }),
]


# Short ~5-6h profile to check for signal before committing to the full sweep.
# Skips easy (already validated locally) and the ablation; runs medium then 40x40.
QUICK_SWEEP = [
    ("v4q_medium_30x30", 0, {
        "total_timesteps": 1_500_000,
        "reward_weights": V4_REWARD,
        "env_kwargs_override": {
            "width": 30, "height": 30, "max_episode_steps": 500,
            "config": MEDIUM_CONFIG,
        },
    }),
    ("v4q_main_40x40", 0, {
        "total_timesteps": 2_000_000,
        "reward_weights": V4_REWARD,
        "env_kwargs_override": {"max_episode_steps": 750},
    }),
]


def main(sweep=SWEEP):
    total = len(sweep)
    t_total = time.time()
    print(f"[sweep v4] {total} runs queued", flush=True)
    for i, (label, seed, kwargs) in enumerate(sweep, 1):
        print(f"\n{'='*70}\n[sweep v4] run {i}/{total}: {label} seed={seed}\n"
              f"{'='*70}", flush=True)
        try:
            run_one(label, seed, **kwargs)
        except Exception:
            import traceback
            print(f"[sweep v4] {label} seed={seed} FAILED:", flush=True)
            traceback.print_exc()
            print("[sweep v4] continuing with next run", flush=True)
    print(f"\n[sweep v4] ALL DONE in {(time.time()-t_total)/3600:.2f} h",
          flush=True)


if __name__ == "__main__":
    import sys
    main(QUICK_SWEEP if "--quick" in sys.argv else SWEEP)
