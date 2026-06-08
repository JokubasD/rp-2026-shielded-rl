"""v5 sweep: STABILISED PPO + global coverage reward on the nice room-maps.

Two evidence-backed changes (from a 5-way design review of the v4 results):

  1. PPO STABILITY. The v4 runs were a trust-region runaway (approx_kl up to ~0.1,
     clip_fraction ~0.4-0.6 -> climb-then-regress). Fix: target_kl=0.02, n_epochs=4,
     linear LR decay to 0, large batch, value clipping, wider value head, ent anneal.
     This turns "climb then regress" into "climb and hold", so the best-by-success
     checkpoint actually captures the peak.

  2. GLOBAL COVERAGE REWARD. Replace the myopic nearest-frontier potential with a
     coverage potential Phi = -w_phi*(1-coverage) (monotone in the true objective:
     full coverage => all victims scanned), zero the farmable novelty/explore terms,
     and add a terminal coverage-fraction bonus so partial coverage still teaches.

Maps come from src/rl/configs.py (sar_config): distinct rooms + corridors. Honest
scope: >80% on victims_found_frac is realistic here; >80% full success_rate is
realistic on 20x20 / with fewer victims, harder on 30x30. We log BOTH.

Each run writes runs/<label>_seed<N>_<ts>/ with config.txt, train.log, best_model.zip
(peak by success), checkpoints/, ppo_model.zip, vecnormalize.pkl.
Usage: python run_sweep_v5.py            # 20 -> 25 -> 30
       python run_sweep_v5.py --quick    # 20x20 only (fast confidence check)
"""
import gc
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from src.rl.env import SaREnv, N_STACK
from src.rl.reward import RewardWeights
from src.rl.train import GridCNN
from src.rl.callbacks import SuccessRateCallback, EntCoefAnneal
from src.rl.configs import sar_config


# Env-var knobs for fast tuning, e.g. HORIZON=500 W_PHI=15 NUM_VICTIMS=4 python run_sweep_v5.py --test
NUM_VICTIMS = int(os.environ.get("NUM_VICTIMS", "6"))
N_ENVS = int(os.environ.get("N_ENVS", "32"))       # parallel envs (set 64 on a 64-core box)
W_PHI = float(os.environ.get("W_PHI", "8"))         # coverage-potential strength
W_COV = float(os.environ.get("W_COV_TERM", "20"))   # terminal coverage bonus
HORIZON = int(os.environ.get("HORIZON", "0"))       # >0 overrides the per-map episode budget
TEST_STEPS = int(os.environ.get("TEST_STEPS", "1500000"))  # steps for a --test run

# v5 reward: global coverage potential + terminal coverage bonus; farmable terms off.
V5_REWARD = RewardWeights(
    w_v=10.0, w_e=0.0, w_s=0.005, w_c_t=0.1, w_c_v=1.0,
    w_h_v=0.0, w_h_f_flam=0.0, w_h_f_burn=0.0,
    w_succ=50.0, w_tout=0.0, w_novelty=0.0,
    w_phi=W_PHI, w_cov_term=W_COV, use_coverage_potential=True,
)

MONITOR_KWARGS = dict(info_keywords=("outcome", "victims_found", "total_victims"))


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def lin(start, end):
    """SB3 schedule callable: progress_remaining goes 1 -> 0."""
    return lambda pr: end + pr * (start - end)


def make_ppo(vec_env, seed):
    """PPO with the anti-divergence settings that fix the v4 trust-region runaway."""
    return PPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # wider heads (was 64,64)
            ortho_init=True,
        ),
        learning_rate=lin(2.5e-4, 0.0),  # decay to 0 -> settles instead of oscillating
        n_steps=1024, batch_size=2048, n_epochs=4,  # fewer epochs + big batch = smaller steps
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.15, clip_range_vf=0.2,          # tighter trust region + clipped value
        ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5,
        target_kl=0.02,                              # THE fix: early-stop runaway updates
        normalize_advantage=True,
        verbose=1, seed=seed, device="auto",
    )


def run_one(label, seed, total_timesteps, size, max_episode_steps,
            num_victims=NUM_VICTIMS, n_envs=N_ENVS, n_stack=N_STACK):
    if HORIZON > 0:
        max_episode_steps = HORIZON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"{label}_seed{seed}_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = dict(label=label, seed=seed, total_timesteps=total_timesteps,
               size=size, max_episode_steps=max_episode_steps, num_victims=num_victims,
               n_envs=n_envs, n_stack=n_stack, reward=asdict(V5_REWARD), stable=True)
    (out / "config.txt").write_text(repr(cfg))

    log_file = open(out / "train.log", "w", buffering=1)
    orig = sys.stdout
    sys.stdout = Tee(orig, log_file)
    try:
        print(f"[v5] config: {cfg}", flush=True)
        env_kwargs = dict(width=size, height=size, max_episode_steps=max_episode_steps,
                          config=sar_config(size, num_victims=num_victims),
                          reward_weights=V5_REWARD)
        vec_env = make_vec_env(SaREnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                               env_kwargs=env_kwargs, monitor_kwargs=MONITOR_KWARGS)
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
        vnorm = vec_env
        if n_stack > 1:
            vec_env = VecFrameStack(vec_env, n_stack=n_stack, channels_order="first")

        model = make_ppo(vec_env, seed)
        callbacks = CallbackList([
            SuccessRateCallback(best_save_dir=out, vecnormalize=vnorm, verbose=1),
            EntCoefAnneal(start=0.03, end=0.005, total_timesteps=total_timesteps),
            CheckpointCallback(save_freq=max(1, 250_000 // n_envs),
                               save_path=str(out / "checkpoints"), name_prefix="ppo"),
        ])
        t0 = time.time()
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
            model.save(out / "ppo_model")
            vnorm.save(str(out / "vecnormalize.pkl"))
            print(f"[v5] {label} seed={seed} done in {(time.time()-t0)/60:.1f} min", flush=True)
        finally:
            try:
                vec_env.close()
            except Exception:
                pass
            del model, vec_env
            gc.collect()
    finally:
        sys.stdout = orig
        log_file.close()


# Horizons sized so a competent agent can cover the whole map (coverage = success).
SWEEP = [
    dict(label="v5_20x20", seed=0, total_timesteps=3_000_000, size=20, max_episode_steps=300),
    dict(label="v5_25x25", seed=0, total_timesteps=3_000_000, size=25, max_episode_steps=400),
    dict(label="v5_30x30", seed=0, total_timesteps=4_000_000, size=30, max_episode_steps=500),
]
QUICK = SWEEP[:1]  # 20x20 only -- the fast confidence check
# Fast single 20x20 run for hyperparameter experiments (~5 min at 1.5M steps).
TEST = [dict(label="v5_test", seed=0, total_timesteps=TEST_STEPS, size=20, max_episode_steps=300)]


def main(sweep=SWEEP):
    t_total = time.time()
    print(f"[v5] {len(sweep)} runs queued", flush=True)
    for i, st in enumerate(sweep, 1):
        print(f"\n{'='*70}\n[v5] run {i}/{len(sweep)}: {st['label']}\n{'='*70}", flush=True)
        try:
            run_one(**st)
        except Exception:
            import traceback
            print(f"[v5] {st['label']} FAILED:", flush=True)
            traceback.print_exc()
            print("[v5] continuing with next run", flush=True)
    print(f"\n[v5] ALL DONE in {(time.time()-t_total)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    if "--test" in sys.argv:
        main(TEST)
    elif "--quick" in sys.argv:
        main(QUICK)
    else:
        main(SWEEP)
