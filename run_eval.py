"""Held-out evaluation: freeze a trained policy, run it on N unseen maps, report stats.

This replaces training-rollout statistics with a fair, deployment-style evaluation:
the policy is fixed (no learning), maps are generated from a seed range disjoint
from training, the agent always starts from the fixed entrance, and every agent is
measured under the identical protocol so the three-way comparison is controlled.

Per-episode metrics (coverage, victims-found, success, collisions, damage, steps,
decision time) are aggregated to mean +/- 95% CI and written to runs/eval/.

Usage:
  uv run python run_eval.py runs/lstm_<ts>/L1_g0999_h1000_rs1 --lstm --label LSTM --size 20 --victims 6 --horizon 1000 --episodes 300
  uv run python run_eval.py runs/batch_<ts>/08_g0999_rs1_h1000 --label PPO --size 20 --victims 6 --horizon 1000 --episodes 300
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.rl.env import SaREnv, N_STACK
from src.rl.reward import RewardWeights
from src.rl.train import GridCNN  # noqa: F401  (needed so the custom extractor unpickles)
from src.rl.configs import sar_config

EVAL_SEED = 777_000  # held-out seed stream, disjoint from training seeds (0,1,2)


def find_model(path: Path):
    """Accept either a .zip or a run dir; prefer best_model.zip, else ppo/rppo_model.zip."""
    if path.suffix == ".zip":
        return path
    for name in ("best_model.zip", "rppo_model.zip", "ppo_model.zip"):
        if (path / name).exists():
            return path / name
    sys.exit(f"No model .zip found in {path}")


def ci95(xs):
    """Mean and half-width of the 95% confidence interval."""
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    half = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a model .zip or a run directory")
    ap.add_argument("--lstm", action="store_true", help="model is RecurrentPPO (no frame-stack)")
    ap.add_argument("--label", default="agent")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--victims", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--deterministic", action="store_true", help="argmax actions (default: sample)")
    ap.add_argument("--seed", type=int, default=EVAL_SEED)
    args = ap.parse_args()

    model_path = find_model(Path(args.model))
    reward = RewardWeights(w_phi=30.0, w_cov_term=50.0, use_coverage_potential=True)
    env_kwargs = dict(width=args.size, height=args.size, max_episode_steps=args.horizon,
                      config=sar_config(args.size, num_victims=args.victims, corridor=1),
                      reward_weights=reward, random_start=False)  # fixed entrance = deployment

    info_kw = ("outcome", "victims_found", "total_victims", "area_explored",
               "terrain_collisions", "victim_collisions", "damage", "steps_taken")
    venv = DummyVecEnv([lambda: Monitor(SaREnv(**env_kwargs), info_keywords=info_kw)])
    venv.seed(args.seed)                         # held-out map stream
    if not args.lstm:
        venv = VecFrameStack(venv, n_stack=N_STACK, channels_order="first")

    Algo = None
    if args.lstm:
        from sb3_contrib import RecurrentPPO
        Algo = RecurrentPPO
    else:
        Algo = PPO
    model = Algo.load(model_path, env=venv, device="auto")
    print(f"[eval] {args.label}: {model_path}  | {args.episodes} held-out maps "
          f"({args.size}x{args.size}, {args.victims} victims), "
          f"{'deterministic' if args.deterministic else 'stochastic'}", flush=True)

    rows = []
    obs = venv.reset()
    lstm_state, ep_start = None, np.ones((1,), dtype=bool)
    t_pred = []
    while len(rows) < args.episodes:
        t0 = time.perf_counter()
        action, lstm_state = model.predict(obs, state=lstm_state, episode_start=ep_start,
                                           deterministic=args.deterministic)
        t_pred.append(time.perf_counter() - t0)
        obs, _, dones, infos = venv.step(action)
        ep_start = dones
        if dones[0]:
            info = infos[0]
            total = int(info.get("total_victims", 0)) or 1
            rows.append(dict(
                coverage=float(info.get("area_explored", 0.0)),
                victims=float(info.get("victims_found", 0)) / total,
                success=1.0 if info.get("outcome") == "success" else 0.0,
                terr_coll=int(info.get("terrain_collisions", 0)),
                vict_coll=int(info.get("victim_collisions", 0)),
                damage=float(info.get("damage", 0)),
                steps=int(info.get("steps_taken", 0)),
            ))
            lstm_state = None  # reset recurrent state between episodes

    out = Path("runs") / "eval"
    out.mkdir(parents=True, exist_ok=True)
    metrics = ["coverage", "victims", "success", "terr_coll", "vict_coll", "damage", "steps"]
    # Per-episode CSV (for plots / error bars later).
    per = [",".join(metrics)] + [",".join(str(round(r[m], 4)) for m in metrics) for r in rows]
    (out / f"{args.label}_episodes.csv").write_text("\n".join(per))

    # Summary row: mean +/- 95% CI per metric.
    summ = {m: ci95([r[m] for r in rows]) for m in metrics}
    dt_ms, dt_ci = ci95([t * 1000 for t in t_pred])
    print(f"\n[eval] {args.label}  (N={len(rows)} held-out maps)")
    print("-" * 52)
    for m in metrics:
        mean, half = summ[m]
        print(f"  {m:<12} {mean:8.4f}  +/- {half:.4f}")
    print(f"  {'decision_ms':<12} {dt_ms:8.4f}  +/- {dt_ci:.4f}")

    line = ",".join([args.label, str(len(rows))] +
                    [f"{summ[m][0]:.4f}" for m in metrics] +
                    [f"{summ[m][1]:.4f}" for m in metrics] + [f"{dt_ms:.4f}"])
    header = ",".join(["label", "N"] + [f"{m}_mean" for m in metrics] +
                      [f"{m}_ci95" for m in metrics] + ["decision_ms"])
    summary = out / "summary.csv"
    if not summary.exists():
        summary.write_text(header + "\n")
    with summary.open("a") as f:
        f.write(line + "\n")
    print(f"\n[eval] appended to {summary}", flush=True)


if __name__ == "__main__":
    main()
