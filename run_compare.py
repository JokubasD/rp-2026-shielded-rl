"""Unified held-out comparison: drive every agent through the SAME Simulator on the
SAME held-out maps, collect identical metrics, aggregate to mean +/- 95% CI.

Agents (all on identical seeded maps, same entrance, same sensing, same metrics):
  rl     - the trained policy (frame-stack PPO or LSTM), acting in the sim
  mpc    - exhaustive-search MPC (sub-project 1)
  tmpc   - tube / SDD-MPC (sub-project 1), unmodified
  shield - SHIELDED RL (this work): take the RL action if its slip-tube is safe,
           otherwise defer to the SDD-MPC backup (tmpc), unmodified.

Usage:
  uv run python run_compare.py --agents mpc,tmpc --size 20 --victims 6 --horizon 300 --episodes 30
  uv run python run_compare.py --agents rl,shield --rl-model runs/lstm_*/L1_g0999_h1000_rs1 --lstm \
      --size 20 --victims 6 --horizon 1000 --episodes 100
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

from src.simulator import Simulator
from src.constants import RunOutcome, AgentAction, TraversabilityLevel, FireLevel, MapConfig
from src.agent import Agent
from src.agents.mpc import MpcAgent
from src.agents.tmpc import TmpcAgent
from src.rl.env import build_observation, N_CHANNELS, N_STACK
from src.rl.configs import sar_config

SCAN = dict(decay=0.01, scan_accuracy=0.9, scan_radius=3, scan_falloff=True)
START = (0, 0)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ---------------------------------------------------------------------------
# RL policy acting inside the simulator (builds obs from belief, runs the model)
# ---------------------------------------------------------------------------
class PolicyController:
    """Wraps a trained SB3 model so it can act on an Agent's belief each step."""

    def __init__(self, model, lstm, size, deterministic):
        self.model = model
        self.lstm = lstm
        self.det = deterministic
        self.size = size
        self.lstm_state = None
        self.ep_start = True
        # frame-stack buffer (oldest first, newest last), matching VecFrameStack.
        self.buffer = None if lstm else np.zeros((N_STACK, N_CHANNELS, size, size), dtype=np.float32)

    def act(self, agent) -> int:
        obs = build_observation(agent)  # (N_CHANNELS, H, W)
        if self.lstm:
            x = obs[None]
            action, self.lstm_state = self.model.predict(
                x, state=self.lstm_state, episode_start=np.array([self.ep_start]),
                deterministic=self.det)
            self.ep_start = False
        else:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = obs
            x = self.buffer.reshape(N_STACK * N_CHANNELS, self.size, self.size)[None]
            action, _ = self.model.predict(x, deterministic=self.det)
        return int(np.asarray(action).reshape(-1)[0])


class PolicyAgent(Agent):
    """Pure-RL agent: always executes the policy's action (clean RL baseline in-sim)."""

    def __init__(self, name, x, y, size, controller):
        super().__init__(name, x, y, size, size, **SCAN)
        self.controller = controller

    def get_action(self) -> AgentAction:
        return AgentAction(self.controller.act(self))


class ShieldedAgent(TmpcAgent):
    """Shielded RL (this work): execute the RL action if its slip-tube is safe,
    else defer to the SDD-MPC backup (TmpcAgent.get_action), which is left untouched."""

    def __init__(self, name, x, y, size, controller, stuck_patience=0):
        super().__init__(name, x, y, size, size, **SCAN)
        self.controller = controller
        self.n_decisions = 0
        self.n_override = 0
        # Completion fallback: after `stuck_patience` steps with no new exploration,
        # latch control to the SDD-MPC backup so it finishes the sweep (0 = disabled).
        self.stuck_patience = stuck_patience
        self._best_explored = 0
        self._stale = 0
        self._mpc_latch = False

    def _in_safe_set(self, x, y) -> bool:
        """A cell is safe if in-grid, traversable (not wall), no victim, not burning (per belief)."""
        if not (0 <= x < self.world_width and 0 <= y < self.world_height):
            return False
        if self.perception.traversability[y][x] != TraversabilityLevel.TRAVERSIBLE:
            return False
        if self.perception.victims[y][x] == 1:
            return False
        return self.perception.fire[y][x] != FireLevel.BURNING

    def _shield_safe(self, action: AgentAction) -> bool:
        """Slip-aware: the target must be safe, and every TRAVERSABLE neighbour the
        agent could slip into must be safe (walls are not slip targets, so excluded)."""
        tx, ty = self._target_cell(action)
        if not self._in_safe_set(tx, ty):
            return False
        for dx, dy in N4:
            nx, ny = tx + dx, ty + dy
            if (0 <= nx < self.world_width and 0 <= ny < self.world_height
                    and self.perception.traversability[ny][nx] == TraversabilityLevel.TRAVERSIBLE
                    and not self._in_safe_set(nx, ny)):
                return False
        return True

    def get_action(self) -> AgentAction:
        self.n_decisions += 1
        # Completion fallback: detect a stall (no newly explored cell for K steps).
        if self.stuck_patience > 0 and not self._mpc_latch:
            explored = int(self.explored.sum())
            if explored > self._best_explored:
                self._best_explored, self._stale = explored, 0
            else:
                self._stale += 1
                if self._stale >= self.stuck_patience:
                    self._mpc_latch = True  # hand off to the SDD-MPC to finish
        if self._mpc_latch:
            self.n_override += 1
            return super().get_action()
        rl_action = AgentAction(self.controller.act(self))
        if self._shield_safe(rl_action):
            return rl_action
        self.n_override += 1
        return super().get_action()  # SDD-MPC backup (unmodified)


def make_agent(kind, size, model, lstm, deterministic, completion, start=START):
    x, y = start
    if kind == "mpc":
        return MpcAgent("mpc", x, y, size, size, **SCAN)
    if kind == "tmpc":
        return TmpcAgent("tmpc", x, y, size, size, **SCAN)
    if kind in ("rl", "shield"):
        ctrl = PolicyController(model, lstm, size, deterministic)
        if kind == "rl":
            return PolicyAgent("rl", x, y, size, ctrl)
        return ShieldedAgent("shield", x, y, size, ctrl, stuck_patience=completion)
    raise ValueError(f"unknown agent kind: {kind}")


def open_config(size, victims):
    """Big-room, wide-tunnel, hazard-free map where the SDD-MPC can navigate."""
    return MapConfig(
        num_rooms=max(5, size // 5), num_victims=victims, num_agents=0,
        unconnected_probability=0.0,
        min_room_width=5, max_room_width=10, min_room_length=5, max_room_length=10,
        min_tunnel_thickness=2, max_tunnel_thickness=3,
        initial_fire_points=0, fire_spread_rate=0.0, fire_duration=0,
        room_vulnerability_probability=0.0, room_vulnerability_severity=0.0,
        tunnel_vulnerability_probability=0.0, tunnel_vulnerability_severity=0.0,
    )


def run_episode(kind, size, victims, horizon, seed, model, lstm, deterministic, corridor, completion, start, open_map):
    sim = Simulator(size, size)
    cfg = open_config(size, victims) if open_map else sar_config(size, num_victims=victims, corridor=corridor)
    sim.generate_ground_truth(cfg, seed=seed)
    agent = make_agent(kind, size, model, lstm, deterministic, completion, start)
    sim.add_agent(agent)

    t0 = time.perf_counter()
    sim.run(horizon)
    elapsed = time.perf_counter() - t0

    curr = sim.metrics.history[-1]
    total = int(curr.total_victims) or 1
    steps = int(getattr(curr, "step", sim.metrics.steps_taken)) or 1
    return dict(
        coverage=float(curr.area_explored.get(agent, 0.0)),
        victims=float(curr.victims_found) / total,
        success=1.0 if curr.outcome == RunOutcome.SUCCESS else 0.0,
        terr_coll=int(curr.terrain_collisions.get(agent, 0)),
        vict_coll=int(curr.victim_collisions.get(agent, 0)),
        damage=float(curr.damage.get(agent, 0.0)),
        steps=steps,
        decision_ms=1000.0 * elapsed / steps,
        override_frac=(agent.n_override / max(1, agent.n_decisions)) if kind == "shield" else 0.0,
    )


def ci95(xs):
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    half = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, half


METRICS = ["coverage", "victims", "success", "terr_coll", "vict_coll", "damage",
           "steps", "decision_ms", "override_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="mpc,tmpc")
    ap.add_argument("--rl-model", default=None, help="run dir or .zip (for rl/shield)")
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--victims", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777_000)
    ap.add_argument("--corridor", type=int, default=1, help="tunnel width (>=2 lets the SDD-MPC navigate)")
    ap.add_argument("--completion", type=int, default=0,
                    help="shield: hand off to SDD-MPC after K steps with no new exploration (0=safety-only)")
    ap.add_argument("--start", default="0,0", help="start cell 'x,y' or 'center' (corner traps the tube MPC)")
    ap.add_argument("--open", action="store_true", help="big-room hazard-free maps where the SDD-MPC navigates")
    args = ap.parse_args()
    start = (args.size // 2, args.size // 2) if args.start == "center" else tuple(int(v) for v in args.start.split(","))

    kinds = [k.strip() for k in args.agents.split(",")]
    model = None
    if any(k in ("rl", "shield") for k in kinds):
        mp = Path(args.rl_model)
        if mp.suffix != ".zip":
            mp = next((mp / n for n in ("best_model.zip", "rppo_model.zip", "ppo_model.zip")
                       if (mp / n).exists()), mp)
        if args.lstm:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(mp, device="auto")
        else:
            from stable_baselines3 import PPO
            model = PPO.load(mp, device="auto")
        print(f"[compare] loaded {'LSTM' if args.lstm else 'PPO'} model: {mp}", flush=True)

    out = Path("runs") / "compare"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.csv"
    header = ",".join(["agent", "N"] + [f"{m}_mean" for m in METRICS] + [f"{m}_ci95" for m in METRICS])
    if not summary.exists():
        summary.write_text(header + "\n")

    for kind in kinds:
        print(f"\n[compare] {kind}: {args.episodes} held-out maps "
              f"({args.size}x{args.size}, {args.victims} victims, horizon {args.horizon})", flush=True)
        rows = []
        for i in range(args.episodes):
            r = run_episode(kind, args.size, args.victims, args.horizon, args.seed + i,
                            model, args.lstm, args.deterministic, args.corridor, args.completion, start, args.open)
            rows.append(r)
            print(f"  ep {i+1}/{args.episodes} seed={args.seed+i}: cov={r['coverage']:.3f} "
                  f"vic={r['victims']:.3f} succ={int(r['success'])} "
                  f"coll={r['terr_coll']+r['vict_coll']} dmg={r['damage']:.1f}"
                  + (f" ovr={r['override_frac']:.2f}" if kind == "shield" else ""), flush=True)

        agg = {m: ci95([r[m] for r in rows]) for m in METRICS}
        per = [",".join(METRICS)] + [",".join(str(round(r[m], 4)) for m in METRICS) for r in rows]
        (out / f"{kind}_episodes.csv").write_text("\n".join(per))
        line = ",".join([kind, str(len(rows))] +
                        [f"{agg[m][0]:.4f}" for m in METRICS] + [f"{agg[m][1]:.4f}" for m in METRICS])
        with summary.open("a") as f:
            f.write(line + "\n")
        print(f"  -> {kind}: " + "  ".join(f"{m}={agg[m][0]:.3f}" for m in METRICS), flush=True)

    print(f"\n[compare] summary -> {summary}", flush=True)


if __name__ == "__main__":
    main()
