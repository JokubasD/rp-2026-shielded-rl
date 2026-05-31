"""Run a trained pure-RL policy inside the Simulator and view it.

Two presets:
  default (no flag) - small diagnostic env (12x12, 1 victim, no hazards):
      python -m src.rl.visualize          # default seed
      python -m src.rl.visualize 30018    # custom seed

  --full              - full SaREnv config (40x40, 5 victims, hazards on):
      python -m src.rl.visualize --full
      python -m src.rl.visualize --full 20000
"""
import sys

import numpy as np
from stable_baselines3 import PPO

from src.constants import AgentAction, MapConfig
from src.rl.agent import RLAgent
from src.rl.env import N_STACK, build_observation
from src.simulator import Simulator
from src.visualization import Visualizer


class TrainedRLAgent(RLAgent):
    """RLAgent driven by a frozen, trained PPO policy.

    Maintains its own frame-stack buffer so inference matches the
    VecFrameStack(n_stack=N_STACK) layer used during training.
    """

    def __init__(self, model, *args, n_stack: int = N_STACK, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.n_stack = n_stack
        self._stack: np.ndarray | None = None  # filled on first get_action

    def get_action(self) -> AgentAction:
        obs = build_observation(self)  # (C, H, W)
        if self._stack is None:
            # First call: replicate the initial observation n_stack times,
            # mirroring SB3 VecFrameStack's reset behaviour.
            self._stack = np.tile(obs, (self.n_stack, 1, 1))  # (C*n_stack, H, W)
        else:
            c = obs.shape[0]
            self._stack = np.concatenate([self._stack[c:], obs], axis=0)
        action, _ = self.model.predict(self._stack[None, ...], deterministic=True)
        return AgentAction(int(action[0]))


def _run(model_path, config, width, height, max_steps, seed, viz_size):
    print(f"loading model from {model_path}.zip ...")
    model = PPO.load(model_path)

    print(f"generating map (seed={seed}) ...")
    sim = Simulator(width, height)
    sim.generate_ground_truth(config, seed=seed)

    agent = TrainedRLAgent(
        model, name="rl", x=0, y=0, width=width, height=height,
        decay=0.01, scan_accuracy=0.9, scan_radius=3, scan_falloff=True,
    )
    sim.add_agent(agent)

    print(f"running up to {max_steps} steps ...")
    history = sim.run(max_steps)
    m = sim.metrics
    print(
        f"outcome={m.outcome.value}, steps={m.steps_taken}, "
        f"found={m.victims_found}/{m.total_victims}, "
        f"damage={m.damage.get(agent, 0)}"
    )

    print("launching Visualizer ...")
    viz = Visualizer(history, viz_size, viz_size)
    viz.run()


def main():
    args = sys.argv[1:]
    full = "--full" in args
    args = [a for a in args if a != "--full"]

    if full:
        # Matches the new "Balanced" SaREnv default used in train.py (runs/ppo_sanity).
        config = MapConfig(
            num_rooms=8, num_victims=3, num_agents=0,
            min_room_width=5, max_room_width=10,
            min_room_length=5, max_room_length=10,
            max_tunnel_thickness=2,
            initial_fire_points=1, fire_spread_rate=0.1, fire_duration=8,
            room_vulnerability_probability=0.25, room_vulnerability_severity=0.3,
            tunnel_vulnerability_probability=0.2, tunnel_vulnerability_severity=0.3,
        )
        seed = int(args[0]) if args else 20_002
        _run("runs/ppo_sanity", config, 40, 40, 300, seed, 800)
    else:
        # Matches diagnose.py training (runs/ppo_diag_small).
        config = MapConfig(
            num_rooms=2, num_victims=1, num_agents=0,
            initial_fire_points=0, fire_spread_rate=0.0,
            room_vulnerability_probability=0.0, tunnel_vulnerability_probability=0.0,
            min_room_width=3, max_room_width=5,
            min_room_length=3, max_room_length=5,
            max_tunnel_thickness=1,
        )
        seed = int(args[0]) if args else 30_009
        _run("runs/ppo_diag_small", config, 12, 12, 100, seed, 600)


if __name__ == "__main__":
    main()
