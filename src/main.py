from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.tmpc import TmpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

import matplotlib.pyplot as plt

WIDTH = 35
HEIGHT = 35

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=10, num_victims=15, min_room_length=5, min_room_width=5, max_room_length=10, max_room_width=10, max_tunnel_thickness=2, fire_spread_rate=0.05, fire_duration=10, initial_fire_points=2)
    sim.generate_ground_truth(config)

    # Saved seeds: 182840517, 210577037, 335492940, 95007438

    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, True)
    agent1 = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    sim.add_agent(agent1)

    visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
                       sim.ground_truth.victims, sim.ground_truth.vulnerability,
                       sim.ground_truth.fire)

    print("Running Simulation steps...")
    history = sim.run(500)

    # print(agent1.closest_unexplored())
    # distances_end = closest_unexplored(agent1.world_height, agent1.world_width, agent1.explored, agent1.perception.traversability.matrix, agent1.perception.victims.matrix)

    # plt.matshow(agent1._closest_unexplored())
    # plt.show()

    print("Launching Visualizer...")
    viz = Visualizer(history, 1200, 1200)
    viz.run()

    # viz = Visualizer.from_file("saved_runs/private\sim_20260514_160007_127steps.pkl", 2000, 2000)
    # viz.run()

if __name__ == "__main__":
    main()