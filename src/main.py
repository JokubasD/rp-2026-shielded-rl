from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent, closest_unexplored
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

WIDTH = 60
HEIGHT = 55

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=15, num_victims=15, min_room_length=5, min_room_width=5, max_room_length=10, max_room_width=10, max_tunnel_thickness=1, fire_spread_rate=0.05, fire_duration=10, initial_fire_points=2)
    sim.generate_ground_truth(config, 95007438)

    # Saved seeds: 182840517 (181 steps), 210577037; 335492940

    agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, True)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = Agent("mpc", 0, 1, WIDTH, HEIGHT, 0.05, 0.9, 4, False)
    sim.add_agent(agent1)

    visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
                       sim.ground_truth.victims, sim.ground_truth.vulnerability,
                       sim.ground_truth.fire)

    print("Running Simulation steps...")
    history = sim.run(700)

    # print(agent1.closest_unexplored())
    # distances_end = closest_unexplored(agent1.world_height, agent1.world_width, agent1.explored, agent1.perception.traversability.matrix, agent1.perception.victims.matrix)

    print("Launching Visualizer...")
    viz = Visualizer(history, 1200, 1200)
    viz.run()

    # viz = Visualizer.from_file("saved_runs/private\sim_20260514_160007_127steps.pkl", 2000, 2000)
    # viz.run()

if __name__ == "__main__":
    main()