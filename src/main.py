from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent, closest_unexplored
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

WIDTH = 50
HEIGHT = 50

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=4, num_victims=5, min_room_length=4, min_room_width=4, max_room_length=7, max_room_width=7, max_tunnel_thickness=1)
    sim.generate_ground_truth(config)

    # Saved seeds: 182840517 (181 steps), 210577037; 

    agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, True)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = Agent("mpc", 0, 1, WIDTH, HEIGHT, 0.05, 0.9, 4, False)
    sim.add_agent(agent1)

    visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
                       sim.ground_truth.victims, sim.ground_truth.vulnerability,
                       sim.ground_truth.fire)

    print("Running Simulation steps...")
    history = sim.run(500)

    # print(agent1.closest_unexplored())
    # print(closest_unexplored(agent1.world_height, agent1.world_width, agent1.explored, agent1.perception.traversability.matrix))

    print("Launching Visualizer...")
    viz = Visualizer(history, 2000, 2000)
    viz.run()

    # viz = Visualizer.from_file("saved_runs/private\sim_20260514_160007_127steps.pkl", 2000, 2000)
    # viz.run()

if __name__ == "__main__":
    main()