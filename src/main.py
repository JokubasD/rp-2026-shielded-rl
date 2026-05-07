from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

WIDTH = 50
HEIGHT = 35

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=5, num_victims=5)
    sim.generate_ground_truth(config, 182840517)

    agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = Agent("mpc", 0, 1, WIDTH, HEIGHT, 0.05, 0.9, 4, False)
    sim.add_agent(agent1)

    # visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
    #                    sim.ground_truth.victims, sim.ground_truth.vulnerability,
    #                    sim.ground_truth.fire)

    print("Running Simulation steps...")
    history = sim.run(300) 

    print("Launching Visualizer...")
    viz = Visualizer(history, cell_size=25)
    viz.run()

if __name__ == "__main__":
    main()