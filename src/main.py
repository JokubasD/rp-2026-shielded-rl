from .simulator import Simulator, MapConfig, visualize_grid_gen
from src.agents.mpc import MpcAgent
from .visualization import Visualizer

WIDTH = 50
HEIGHT = 35

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=5, num_victims=5)
    sim.generate_ground_truth(config, 683192239)

    agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.05, 0.9, 2)
    sim.add_agent(agent1)

    print("Running Simulation steps...")
    history = sim.run(10) 

    print("Launching Visualizer...")
    viz = Visualizer(history, cell_size=25)
    viz.run()

if __name__ == "__main__":
    main()