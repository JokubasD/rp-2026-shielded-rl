from simulator import Simulator, MapConfig, visualize_grid_gen
from agent import Agent
from visualization import Visualizer

WIDTH = 40
HEIGHT = 25

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=5, num_victims=5)
    sim.generate_ground_truth(config)

   #  visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims)

    bob = Agent("Bob", 12, 12, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob)

    print("Running Simulation steps...")
    history = sim.run(30) 

    print("Launching Visualizer...")
    viz = Visualizer(history, cell_size=25)
    viz.run()

if __name__ == "__main__":
    main()