from simulator import Simulator, MapConfig, visualize_grid_gen
from agent import Agent
from visualization import Visualizer

WIDTH = 40
HEIGHT = 25

def main():
    # 1. Setup the world
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=5, num_victims=5)
    sim.generate_ground_truth(config)

   #  visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims)

    # 2. Setup Bob (The Agent)
    # x=12, y=12 is roughly center. sigma=0.01, accuracy=0.9, radius=4
    bob = Agent("Bob", 12, 12, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob)

    # 3. Run the simulation
    # This will now trigger bob.scan() and bob.move() inside simulator.step()
    print("Running Simulation steps...")
    history = sim.run(30) 

    # 4. Visualize the result
    print("Launching Visualizer...")
    viz = Visualizer(history, cell_size=25)
    viz.run()

if __name__ == "__main__":
    main()