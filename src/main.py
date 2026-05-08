from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

WIDTH = 30
HEIGHT = 30

def main():
    viz = Visualizer.from_file("saved_runs/shared\sim_20260508_153308_134steps.pkl")
    print("Launching Visualizer...")
    viz.run()

if __name__ == "__main__":
    main()