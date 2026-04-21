from .simulator import Simulator
from .agent import *

WIDTH = 4
HEIGHT = 3

def main():
    sim = Simulator(WIDTH, HEIGHT)
    bob = Agent("Bob", 0, 0, WIDTH, HEIGHT)
    sim.add_agent(bob)

    print("NEW RUN =====================")
    print(sim.ground_truth.agents.matrix, "\n")
    steps = sim.run(3)
    for step in steps:
        print(step.agents.matrix, "\n")

if __name__ == "__main__":
    main()