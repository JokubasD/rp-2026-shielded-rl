from .simulator import Simulator
from .agent import *

WIDTH = 9
HEIGHT = 9

def main():
    bob = Agent("Bob", 4, 4, WIDTH, HEIGHT, 0.05, 0.9, 4, True)

    ground_truth = State(WIDTH, HEIGHT)
    ground_truth.traversability[3, 4] = 1
    ground_truth.traversability[5, 3:6] = 1
    bob.scan(ground_truth)

    print("NEW RUN =====================")
    print(bob.perception.agents.matrix, "\n")
    print(bob.perception.traversability.matrix, "\n")
    print(bob.perception.confidence.matrix.round(3), "\n")

if __name__ == "__main__":
    main()