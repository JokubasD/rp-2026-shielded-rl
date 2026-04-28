from .simulator import Simulator
from .agent import *

WIDTH = 9
HEIGHT = 9

def main():
    bob = Agent("Bob", 4, 4, WIDTH, HEIGHT, 0.05, 0.9, 2, True)
    bob.scan(State(WIDTH, HEIGHT))

    print("NEW RUN =====================")
    print(bob.perception.agents.matrix, "\n")
    print(bob.perception.traversability.matrix, "\n")

if __name__ == "__main__":
    main()