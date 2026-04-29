from .simulator import Simulator
from .agent import *

WIDTH = 9
HEIGHT = 9

def main():
    bob = Agent("Bob", 4, 4, WIDTH, HEIGHT, 0.05, 0.9, 2, False)

    x, y = 2, 8 # target 

    points = bob.grid_DDA(y, x)

    mock_grid = np.zeros((HEIGHT, WIDTH), dtype=int)

    mock_grid[bob.y][bob.x] = 8
    mock_grid[y][x] = 7
    
    for p in points:
        mock_grid[p] = 1

    print(mock_grid)

    # bob.scan(State(WIDTH, HEIGHT))

    # print("NEW RUN =====================")
    # print(bob.perception.agents.matrix, "\n")
    # print(bob.perception.traversability.matrix, "\n")

if __name__ == "__main__":
    main()