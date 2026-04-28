import numpy as np
from state import State
from visualization import Visualizer

def main_demo():
    history = []
    
    # Create 10 frames of dummy data
    for i in range(10):
        mock_state = State(width=20, height=15)
        
        # Hardcode a wall at (5,5)
        mock_state.traversability.matrix[5][5] = 1 
        
        # Make a "Robot" move across the screen
        # Since state.agents is a Grid object, we use .matrix
        mock_state.agents.matrix[5][i] = 1 
        
        # Make a "Victim" appear at step 5
        if i >= 5:
            mock_state.victims.matrix[10][10] = 1
            
        # Add a "Confidence" heatmap that grows
        mock_state.confidence.matrix[0:i, 0:i] = 0.5
        
        history.append(mock_state)

    print("Launching Visualizer with Mock Data...")
    viz = Visualizer(history, cell_size=30)
    viz.run()

if __name__ == "__main__":
     main_demo()

# from simulator import Simulator, MapConfig
# from agent import *
# from visualization import Visualizer

# WIDTH = 4
# HEIGHT = 3

# def main():
#     sim = Simulator(WIDTH, HEIGHT)
#     bob = Agent("Bob", 0, 0, WIDTH, HEIGHT)
#     sim.add_agent(bob)

#     print("NEW RUN =====================")
#     print(sim.ground_truth.agents.matrix, "\n")
#     steps = sim.run(3)
#     for step in steps:
#         print(step.agents.matrix, "\n")

# if __name__ == "__main__":
#     main()