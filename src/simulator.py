from .state import State
from .agent import *

from copy import deepcopy

class Simulator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.ground_truth = State(width, height)



    # Adds an agent to the simulation
    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)
        self.ground_truth.agents.matrix[agent.y][agent.x] = 1



    # Performs a single step of the simulation.
    # This includes agent actions, environment actions, and updating the ground truth.
    # Returns the ground truth state after the step.
    def step(self) -> State:
        # Perform agent actions
        for agent in self.agents:
            action = agent.get_action()
            is_move = action < 4
            
            if is_move:
                self.ground_truth.agents.matrix[agent.y][agent.x] = 0
                agent.move(action)
                self.ground_truth.agents.matrix[agent.y][agent.x] = 1
            
            elif action == AgentAction.SCAN:
                agent.scan(self.ground_truth)

        # Perform environment actions (firespread, etc.)

        # Don't let the returned state modify current state
        result = deepcopy(self.ground_truth)
        return result 

    # Performs a number of steps of the simulation.
    # Returns a list of the ground truth states after each step.
    def run(self, steps: int) -> list[State]:
        record: list[State] = []
        for _ in range(steps):
            record.append(self.step())
        return record