from .state import State
from .agent import *

from copy import deepcopy

class Simulator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.ground_truth = State(width, height)


    def add_agent(self, agent: Agent) -> None:
        """
        Adds an agent to the simulation and updates the ground truth

        Parameters:
        agent: The agent to add
        """
        self.agents.append(agent)
        self.ground_truth.agents.matrix[agent.y][agent.x] = 1



    def step(self) -> State:
        """
        Performs a single step of the simulation.
        This includes agent actions, environment actions, and updating the ground truth.

        Returns:
        The ground truth state after the step.
        """
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

    def run(self, steps: int) -> list[State]:
        """
        Performs a number of steps of the simulation.

        Parameters:
        steps: The number of steps to perform

        Returns:
        A list of the ground truth states after each step.
        """
        record: list[State] = []
        for _ in range(steps):
            record.append(self.step())
        return record