from .state import State
from enum import IntEnum

import numpy as np

class AgentAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class Agent:
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 sigma: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        self.name = name
        self.perception = State(width, height)
        self.x = x
        self.y = y
        self.perception.agents[y][x] = 1
        self.world_width = width
        self.world_height = height

        self.sigma = sigma # Certainty decay per time step [0,1]
        self.scan_accuracy = scan_accuracy # Scan accuracy [0,1]
        self.scan_radius = scan_radius # How far the agent can see when it scans
        self.scan_falloff = scan_falloff # Whether to have scan be less accurate further from robot

        self.move_history: list[tuple[int, int]] = [(x, y)] # Maintain a log of what positions the agent has been in for stats

        self.illegal_moves = 0

    def scan(self, state: State) -> None:
        """
        Retrieves information from the state passed in (with some noise/uncertainties?)

        Parameters:
        state: The true world to scan from
        """
        for i in range(self.world_height):
            for j in range(self.world_width):
                if self._tile_scanned(i, j):
                    accuracy = self.scan_accuracy
                    if (self.scan_falloff):
                        delta_x = j - self.x
                        delta_y = i - self.y
                        variance = (self.scan_radius / 3) ** 2
                        accuracy *= np.exp(-0.5 * (delta_x + delta_y) / variance)
                    
                    self.perception.confidence[i][j] = max(self.perception.confidence[i][j] - self.sigma, accuracy)
                        
                    self.perception.traversability[i][j] = state.traversability[i][j]
                    self.perception.victims[i][j] = state.victims[i][j]
                    self.perception.agents[i][j] = state.agents[i][j]
                else:
                    self.perception.confidence[i][j] = max(self.perception.confidence[i][j] - self.sigma, 0)
                    

    def move(self, direction: AgentAction) -> None:
        """
        Updates an agent's internal position based on an action

        Parameters:
        direction: The action to perform
        """

        target_x, target_y = self.x, self.y

        match direction:
            case AgentAction.MOVE_UP:
                target_y -= 1
            case AgentAction.MOVE_DOWN:
                target_y += 1
            case AgentAction.MOVE_LEFT:
                target_x -= 1
            case AgentAction.MOVE_RIGHT:
                target_x += 1
            case _:
                return

        if not (0 <= target_x < self.world_width and 0 <= target_y < self.world_height):
            self.illegal_moves += 1
            return

        self.perception.agents[self.y][self.x] = 0
        self.x, self.y = target_x, target_y
        self.move_history.append((self.x, self.y))
        self.perception.agents[self.y][self.x] = 1

    def get_action(self) -> AgentAction:
        """
        To be overwritten for each research question.

        Returns:
        The action the agent wants to perform given its internal perception matrix
        """
        return AgentAction.MOVE_RIGHT

    def _tile_scanned(self, row: int, col: int) -> bool:
        """
        Checks if a tile is within the scan radius and within line-of-sight of the agent
        """
        return self._tile_in_range(row, col) and self._tile_in_line_of_sight(row, col)

    def _tile_in_range(self, row: int, col: int) -> bool:
        """
        Checks if a tile is within the scan radius of the agent
        """
        return abs(row - self.y) ** 2 + abs(col - self.x) ** 2 <= self.scan_radius ** 2 # euclidean distance

    def _tile_in_line_of_sight(self, row: int, col: int) -> bool:
        """
        Checks if a tile is within line-of-sight of the agent.
        Uses modified bresenham's line algorithm
        """
        return True