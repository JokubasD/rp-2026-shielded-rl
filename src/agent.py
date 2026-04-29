from .state import State
from enum import IntEnum

import numpy as np
from math import floor, ceil

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
                if self._tile_scanned(i, j, state):
                    self.perception.confidence[i][j] = max(self.perception.confidence[i][j] - self.sigma, self._tile_accuracy(i, j))
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

    def _tile_scanned(self, row: int, col: int, state: State) -> bool:
        """
        Checks if a tile is within the scan radius and within line-of-sight of the agent
        """
        return self._tile_in_range(row, col) and self._tile_in_line_of_sight(row, col, state)

    def _tile_in_range(self, row: int, col: int) -> bool:
        """
        Checks if a tile is within the scan radius of the agent
        """
        return abs(row - self.y) ** 2 + abs(col - self.x) ** 2 <= self.scan_radius ** 2 # euclidean distance
    
    def _tile_in_line_of_sight(self, row: int, col: int, state: State) -> bool:
        """
        Checks if a tile is within line-of-sight of the agent.
        Uses modified bresenham's line algorithm
        """
        LOS_THRESHOLD = 1 # at what level of (un-)traversability is LoS blocked?
        
        points = self.grid_DDA(row, col)
        return not np.any([state.traversability[p] >= LOS_THRESHOLD for p in points])
    
    def grid_DDA(self, row: int, col: int) -> list[tuple[int, int]]:
        x1, y1 = self.x, self.y
        x2, y2 = col, row
        dx, dy = x2 - x1, y2 - y1

        dx, step_x = (dx, 1) if dx > 0 else (-dx, -1)
        dy, step_y = (dy, 1) if dy > 0 else (-dy, -1)

        if dy == 0: # horizontal line - unnecessary but much more efficient
            return [(y1, x) for x in range(x1 + step_x, x2, step_x)]
        if dx == 0: # vertical line - needed to avoid divide-by-zero
            return [(y, x1) for y in range(y1 + step_y, y2, step_y)]

        slope = dy / dx
        step_s = step_y * slope

        x = x1
        last = y1
        next = y1 + 0.5 * step_s
        points = []

        if step_y > 0:
            for _ in range(dx + 1):
                lb = ceil(last - 0.5) # round half down
                ub = floor(next + 0.5) # round half up

                for y in range(lb, ub + 1):
                    points.append((y, x))

                x += step_x
                last = next

                next += step_s
                if next > y2:
                    next = y2

        else:
            for _ in range(dx + 1):
                lb = floor(last + 0.5) # round half up
                ub = ceil(next - 0.5) # round half down

                for y in range(ub, lb + 1):
                    points.append((y, x))

                x += step_x
                last = next

                next += step_s
                if next < y2:
                    next = y2
        
        return [p for p in points if p not in ((y1, x1), (y2, x2))] # filter out start and end points
    
    def _tile_accuracy(self, row: int, col: int) -> float:
        acc = self.scan_accuracy
        if self.scan_falloff:
            var = (self.scan_radius / 3) ** 2
            dx = col - self.x
            dy = row - self.y
            acc *= np.exp(-0.5 * (dx**2 + dy**2) / var)
        
        return acc