from .state import State
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

import tcod
from tcod import libtcodpy

LOS_THRESHOLD = 1

class AgentAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class Agent:
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        self.name = name
        self.perception = State(width, height)
        self.x = x
        self.y = y
        self.perception.agents[y][x] = 1
        self.world_width = width
        self.world_height = height

        self.decay = decay # Certainty decay per time step [0,1]
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
        visible = self._tiles_in_radius() & self._tiles_in_los(state)
        
        for y in range(self.world_height):
            for x in range(self.world_width):
                if visible[y, x] == 1:
                    self.perception.confidence[y][x] = max(self.perception.confidence[y][x] - self.decay, self._tile_accuracy(y, x))
                    self.perception.traversability[y][x] = state.traversability[y][x]
                    self.perception.victims[y][x] = state.victims[y][x]
                    self.perception.agents[y][x] = state.agents[y][x]
                else:
                    self.perception.confidence[y][x] = max(self.perception.confidence[y][x] - self.decay, 0)

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

    def _tiles_in_radius(self) -> NDArray:
        """
        Returns a mask of all tiles within scan_radius of the agent
        """
        y, x = np.ogrid[:self.world_height, :self.world_width]
        return (y - self.y)**2 + (x - self.x)**2 <= self.scan_radius**2 # euclidean distance
    
    def _tiles_in_los(self, state: State) -> NDArray:
        """
        Basically just TCOD's Restrictive Precise Angle Shadowcasting FOV calculation

        Returns a mask of tiles which are within line of sight from the agent. Note that TCOD's definition
        of radius is different to our own, so we need to combine it with a radius mask later. 
        """
        transparency = np.where(state.traversability.matrix >= LOS_THRESHOLD, 0, 1)
        return tcod.map.compute_fov(transparency, (self.y, self.x), self.scan_radius, True, libtcodpy.FOV_RESTRICTIVE)
    
    def _tile_accuracy(self, row: int, col: int) -> float:
        acc = self.scan_accuracy
        if self.scan_falloff:
            var = (self.scan_radius / 3) ** 2
            dx = col - self.x
            dy = row - self.y
            acc *= np.exp(-0.5 * (dx**2 + dy**2) / var)
        
        return acc