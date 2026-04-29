import random
import numpy as np

from .state import State
from .constants import *

class FireManager:
    def __init__(self, width: int, height: int, s_r: float):
        self.width = width
        self.height = height
        self.spread_rate = s_r
        self.active_fire_front: set[tuple[int, int]] = set()
    
    def initialize_fire(self,
            ground_truth: State,
            n: int,
            rooms: list[dict]
    ) -> None:
        """
        Initializes the fire points in random rooms.

        Parameters:
        n: The amount of fire points to start with
        rooms: The rooms generated in the traversability matrix.
        """
        ground_truth.fire.matrix = np.full((self.height, self.width), FireLevel.SAFE)

        room_n = len(rooms)

        for _ in range(n):
            random_room = rooms[random.randint(0, room_n - 1)]
            x_range = random_room['x_range']
            y_range = random_room['y_range']
            random_x = random.randint(x_range[0], x_range[1] - 1)
            random_y = random.randint(y_range[0], y_range[1] - 1)
            while (ground_truth.fire[random_y, random_x] == FireLevel.BURNING):
                random_x = random.randint(x_range[0], x_range[1] - 1)
                random_y = random.randint(y_range[0], y_range[1] - 1)
            ground_truth.fire[random_y, random_x] = FireLevel.BURNING
            self.active_fire_front.add((random_x, random_y))
            self._expose_neighbors(ground_truth, random_x, random_y)
        
        return
        

    def spread_fire(self,
            ground_truth: State,
    ) -> None:
        """
        Spread the fire in the ground truth based on the spread rate, in place.

        Parameters:
        ground_truth: The ground truth.
        """
        newly_ignited = set()
        burnt_out = set()

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for x,y in self.active_fire_front:
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                    continue

                is_traversable = ground_truth.traversability[new_y][new_x] == TraversabilityLevel.TRAVERSIBLE
                if not (is_traversable):
                    continue

                if (ground_truth.fire[new_y][new_x] == FireLevel.FLAMMABLE):
                    if (random.random() <= self.spread_rate):
                        ground_truth.fire[new_y][new_x] = FireLevel.BURNING
                        newly_ignited.add((new_x, new_y))
                
        for new_x, new_y in newly_ignited:
            self._expose_neighbors(ground_truth, new_x, new_y)
        
        self.active_fire_front.update(newly_ignited)
    

    def _expose_neighbors(self, ground_truth: State, x: int, y: int) -> None:
        """
        Immediately turns SAFE neighboring cells into FLAMMABLE cells.
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
                
            if ground_truth.traversability[ny, nx] == TraversabilityLevel.UNTRAVERSIBLE:
                continue
                
            if ground_truth.fire[ny, nx] == FireLevel.SAFE:
                ground_truth.fire[ny, nx] = FireLevel.FLAMMABLE


