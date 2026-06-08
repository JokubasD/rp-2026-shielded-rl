"""
Utility function to compute the BFS distance from the agent to the nearest
unexplored region.
    
TODO for refactor - Move this to src/util/frontier.py and import it from both places 
"""
from collections import deque

import numpy as np


def nearest_frontier_distance(
    explored: np.ndarray, traversable: np.ndarray, y: int, x: int
) -> int:
    """Hops from (y, x) to the nearest explored cell that borders an unexplored cell.
    Uses belief state traversability to only hop through cells the agent believes are traversable.
    """
    h, w = explored.shape
    visited = np.zeros((h, w), dtype=bool)
    visited[y, x] = True
    q: deque = deque(((y, x, 0),))
    while q:
        cy, cx, d = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not explored[ny, nx]: # checks if any of the neighbors of (cy, cx) are unexplored, if so we have found the frontier and can return the distance.
                return d
            if not visited[ny, nx] and explored[ny, nx] and traversable[ny, nx]: # else if the neighbor is explored and traversable, we can add it to the queue to continue the search.
                visited[ny, nx] = True
                q.append((ny, nx, d + 1))
    return 0
