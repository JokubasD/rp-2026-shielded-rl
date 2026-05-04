from .grid import Grid

class State:
    def __init__(self, width: int, height: int):
        self.traversability = Grid(width=width, height=height)
        self.vulnerability = Grid(width=width, height=height)
        self.fire = Grid(width=width, height=height)
        self.victims = Grid(width=width, height=height)
        self.agents = Grid(width=width, height=height)
        self.confidence = Grid(width=width, height=height)