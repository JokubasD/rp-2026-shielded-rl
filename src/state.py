from .grid import Grid

class State:
    def __init__(self, width, height):
        self.traversability = Grid(width=width, height=height)
        self.victims = Grid(width=width, height=height)
        self.agents = Grid(width=width, height=height)
        self.confidence = Grid(width=width, height=height)