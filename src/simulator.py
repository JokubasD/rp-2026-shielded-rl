from state import State

class Simulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.ground_truth = State(width, height)