from state import State

class Agent:
    def __init__(self, name, width, height):
        self.name = name
        self.perception = State(width, height)