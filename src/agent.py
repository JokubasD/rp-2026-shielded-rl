from .state import State
from enum import IntEnum

class AgentAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    SCAN = 4
    WAIT = 5

class Agent:
    def __init__(self, name: str, x: int, y: int, width: int, height: int):
        self.name = name
        self.perception = State(width, height)
        self.x = x
        self.y = y
        self.perception.agents.matrix[y][x] = 1
        self.world_width = width
        self.world_height = height

    # Perform a scan to gather information.
    # Should update perceived state matrices.
    # Input: State should always be ground truth
    def scan(self, state: State):
        pass

    # Update internal position (x, y), as well as perceived agent state.
    def move(self, direction: AgentAction) -> None:
        self.perception.agents.matrix[self.y][self.x] = 0

        match direction:
            case AgentAction.MOVE_UP:
                self.y -= 1
            case AgentAction.MOVE_DOWN:
                self.y += 1
            case AgentAction.MOVE_LEFT:
                self.x -= 1
            case AgentAction.MOVE_RIGHT:
                self.x += 1
        
        self.perception.agents.matrix[self.y][self.x] = 1

    # Individual per our research question
    # Can be overwritten in a subclass by just redefining the function
    def get_action(self) -> AgentAction:
        return AgentAction.MOVE_RIGHT
