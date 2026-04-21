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
        self.perception.agents[y][x] = 1
        self.world_width = width
        self.world_height = height

    # Perform a scan to gather information.
    # Should update perceived state matrices.
    # Input: State should always be ground truth
    def scan(self, state: State):
        pass

    def move(self, direction: AgentAction) -> None:
        """
        Updates an agent's internal position based on an action

        Parameters:
        direction: The action to perform
        """


        self.perception.agents[self.y][self.x] = 0

        match direction:
            case AgentAction.MOVE_UP:
                self.y -= 1
            case AgentAction.MOVE_DOWN:
                self.y += 1
            case AgentAction.MOVE_LEFT:
                self.x -= 1
            case AgentAction.MOVE_RIGHT:
                self.x += 1
        
        self.perception.agents[self.y][self.x] = 1

    # Individual per our research question
    # Can be overwritten in a subclass by just redefining the function
    def get_action(self) -> AgentAction:
        """
        To be overwritten for each research question.

        Returns:
        The action the agent wants to perform given its internal perception matrix
        """
        return AgentAction.MOVE_RIGHT
