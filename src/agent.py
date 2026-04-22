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
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 sigma: float, scan_accuracy: float, scan_radius: int):
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

        self.move_history: list[tuple[int, int]] = [(x, y)] # Maintain a log of what positions the agent has been in for stats

        self.illegal_moves = 0

    def scan(self, state: State) -> None:
        """
        Retrieves information from the state passed in with some noise/uncertainties

        Parameters:
        state: The true world to scan from
        """
        # Scan all indices where manhattan distance to agent <= scan_radius
        scanned_indices = []
        top_scan_limit = max(0, self.y - self.scan_radius)
        bottom_scan_limit = min(self.world_height, self.y + self.scan_radius + 1)
        left_scan_limit = max(0, self.x - self.scan_radius)
        right_scan_limit = min(self.world_width, self.x + self.scan_radius + 1)
        for i in range(top_scan_limit, bottom_scan_limit):
            for j in range(left_scan_limit, right_scan_limit):
                if abs(i - self.y) + abs(j - self.x) <= self.scan_radius:
                    scanned_indices.append((i, j))

        # for i, j in scanned_indices: do stuff

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

    # Individual per our research question
    # Can be overwritten in a subclass by just redefining the function
    def get_action(self) -> AgentAction:
        """
        To be overwritten for each research question.

        Returns:
        The action the agent wants to perform given its internal perception matrix
        """
        return AgentAction.MOVE_RIGHT
