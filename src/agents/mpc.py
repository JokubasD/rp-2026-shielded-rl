from src.agent import Agent, AgentAction
from src.state import State
from copy import deepcopy

class MpcAgent(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius)
        
        # How many steps ahead to simulate.
        # The number of deep-copies of the perception state made when calculating the action to take
        # will be 5^horizon
        self.horizon = 3 

    def get_action(self) -> AgentAction:
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        return AgentAction.MOVE_RIGHT # Stand-in


    def _is_feasible(self, state: State, action: AgentAction) -> bool:
        """
        Decides whether a given action will result in a feasible state
        (e.g. not walking into a wall)

        Parameters:
        state: The perceived state of the world
        action: The action to check

        Returns:
        Whether the action is feasible
        """
        target_cell_x, target_cell_y = self._target_cell(action)
        if (target_cell_x < 0 or target_cell_x >= self.world_width or 
            target_cell_y < 0 or target_cell_y >= self.world_height): # Out of bounds
            return False
        
        return state.traversability[target_cell_y][target_cell_x] == 0 # Didn't hit wall

    def _predict_next_state(self, state: State, action: AgentAction) -> State:
        """
        Predicts what the next state will look like if the agent performs the given action.
        This assumes the action is feasible (see self._is_feasible)

        Parameters:
        state: The current state of the world
        action: The action to perform

        Returns:
        The predicted next state
        """
        new_state = deepcopy(state)

        match action:
            case AgentAction.MOVE_UP:
                new_state.agents[self.y - 1][self.x] = 1
                new_state.agents[self.y][self.x] = 0
                self.y -= 1
            case AgentAction.MOVE_DOWN:
                new_state.agents[self.y + 1][self.x] = 1
                new_state.agents[self.y][self.x] = 0
                self.y += 1
            case AgentAction.MOVE_LEFT:
                new_state.agents[self.y][self.x - 1] = 1
                new_state.agents[self.y][self.x] = 0
                self.x -= 1
            case AgentAction.MOVE_RIGHT:
                new_state.agents[self.y][self.x + 1] = 1
                new_state.agents[self.y][self.x] = 0
                self.x += 1

        # TODO: Spread fire.
        # ? How?
        # ? Expected value (0.7 * FLAMMABLE + 0.3 * BURNING)? idk how we'd store this
        # ? Worst-case (Predict that anything flammable will ignite, spread flammability)?
        # ? Probabilistically (Randomly ignite (agent doesnt know spread_rate))?
        # ? Best-case (Predict that fire won't spread)?
        
        return new_state

    def _target_cell(self, action: AgentAction) -> tuple[int, int]:
        """
        Calculates the cell targeted by the agent given its action.

        Parameters:
        action: The action the agent wants to perform

        Returns:
        The target cell indices
        """
        tx, ty = self.x, self.y
        match action:
            case AgentAction.MOVE_UP:
                ty -= 1
            case AgentAction.MOVE_DOWN:
                ty += 1
            case AgentAction.MOVE_LEFT:
                tx -= 1
            case AgentAction.MOVE_RIGHT:
                tx += 1
        return tx, ty