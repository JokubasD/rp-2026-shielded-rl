import itertools
import numpy as np

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
        self.gamma = 0.9 # Discount factor, nearer moves are more important than later moves

    def get_action(self) -> AgentAction:
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        best_sequence = [AgentAction.WAIT] * self.horizon
        best_objective = float('-inf')

        for sequence in itertools.product(AgentAction, repeat=self.horizon):
            state = deepcopy(self.perception)

            total_objective = 0.0
            feasible = True

            for step, action in enumerate(sequence):
                if not self._is_feasible(state, action):
                    feasible = False
                    break

                state = self._predict_next_state(state, action)
                total_objective += (self.gamma ** step) * self._objective(state)
            
            if feasible and total_objective > best_objective:
                best_sequence = sequence
                best_objective = total_objective

        return best_sequence[0]

    def _objective(self, state: State) -> float:
        """
        Calculates the objective value of a given state

        Parameters:
        state: The state to calculate the objective for

        Returns:
        The objective value
        """
        w1, w2, w3, w4 = 1, 1, 1, 1 # To be adjusted

        victim_term =  w1 * self._victim_score(state)
        exploration =  w2 * self._exploration_score(state)
        safety      = -w3 * self._safety_penalty(state)
        confidence  =  w4 * self._confidence_score(state)

        return victim_term + exploration + safety + confidence

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
            case AgentAction.MOVE_DOWN:
                new_state.agents[self.y + 1][self.x] = 1
                new_state.agents[self.y][self.x] = 0
            case AgentAction.MOVE_LEFT:
                new_state.agents[self.y][self.x - 1] = 1
                new_state.agents[self.y][self.x] = 0
            case AgentAction.MOVE_RIGHT:
                new_state.agents[self.y][self.x + 1] = 1
                new_state.agents[self.y][self.x] = 0

        # TODO: Spread fire.
        # ? What approach?
        # ? Expected value (0.7 * FLAMMABLE + 0.3 * BURNING)? idk how we'd store this
        # ? Worst-case (Predict that anything flammable will ignite, spread flammability)? <- TMPC is the one that should be overly cautious
        # ? Probabilistically (Randomly ignite, but agent doesnt know spread_rate), optimizer would no longer be deterministic?
        # ? Best-case/naïve (Predict that fire won't spread)? <- Kinda seems like what Anahita is expecting
        
        # TODO: Mock scanning (updating confidences, tiles explored), currently scanning would scan from the old position; might not be necessary

        return new_state

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
    
    def _victim_score(self, state: State) -> float:
        """
        Calculates a score of a theoretical state in regards to the number of victims found.

        Parameters:
        state: The state to calculate the score for

        Returns:
        The score
        """
        # ? A predicted state should never have more victims
        # ? so this should be equal for all predicted states;
        # ? is there a point in having this?
        return np.sum(state.victims.matrix) 
    
    def _exploration_score(self, state: State) -> float:
        """
        Calculates a score of a theoretical state in regards to how much area is explored.

        Parameters:
        state: The state to calculate the score for

        Returns:
        The score
        """
        explored = np.count_nonzero(state.confidence.matrix)
        return explored
        # Find agent position in this predicted state
        