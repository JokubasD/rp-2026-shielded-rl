import itertools
import numpy as np

from src.agent import Agent, AgentAction
from src.state import State
from copy import deepcopy

class Model:
    # Models are needed for predicted theoretical states keep track of the agent's position
    def __init__(self, state: State, x: int, y: int):
        self.state = state
        self.agent_x = x
        self.agent_y = y

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
            model = Model(deepcopy(self.perception), self.x, self.y)

            total_objective = 0.0
            feasible = True

            for step, action in enumerate(sequence):
                if not self._is_feasible(model, action):
                    feasible = False
                    break

                model = self._predict_next_state(model, action)
                total_objective += (self.gamma ** step) * self._objective(model)
            
            if feasible and total_objective > best_objective:
                best_sequence = sequence
                best_objective = total_objective

        return best_sequence[0]

    def _objective(self, model: Model) -> float:
        """
        Calculates the objective value of a given model

        Parameters:
        model: The model to calculate the objective for

        Returns:
        The objective value
        """
        w1, w2, w3, w4 = 1, 1, 1, 1 # To be adjusted

        victim_term =  w1 * self._victim_score(model)
        exploration =  w2 * self._exploration_score(model)
        safety      = -w3 * self._safety_penalty(model)
        confidence  =  w4 * self._confidence_score(model)

        return victim_term + exploration + safety + confidence

    def _predict_next_state(self, model: Model, action: AgentAction) -> Model:
        """
        Predicts what the next state will look like if the agent performs the given action.
        This assumes the action is feasible (see self._is_feasible)

        Parameters:
        model: The current model of the world
        action: The action to perform

        Returns:
        The predicted next model
        """
        new_model = deepcopy(model)

        match action:
            case AgentAction.MOVE_UP:
                new_model.state.agents[self.y - 1][self.x] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.y -= 1
            case AgentAction.MOVE_DOWN:
                new_model.state.agents[self.y + 1][self.x] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.y += 1
            case AgentAction.MOVE_LEFT:
                new_model.state.agents[self.y][self.x - 1] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.x -= 1
            case AgentAction.MOVE_RIGHT:
                new_model.state.agents[self.y][self.x + 1] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.x += 1

        # TODO: Spread fire.
        # ? What approach?
        # ? Expected value (0.7 * FLAMMABLE + 0.3 * BURNING)? idk how we'd store this
        # ? Worst-case (Predict that anything flammable will ignite, spread flammability)? <- TMPC is the one that should be overly cautious
        # ? Probabilistically (Randomly ignite, but agent doesnt know spread_rate), optimizer would no longer be deterministic?
        # ? Best-case/naïve (Predict that fire won't spread)? <- Kinda seems like what Anahita is expecting
        
        # TODO: Mock scanning (updating confidences, tiles explored), currently scanning would scan from the old position; might not be necessary

        return new_model

    def _is_feasible(self, model: Model, action: AgentAction) -> bool:
        """
        Decides whether a given action will result in a feasible position
        (e.g. not walking into a wall)

        Parameters:
        model: The perceived model of the world
        action: The action to check

        Returns:
        Whether the action is feasible
        """
        target_cell_x, target_cell_y = self._target_cell(model.agent_x, model.agent_y, action)
        if (target_cell_x < 0 or target_cell_x >= self.world_width or 
            target_cell_y < 0 or target_cell_y >= self.world_height): # Out of bounds
            return False
        
        return model.state.traversability[target_cell_y][target_cell_x] == 0 # Didn't hit wall


    def _target_cell(self, x: int, y: int, action: AgentAction) -> tuple[int, int]:
        """
        Calculates the cell targeted by the agent given its action.

        Parameters:
        x: Starting x position of the agent
        y: Starting y position of the agent
        action: The action the agent wants to perform

        Returns:
        The target cell indices
        """
        match action:
            case AgentAction.MOVE_UP:
                return x, y - 1
            case AgentAction.MOVE_DOWN:
                return x, y + 1
            case AgentAction.MOVE_LEFT:
                return x - 1, y
            case AgentAction.MOVE_RIGHT:
                return x + 1, y
        
        return x, y # Wait action
    
    def _victim_score(self, model: Model) -> float:
        """
        Calculates a score of a model in regards to the number of victims found.

        Parameters:
        model: The model to calculate the score for

        Returns:
        The score
        """
        # ? A predicted state should never have found more victims
        # ? so this should be equal for all predicted states;
        # ? is there a point in having this?
        return np.sum(model.state.victims.matrix) 
    
    def _exploration_score(self, model: Model) -> float:
        """
        Calculates a score of a model in regards to how much area is explored.

        Parameters:
        model: The model to calculate the score for

        Returns:
        The score
        """
        explored = np.count_nonzero(model.state.confidence.matrix)
        return explored
        # Find agent position in this predicted state
        