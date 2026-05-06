import itertools
import numpy as np
import random
from copy import deepcopy

from src.agent import Agent, AgentAction
from src.state import State
from src.constants import FireLevel, TraversabilityLevel

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

                model = self._predict_next_model(model, action)
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
        # TODO: Normalize all scores (maybe in methods?)
        w_exploration, w_safety, w_confidence = 10, 1, 2 # To be adjusted

        exploration =  w_exploration * self._exploration_score(model)
        safety      = -w_safety * self._safety_penalty(model)
        confidence  =  w_confidence * self._confidence_score(model)

        return exploration + safety + confidence

    def _predict_next_model(self, model: Model, action: AgentAction) -> Model:
        """
        Creates a predicted model of the world if the agent performs the given action.
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
                new_model.agent_y -= 1
            case AgentAction.MOVE_DOWN:
                new_model.state.agents[self.y + 1][self.x] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.agent_y += 1
            case AgentAction.MOVE_LEFT:
                new_model.state.agents[self.y][self.x - 1] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.agent_x -= 1
            case AgentAction.MOVE_RIGHT:
                new_model.state.agents[self.y][self.x + 1] = 1
                new_model.state.agents[self.y][self.x] = 0
                new_model.agent_x += 1

        self._predict_fire_spread(new_model)        
        self._mock_scan(new_model)

        return new_model

    def _predict_fire_spread(self, model: Model) -> None:
        """
        Predicts the spread of fire, updates model in place.
        Does so probabilistically (where random.random() should be seeded), and with an educated guess on spread rate

        Parameters:
        model: The model to predict the spread for
        """
        # Probabilistically (Randomly (seeded) ignite, educated guess on spread rate)
        predicted_spread_rate = 0.3

        for y in range(self.world_height):
            for x in range(self.world_width):
                if model.state.fire[y][x] == FireLevel.FLAMMABLE and random.random() < predicted_spread_rate:
                    model.state.fire[y][x] = FireLevel.BURNING
                    
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        
                        if not (0 <= nx < self.width and 0 <= ny < self.height):
                            continue
                        if model.state.traversability[ny, nx] == TraversabilityLevel.UNTRAVERSIBLE:
                            continue

                        if model.state.fire[ny, nx] == FireLevel.SAFE:
                            model.state.fire[ny, nx] = FireLevel.FLAMMABLE

    def _mock_scan(self, model: Model) -> None:
        """
        Mocks a scan by updating the model's confidence values in place.

        Parameters:
        model: The model to update
        """
        # TODO: Add LoS check to if tile is scanned
        # TODO: Update Jacob's metric?
        for i in range(self.world_height):
            for j in range(self.world_width):
                if abs(i - model.agent_y) ** 2 + abs(j - model.agent_x) ** 2 <= self.scan_radius ** 2:
                    model.state.confidence[i][j] = max(model.state.confidence[i][j] - self.sigma, self.scan_accuracy)
                else:
                    model.state.confidence[i][j] = max(model.state.confidence[i][j] - self.sigma, 0)

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
    
    def _exploration_score(self, model: Model) -> float:
        """
        Calculates a score of a model in regards to how much area is explored.

        Parameters:
        model: The model to calculate the score for

        Returns:
        The score
        """
        # Should be updated; confidence == 0 does not mean unexplored; use Jacobs metric later
        explored = np.count_nonzero(model.state.confidence.matrix)
        unexplored = np.argwhere(model.state.confidence.matrix == 0)
    
        if len(unexplored) == 0:
            return explored
        
        # Heading towards unexplored tiles is more important
        distances = np.linalg.norm(unexplored - np.array([model.agent_y, model.agent_x]), axis=1)
        min_distance = np.min(distances)
        proximity_bonus = 1.0 / (1.0 + min_distance)
        
        return explored + proximity_bonus

    def _safety_penalty(self, model: Model) -> float:
        """
        Calculates a score of a model in regards to how unsafe the agent is

        Parameters:
        model: The model to calculate the score for

        Returns:
        The score
        """
        # Penalty for being on vulnerable tile
        vulnerability_penalty = model.state.vulnerability[model.agent_y][model.agent_x]

        # Penalty for being near fire
        fire_penalty = 0.0
        for row in range(self.world_height):
            dy = row - model.agent_y
            for col in range(self.world_width):
                fire_level: FireLevel = model.state.fire[row][col]
                if fire_level == FireLevel.SAFE or fire_level == FireLevel.BURNT:
                    continue
                dx = col - model.agent_x
                distance = dy ** 2 + dx ** 2
                danger = (int(fire_level) ** 2) # Burning is much more dangerous than flammable
                fire_penalty += danger / distance

        return vulnerability_penalty + fire_penalty
        
    def _confidence_score(self, model: Model) -> float:
        """
        Calculates a score of a model in regards to how confident the agent is

        Parameters:
        model: The model to calculate the score for

        Returns:
        The score
        """
        return np.sum(model.state.confidence.matrix)