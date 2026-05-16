from .mpc import MpcAgent, closest_unexplored
from ..constants import AgentAction

class TmpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)

    def get_action(self):
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        self.frontier_distances = closest_unexplored(self.world_height, self.world_width, self.explored, self.perception.traversability.matrix, self.perception.victims.matrix)
        # self.frontier_distances -= self.frontier_distances[self.y][self.x]
        self.fire_prediction = self._predict_fire_spread_horizon()
        
        def dfs(model_state: MpcAgent, depth: int, objective: float) -> tuple[float, list]:
            if depth >= self.horizon:
                return objective + model_state._objective_terminal(), []

            best_objective = float('-inf')
            best_sequence = []

            for action in AgentAction:
                if not model_state._is_feasible(action):
                    continue

                next_state = model_state._predict_next_state(action, depth)
                next_objective = objective + ((self.discount ** depth) * next_state._objective_stage())
                
                future_obj, future_seq = dfs(next_state, depth + 1, next_objective)

                if future_obj > best_objective:
                    best_objective = future_obj
                    best_sequence = [action] + future_seq

            return best_objective, best_sequence

        best_objective, best_sequence = dfs(self.copy(), 0, 0)

        if len(best_sequence) == 0:
            self.infeasible_states += 1
            print("INFEASIBLE STATE REACHED ==========================")
            print("Dscv:", self._discovery_score())
            print("Safety:", self._safety_penalty())
            print("Conf:", self._confidence_score())
            print("Expl:", self._exploration_penalty())
            return AgentAction.WAIT

        return best_sequence[0]