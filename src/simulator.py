from copy import deepcopy
from dataclasses import dataclass
import random
import numpy as np
import matplotlib.pyplot as plt
from .grid import Grid
from .state import State
from .agent import *
from .metric import Metric, RunOutcome

TRAVERSIBLE = 0
UNTRAVERSIBLE = 1

AGENT_NOT_PRESENT = 0
AGENT_PRESENT = 1

VICTIM_NOT_PRESENT = 0
VICTIM_PRESENT = 1

@dataclass
class MapConfig:
    num_rooms: int = 4
    unconnected_probability: float = 0.0
    min_room_width: int = 6
    max_room_width: int = 12
    min_room_length: int = 6
    max_room_length: int = 12
    min_tunnel_thickness: int = 1
    max_tunnel_thickness: int = 3
    num_agents: int = 0
    num_victims: int = 2

class Simulator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.ground_truth = State(width, height)
        self.metrics = Metric()

    def add_agent(self, agent: Agent) -> None:
        """
        Adds an agent to the simulation and updates the ground truth

        Parameters:
        agent: The agent to add
        """
        self.agents.append(agent)
        self.ground_truth.agents[agent.y][agent.x] = 1
        self.metrics.register_agent(agent)

    def step(self) -> State:
        """
        Performs a single step of the simulation.
        This includes agent actions, environment actions, and updating the ground truth.

        Returns:
        The ground truth state after the step.
        """
        intents = self._collect_intents()
        self._resolve_agent_conflicts(intents)
        self._commit_moves(intents)

        for agent in self.agents:
            agent.scan(self.ground_truth)

        # Perform environment actions (firespread, etc.)
        self.metrics.steps_taken += 1
        self._update_found_metrics()

        return deepcopy(self.ground_truth)

    def run(self, steps: int) -> list[State]:
        """
        Performs a number of steps of the simulation.

        Parameters:
        steps: The maximum number of steps to perform

        Returns:
        A list of the ground truth states after each step.
        list[0] is the initial state before the sim is run
        """
        record: list[State] = []
        record.append(deepcopy(self.ground_truth))
        for _ in range(steps):
            record.append(self.step())
            if self.metrics.outcome != RunOutcome.IN_PROGRESS:
                break
        if self.metrics.outcome == RunOutcome.IN_PROGRESS:
            self.metrics.outcome = RunOutcome.TIMEOUT
        return record

    def _collect_intents(self) -> dict[Agent, tuple[int, int]]:
        """
        Each agent picks an action. WAIT and collisions are recorded immediately and the
        agent's intent becomes its current position. Otherwise the intent is
        the target cell.
        """
        intents: dict[Agent, tuple[int, int]] = {}
        for agent in self.agents:
            action = agent.get_action()
            if action == AgentAction.WAIT:
                self.metrics.record_wait(agent)
                intents[agent] = (agent.x, agent.y)
                continue

            tx, ty = self._target_cell(agent, action)
            if not (0 <= tx < self.width and 0 <= ty < self.height):
                self.metrics.record_terrain_collision(agent)
                intents[agent] = (agent.x, agent.y)
            elif self.ground_truth.traversability[ty][tx] == UNTRAVERSIBLE:
                self.metrics.record_terrain_collision(agent)
                intents[agent] = (agent.x, agent.y)
            elif self.ground_truth.victims[ty][tx] == VICTIM_PRESENT:
                self.metrics.record_victim_collision(agent)
                intents[agent] = (agent.x, agent.y)
            else:
                intents[agent] = (tx, ty)
        return intents

    def _target_cell(self, agent: Agent, action: AgentAction) -> tuple[int, int]:
        tx, ty = agent.x, agent.y
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

    
    
    def generate_ground_truth(self, config: MapConfig | None = None) -> None:
        if config is None:
            config = MapConfig()
        # generates and sets the 2D grid with agent and victims, currently with preset values.
        self.ground_truth.traversability.matrix, rooms = _generate_traversability_matrix(self.width, self.height, 
                                                                                        config.num_rooms, config.unconnected_probability, 
                                                                                        config.min_room_width, config.max_room_width, 
                                                                                        config.min_room_length, config.max_room_length, 
                                                                                        config.min_tunnel_thickness, config.max_tunnel_thickness)
        
        self.ground_truth.agents.matrix = _place_agents(self.width, self.height, config.num_agents, rooms, self.ground_truth.victims)
        self.ground_truth.victims.matrix = _place_victims(self.width, self.height, config.num_victims, rooms, self.ground_truth.agents)
        self.ground_truth.confidence.matrix = np.ones((self.height, self.width))
        return

def _generate_traversability_matrix(
        x: int, y: int,
        n: int, u_p: float,
        w_min: int, w_max: int,
        l_min: int, l_max: int,
        t_min: int, t_max: int
        ) -> tuple[np.ndarray, list[dict]]:
    """
    Generates 2D traversability matrix with rooms and connecting corridors, and returns the room bounds and the matrix.
    
    Parameters:
    x, y: Dimensions of the grid
    n: Number of rooms
    u_p: (unconnected_probability) The probability a room is unconnected
    w_min, w_max: Min/Max width of the rooms
    l_min, l_max: Min/Max length of the rooms
    t_min, t_max: Min/Max thickness of the tunnels
    """

    if not (0 <= u_p <= 1):
        raise ValueError(f"probability of unconnected rooms should be between 0 and 1 but was: {u_p}")
    
    matrix = np.full((y, x), UNTRAVERSIBLE)
    room_seeds = np.zeros((2, n), dtype=int)
    rooms = []

    tunnel_width = random.randint(t_min, t_max)
    half_tunnel_width = tunnel_width // 2

    for p in range(n):
        c = random.randint(0,x - 1)
        f = random.randint(0,y - 1)

        room_seeds[0][p] = c
        room_seeds[1][p] = f

        random_width = random.randint(w_min, w_max)
        half_random_width = random_width // 2

        random_length = random.randint(l_min, l_max)
        half_random_length = random_length // 2

        x_start = max(0, c - half_random_width)
        x_end = min(x, c + half_random_width + 1)
        y_start = max(0, f - half_random_length)
        y_end = min(y, f + half_random_length + 1)

        matrix[y_start:y_end, x_start:x_end] = TRAVERSIBLE
        rooms.append({
            'x_range': (x_start, x_end),
            'y_range': (y_start, y_end),
            'center': (c, f)
        })
    
    for q in range(n - 1):
        connect = random.random() >= u_p

        if (connect):
            a_x, a_y = room_seeds[0][q], room_seeds[1][q]
            b_x, b_y = room_seeds[0][q + 1], room_seeds[1][q + 1]

            def get_slice(start, end):
                return slice(min(start, end), max(start, end) + 1)
            
            direction = random.randint(0, 1)
            if (direction == 1):
                x_slice = get_slice(a_x, b_x)
                y_start = max(0, a_y - half_tunnel_width)
                y_end = min(y, a_y + half_tunnel_width + 1)
                matrix[y_start:y_end, x_slice] = TRAVERSIBLE
                
                x_start = max(0, b_x - half_tunnel_width)
                x_end = min(x, b_x + half_tunnel_width + 1)
                y_slice = get_slice(a_y, b_y)
                matrix[y_slice, x_start:x_end] = TRAVERSIBLE
            else:
                x_start = max(0, a_x - half_tunnel_width)
                x_end = min(x, a_x + half_tunnel_width + 1)
                y_slice = get_slice(a_y, b_y)
                matrix[y_slice, x_start:x_end] = TRAVERSIBLE
                
                # Line 23: Horizontal to B (Fixed coordinates typo from pseudocode)
                x_slice = get_slice(a_x, b_x)
                y_start = max(0, b_y - half_tunnel_width)
                y_end = min(y, b_y + half_tunnel_width + 1)
                matrix[y_start:y_end, x_slice] = TRAVERSIBLE

    return matrix, rooms


def _place_agents(
        x: int, y: int, 
        n: int, 
        rooms: list[dict], 
        victims: Grid
        ) -> np.ndarray:
    """
    Places agents in rooms.

    Parameters:
    x, y: Dimensions of the grid
    n: Number of agents
    rooms: A list of rooms with center and bounds
    victims: A matrix indicating the presence of victims (to avoid placing agents on top of victims)
    """

    agents = np.full((y, x), AGENT_NOT_PRESENT)

    room_n = len(rooms)

    for i in range(n):
        random_room = rooms[random.randint(0, room_n - 1)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = random.randint(x_range[0], x_range[1] - 1)
        random_y = random.randint(y_range[0], y_range[1] - 1)
        while (agents[random_y, random_x] == AGENT_PRESENT or victims[random_y, random_x] == VICTIM_PRESENT):
            random_x = random.randint(x_range[0], x_range[1] - 1)
            random_y = random.randint(y_range[0], y_range[1] - 1)
        agents[random_y, random_x] = AGENT_PRESENT
    
    return agents

def _place_victims(
        x: int, y: int,
        k: int, 
        rooms: list[dict], 
        agents: Grid) -> np.ndarray:
    """
    Places victims in rooms.

    Parameters:
    x, y: Dimensions of the grid
    k: Number of victims
    rooms: A list of rooms with center and bounds
    agents: A matrix indicating the presence of agents (to avoid placing victims on top of agents)
    """

    victims = np.full((y, x), VICTIM_NOT_PRESENT)
    room_n = len(rooms)

    for j in range(k):
        random_room = rooms[random.randint(0, room_n - 1)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = random.randint(x_range[0], x_range[1] - 1)
        random_y = random.randint(y_range[0], y_range[1] - 1)
        while (victims[random_y, random_x] == VICTIM_PRESENT or agents[random_y, random_x] == AGENT_PRESENT):
            random_x = random.randint(x_range[0], x_range[1] - 1)
            random_y = random.randint(y_range[0], y_range[1] - 1)
        victims[random_y, random_x] = VICTIM_PRESENT

    return victims


def visualize_grid_gen(traversability: Grid, agents: Grid, victims: Grid) -> None:
    """
    Visualizes the map, agents, and victims in a single plot.
    """
    plt.figure(figsize=(10, 10))
    
    plt.imshow(traversability.matrix, cmap='binary', interpolation='nearest')

    victim_mask = np.where(victims.matrix == VICTIM_PRESENT, 1, np.nan)
    plt.imshow(victim_mask, cmap='autumn', interpolation='nearest', alpha=1.0)
    
    agent_mask = np.where(agents.matrix == AGENT_PRESENT, 1, np.nan)
    plt.imshow(agent_mask, cmap='winter', interpolation='nearest', alpha=1.0)

    plt.title("Search & Rescue: Map, Agents (Blue), and Victims (Red)")
    plt.axis('off')
    plt.show()