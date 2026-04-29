from copy import deepcopy
from dataclasses import dataclass
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .grid import Grid
from .state import State
from .agent import *
from .fire_manager import FireManager
from .constants import *

@dataclass
class MapConfig:
    num_rooms: int = 4
    unconnected_probability: float = 0.0
    room_vulnerability_probability: float = 0.3
    room_vulnerability_severity: float = 0.4
    tunnel_vulnerability_probability: float = 0.3
    tunnel_vulnerability_severity: float = 0.4
    initial_fire_points: int = 1
    fire_spread_rate: float = 0.3
    start_room_width: int = 3
    start_room_length: int = 3
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
        self.fire_manager: FireManager = FireManager(width, height, 0.0) #? Start with a non-spreading fire manager, wondering if it is the right way

    def add_agent(self, agent: Agent) -> None:
        """
        Adds an agent to the simulation and updates the ground truth

        Parameters:
        agent: The agent to add
        """
        self.agents.append(agent)
        self.ground_truth.agents[agent.y][agent.x] = 1

    def step(self) -> State:
        """
        Performs a single step of the simulation.
        This includes agent actions, environment actions, and updating the ground truth.

        Returns:
        The ground truth state after the step.
        """
        # Perform agent actions
        for agent in self.agents:
            action = agent.get_action()
            is_move = action < 4
            
            if is_move:
                self.ground_truth.agents[agent.y][agent.x] = 0
                agent.move(action)
                self.ground_truth.agents[agent.y][agent.x] = 1
            
            agent.scan(self.ground_truth)

        # Perform environment actions (firespread, etc.)
        self.fire_manager.spread_fire(self.ground_truth)

        # Don't let the returned state modify current state
        result = deepcopy(self.ground_truth)
        return result 

    def run(self, steps: int) -> list[State]:
        """
        Performs a number of steps of the simulation.

        Parameters:
        steps: The number of steps to perform

        Returns:
        A list of the ground truth states after each step.
        list[0] is the initial state before the sim is run
        """
        record: list[State] = []
        record.append(deepcopy(self.ground_truth))
        for _ in range(steps):
            record.append(self.step())
        return record
    
    def generate_ground_truth(self, config: MapConfig | None = None) -> None:
        if config is None:
            config = MapConfig()
        # generates and sets the 2D grid with agent and victims, currently with preset values.
        self.ground_truth.traversability.matrix, rooms, tunnels = _generate_traversability_matrix(self.width, self.height, 
                                                                                        config.num_rooms, config.unconnected_probability, 
                                                                                        config.start_room_width, config.start_room_length,
                                                                                        config.min_room_width, config.max_room_width, 
                                                                                        config.min_room_length, config.max_room_length, 
                                                                                        config.min_tunnel_thickness, config.max_tunnel_thickness)
        self.ground_truth.vulnerability.matrix = _generate_vulnerability_matrix(self.width, self.height, rooms, tunnels,
                                                                                config.room_vulnerability_probability, config.tunnel_vulnerability_probability,
                                                                                config.room_vulnerability_severity, config.tunnel_vulnerability_severity)
        self.ground_truth.agents.matrix = _place_agents(self.width, self.height, config.num_agents, rooms, self.ground_truth.victims)
        self.ground_truth.victims.matrix = _place_victims(self.width, self.height, config.num_victims, rooms, self.ground_truth.agents)

        self.fire_manager = FireManager(self.width, self.height, config.fire_spread_rate)
        self.fire_manager.initialize_fire(self.ground_truth, config.initial_fire_points, rooms)
        
        self.ground_truth.confidence.matrix = np.ones((self.height, self.width))
        return

def _generate_traversability_matrix(
        x: int, y: int,
        n: int, u_p: float,
        s_width: int, s_length: int,
        w_min: int, w_max: int,
        l_min: int, l_max: int,
        t_min: int, t_max: int
        ) -> tuple[np.ndarray, list[dict], list[dict]]:
    """
    Generates 2D traversability matrix with rooms and connecting tunnels, and returns the room and tunnel bounds and the matrix.
    
    Parameters:
    x, y: Dimensions of the grid
    n: Number of rooms (outside of first room in top left)
    s_width, s_length: Dimensions of the first room in the top left
    u_p: (unconnected_probability) The probability a room is unconnected
    w_min, w_max: Min/Max width of the rooms
    l_min, l_max: Min/Max length of the rooms
    t_min, t_max: Min/Max thickness of the tunnels
    """

    if not (0 <= u_p <= 1):
        raise ValueError(f"probability of unconnected rooms should be between 0 and 1 but was: {u_p}")
    
    matrix = np.full((y, x), TraversabilityLevel.UNTRAVERSIBLE)
    room_seeds = np.zeros((2, n + 1), dtype=int)
    rooms = []
    tunnels = []

    tunnel_width = random.randint(t_min, t_max)
    half_tunnel_width = tunnel_width // 2

    # create first room in top left corner
    c = s_width // 2
    f = s_length // 2
    room_seeds[0][0] = c
    room_seeds[1][0] = f
    x_start = 0
    x_end = s_width
    y_start = 0
    y_end = s_length
    matrix[y_start:y_end, x_start:x_end] = TraversabilityLevel.TRAVERSIBLE
    #? Should we store the first room in the rooms list?

    # create rooms and store their centers in room_seeds
    for p in range(1, n + 1):
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

        matrix[y_start:y_end, x_start:x_end] = TraversabilityLevel.TRAVERSIBLE
        rooms.append({
            'x_range': (x_start, x_end),
            'y_range': (y_start, y_end),
            'center': (c, f)
        })
    
    # connect rooms with tunnels, skipping some based on unconnected_probability
    for q in range(n):
        connect = random.random() >= u_p

        if (connect):
            a_x, a_y = room_seeds[0][q], room_seeds[1][q]
            b_x, b_y = room_seeds[0][q + 1], room_seeds[1][q + 1]

            def get_bounds(start, end) -> tuple[int, int]:
                return (min(start, end), max(start, end) + 1)
            
            direction = random.randint(0, 1)
            if (direction == 1): # horizontal first, then vertical # TODO probably should also return the tunnels same as with rooms
                hx_bounds = get_bounds(a_x, b_x)
                hy_bounds = (max(0, a_y - half_tunnel_width), min(y, a_y + half_tunnel_width + 1))
                matrix[hy_bounds[0]:hy_bounds[1], hx_bounds[0]:hx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE
                
                vx_bounds = (max(0, b_x - half_tunnel_width), min(x, b_x + half_tunnel_width + 1))
                vy_bounds = get_bounds(a_y, b_y)
                matrix[vy_bounds[0]:vy_bounds[1], vx_bounds[0]:vx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE
                
                tunnels.append({
                    'horizontal_x_range': hx_bounds,
                    'horizontal_y_range': hy_bounds,
                    'vertical_x_bounds': vx_bounds,
                    'vertical_y_bounds': vy_bounds
                })
            else: # vertical first, then horizontal
                vx_bounds = (max(0, a_x - half_tunnel_width), min(x, a_x + half_tunnel_width + 1))
                vy_bounds = get_bounds(a_y, b_y)
                matrix[vy_bounds[0]:vy_bounds[1], vx_bounds[0]:vx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE
                
                hx_bounds = get_bounds(a_x, b_x)
                hy_bounds = (max(0, b_y - half_tunnel_width), min(y, b_y + half_tunnel_width + 1))
                matrix[hy_bounds[0]:hy_bounds[1], hx_bounds[0]:hx_bounds[1]] = TraversabilityLevel.TRAVERSIBLE

                tunnels.append({
                    'horizontal_x_range': hx_bounds,
                    'horizontal_y_range': hy_bounds,
                    'vertical_x_bounds': vx_bounds,
                    'vertical_y_bounds': vy_bounds
                })

    return matrix, rooms, tunnels

def _generate_vulnerability_matrix(
        x: int, y: int, 
        rooms: list[dict],
        tunnels: list[dict],
        r_v_p: float, t_v_p: float,
        r_v_s: float, t_v_s: float
        ) -> np.ndarray:
    """
    Generates 2D vulnerability matrix for rooms and tunnels based on the provided probabilities.
    
    Parameters:
    x, y: Dimensions of the grid
    rooms, tunnels: the rooms and tunnels generated by _generate_traversability_matrix
    r_v_p, t_v_p: room and tunnel vulnerability probability, respectively. Dictates how probable it is a room will be either high risk or vulnerable.
    r_v_s, t_v_s: room and tunnel vulnerability severity, respectively. A higher value results in a higher chance of high risk.
    """
    if not (0 <= r_v_p <= 1 and 0 <= t_v_p <= 1 and 0 <= r_v_s <= 1 and 0 <= t_v_s <= 1):
        raise ValueError("One of the vulnerability probabilities is not between 0 and 1.")

    vulnerability = np.full((y, x), VulnerabilityLevel.SAFE)
    for tunnel in tunnels:
        vulnerable = random.random() <= t_v_p
        if not (vulnerable):
            continue
        vulnerability_level = VulnerabilityLevel.VULNERABLE
        if (random.random() <= t_v_s):
            vulnerability_level = VulnerabilityLevel.HIGH_RISK
        
        hx = tunnel['horizontal_x_range']
        hy = tunnel['horizontal_y_range']
        vulnerability[hy[0]:hy[1], hx[0]:hx[1]] = vulnerability_level
        
        vx = tunnel['vertical_x_bounds']
        vy = tunnel['vertical_y_bounds']
        vulnerability[vy[0]:vy[1], vx[0]:vx[1]] = vulnerability_level

    for room in rooms:
        vulnerable = random.random() <= r_v_p
        if not (vulnerable):
            continue
        vulnerability_level = VulnerabilityLevel.VULNERABLE
        if (random.random() <= r_v_s):
            vulnerability_level = VulnerabilityLevel.HIGH_RISK
        vulnerability[room['y_range'][0]:room['y_range'][1], room['x_range'][0]:room['x_range'][1]] = vulnerability_level

    return vulnerability

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

    agents = np.full((y, x), AgentPresence.AGENT_NOT_PRESENT)

    room_n = len(rooms)

    for _ in range(n):
        random_room = rooms[random.randint(0, room_n - 1)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = random.randint(x_range[0], x_range[1] - 1)
        random_y = random.randint(y_range[0], y_range[1] - 1)
        while (agents[random_y, random_x] == AgentPresence.AGENT_PRESENT or victims[random_y, random_x] == VictimPresence.VICTIM_PRESENT):
            random_x = random.randint(x_range[0], x_range[1] - 1)
            random_y = random.randint(y_range[0], y_range[1] - 1)
        agents[random_y, random_x] = AgentPresence.AGENT_PRESENT
    
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

    victims = np.full((y, x), VictimPresence.VICTIM_NOT_PRESENT)
    room_n = len(rooms)

    for _ in range(k):
        random_room = rooms[random.randint(0, room_n - 1)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = random.randint(x_range[0], x_range[1] - 1)
        random_y = random.randint(y_range[0], y_range[1] - 1)
        while (victims[random_y, random_x] == VictimPresence.VICTIM_PRESENT or agents[random_y, random_x] == AgentPresence.AGENT_PRESENT):
            random_x = random.randint(x_range[0], x_range[1] - 1)
            random_y = random.randint(y_range[0], y_range[1] - 1)
        victims[random_y, random_x] = VictimPresence.VICTIM_PRESENT

    return victims


def visualize_grid_gen(traversability: Grid, agents: Grid, victims: Grid, vulnerability: Grid, fire: Grid) -> None:
    """
    Visualizes the map, vulnerability, fire, agents, and victims in a single plot.
    """
    plt.figure(figsize=(7, 7))
    
    plt.imshow(traversability.matrix, cmap='binary', interpolation='nearest')

    victim_mask = np.where(victims.matrix == VictimPresence.VICTIM_PRESENT, 1, np.nan)
    plt.imshow(victim_mask, cmap='autumn', interpolation='nearest', alpha=1.0)
    
    agent_mask = np.where(agents.matrix == AgentPresence.AGENT_PRESENT, 1, np.nan)
    plt.imshow(agent_mask, cmap='winter', interpolation='nearest', alpha=1.0)

    vulnerability_mask = np.where(vulnerability.matrix > VulnerabilityLevel.SAFE.value, vulnerability.matrix, np.nan)
    plt.imshow(vulnerability_mask, cmap='autumn_r', interpolation='nearest', alpha=0.3, vmin=0.5, vmax=1)
    vuln_cmap = plt.cm.autumn_r
    
    flammable_mask = np.where(fire.matrix == FireLevel.FLAMMABLE.value, 1, np.nan)
    burning_mask = np.where(fire.matrix == FireLevel.BURNING.value, 1, np.nan)
    burnt_mask = np.where(fire.matrix == FireLevel.BURNT.value, 1, np.nan)

    plt.imshow(flammable_mask, cmap=mcolors.ListedColormap(['gold']), interpolation='nearest', alpha=0.3)
    plt.imshow(burning_mask, cmap=mcolors.ListedColormap(['darkorange']), interpolation='nearest', alpha=0.85)
    plt.imshow(burnt_mask, cmap=mcolors.ListedColormap(['dimgray']), interpolation='nearest', alpha=0.9)

    legend_elements = [
        mpatches.Patch(facecolor='black', edgecolor='black', label='Untraversable (Wall)'),
        mpatches.Patch(facecolor='white', edgecolor='gray', label='Traversable (Floor)'),
        mpatches.Patch(facecolor='blue', label='Agent'),
        mpatches.Patch(facecolor='red', label='Victim'),
        mpatches.Patch(facecolor=vuln_cmap(0.6), alpha=0.6, edgecolor='gray', label='Vulnerable'),
        mpatches.Patch(facecolor=vuln_cmap(1.0), alpha=0.6, edgecolor='gray', label='High Risk'),
        mpatches.Patch(facecolor='gold', alpha=0.5, edgecolor='gray', label='Flammable'),
        mpatches.Patch(facecolor='darkorange', alpha=0.85, edgecolor='darkred', label='Burning'),
        mpatches.Patch(facecolor='dimgray', alpha=0.9, edgecolor='black', label='Burnt')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    rows, cols = traversability.matrix.shape
    plt.xticks(np.arange(-0.5, cols, 1), [])
    plt.yticks(np.arange(-0.5, rows, 1), [])
    plt.grid(color='gray', linewidth=0.1)   
    plt.tick_params(bottom=False, left=False)

    plt.title("Search & Rescue Map")
    plt.tight_layout()
    plt.show()