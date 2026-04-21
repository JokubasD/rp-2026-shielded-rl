import random
import numpy as np
import matplotlib.pyplot as plt
from .state import State

TRAVERSIBLE = 0
UNTRAVERSIBLE = 1

AGENT_NOT_PRESENT = 0
AGENT_PRESENT = 1

VICTIM_NOT_PRESENT = 0
VICTIM_PRESENT = 1

class Simulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.ground_truth = State(width, height)
    

    def generate_ground_truth(self): # TODO add parameters to tweak generation
        # generates and sets the 2D grid with agent and victims, currently with preset values.
        self.ground_truth.traversability.matrix, rooms = generate_traversability_matrix(self.width, self.height, 4, 0, 6, 12, 6, 12, 1, 3)
        self.ground_truth.agents, self.ground_truth.victims = place_agents_and_victims(self.width, self.height, 1, 2, rooms)
        self.ground_truth.confidence.matrix = np.ones((self.height, self.width))
        return



def generate_traversability_matrix(x, y, n, u_p, w_min, w_max, l_min, l_max, t_min, t_max):
    """
    Generates 2D traversability matrix with rooms and connecting corridors, and returns the room bounds and the matrix.
    
    Parameters:
    k: Number of maps to generate
    x, y: Dimensions of the grid
    n: Number of rooms
    u_p: (unconnected_probability) The probability a room is unconnected
    t_min, t_max: Min/Max thickness of the paths
    w_min, w_max: Min/Max width of the rooms
    l_min, l_max: Min/Max length of the rooms
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

def place_agents_and_victims(x, y, n, k, rooms):
    """
    Places agents and victims in rooms.

    Parameters:
    x, y: Dimensions of the grid
    n: Number of agents
    k: Number of victims
    rooms: A list of rooms with center and bounds
    """

    agents = np.full((y, x), AGENT_NOT_PRESENT)
    victims = np.full((y, x), VICTIM_NOT_PRESENT)

    room_n = len(rooms)

    for i in range(n):
        random_room = rooms[random.randint(0, room_n - 1)]
        x_range = random_room['x_range']
        y_range = random_room['y_range']
        random_x = random.randint(x_range[0], x_range[1] - 1)
        random_y = random.randint(y_range[0], y_range[1] - 1)
        while (agents[random_y, random_x] == AGENT_PRESENT):
            random_x = random.randint(x_range[0], x_range[1] - 1)
            random_y = random.randint(y_range[0], y_range[1] - 1)
        agents[random_y, random_x] = AGENT_PRESENT
    
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
        
    return agents, victims

def visualize_grid_gen(matrix, agents, victims):
    """
    Visualizes the map, agents, and victims in a single plot.
    """
    plt.figure(figsize=(10, 10))
    
    plt.imshow(matrix, cmap='binary', interpolation='nearest')

    victim_mask = np.where(victims == VICTIM_PRESENT, 1, np.nan)
    plt.imshow(victim_mask, cmap='autumn', interpolation='nearest', alpha=1.0)
    
    agent_mask = np.where(agents == AGENT_PRESENT, 1, np.nan)
    plt.imshow(agent_mask, cmap='winter', interpolation='nearest', alpha=1.0)

    plt.title("Search & Rescue: Map, Agents (Blue), and Victims (Red)")
    plt.axis('off')
    plt.show()