from .agent import Agent
from .constants import RunOutcome

class Metric:
    def __init__(self):
        self.terrain_collisions: dict[Agent, int] = {} # Number of times agent hits a wall
        self.victim_collisions: dict[Agent, int] = {} # Number of times it hits a victim
        self.inter_agent_collisions: dict[Agent, int] = {} # Number of times it collides with another agent
        self.wait_actions: dict[Agent, int] = {}  # How many times the agent chooses to wait
        self.damage: dict[Agent, int] = {}  # (terrain collisions + victim collisions + inter-agent collisions) 
                                            # * damage per collision (for now all same)

        self.outcome: RunOutcome = RunOutcome.IN_PROGRESS
        self.steps_taken: int = 0
        self.victims_found: int = 0
        self.total_victims: int = 0
        self.time_to_first_found: int | None = None
        self.time_to_all_found: int | None = None

    def register_agent(self, agent: Agent) -> None:
        self.terrain_collisions[agent] = 0
        self.victim_collisions[agent] = 0
        self.inter_agent_collisions[agent] = 0
        self.wait_actions[agent] = 0
        self.damage[agent] = 0

    def record_terrain_collision(self, agent: Agent) -> None:
        self.terrain_collisions[agent] += 1
        self.damage[agent] += 1

    def record_victim_collision(self, agent: Agent) -> None:
        self.victim_collisions[agent] += 1
        self.damage[agent] += 1

    def record_inter_agent_collision(self, agent: Agent) -> None:
        self.inter_agent_collisions[agent] += 1
        self.damage[agent] += 1

    def record_wait(self, agent: Agent) -> None:
        self.wait_actions[agent] += 1
