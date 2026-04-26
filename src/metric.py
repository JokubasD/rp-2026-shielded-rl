from .agent import Agent

class Metric():
    """
    What metrics do we even want here?

    per agent or overall? - maria & tigo?

    "bad" metrics: terrain collisions, victims run over, agent collisions, fire?
    "good" metrics: victims found, what else?
    """

    terrain_collisions: dict[Agent, int]
    victim_collisions: dict[Agent, int]
    inter_agent_collisions: int
    victims_found: int