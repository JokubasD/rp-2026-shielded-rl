"""Shared SaR map configs (sar_config) for the training runner and the map previewer."""
from src.constants import MapConfig


def sar_config(size: int, num_victims: int = 10) -> MapConfig:
    """Balanced, fully-connected SaR map of the given size with num_victims victims."""
    rooms = {20: 7, 25: 10, 30: 14}.get(size, max(6, size // 3))
    return MapConfig(
        num_rooms=rooms,
        num_victims=num_victims,
        num_agents=0,                  # the RL env places its own agent at the start cell
        unconnected_probability=0.0,   # everything reachable -> map is always solvable
        min_room_width=3, max_room_width=5,
        min_room_length=3, max_room_length=5,
        min_tunnel_thickness=1, max_tunnel_thickness=1,
        initial_fire_points=1, fire_spread_rate=0.03, fire_duration=8,
        room_vulnerability_probability=0.12, room_vulnerability_severity=0.3,
        tunnel_vulnerability_probability=0.08, tunnel_vulnerability_severity=0.3,
    )
