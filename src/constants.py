from enum import Enum, IntEnum

class AgentAction(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4

class FireLevel(IntEnum):
    SAFE = 0
    FLAMMABLE = 1
    BURNING = 2
    BURNT = 3

class VulnerabilityLevel(float, Enum): #? Maybe create a separate TUNNEL vulnerability level?
    SAFE = 0.0
    VULNERABLE = 0.6
    HIGH_RISK = 1.0

class TraversabilityLevel(IntEnum):
    TRAVERSIBLE = 0
    UNTRAVERSIBLE = 1

class AgentPresence(IntEnum):
    AGENT_NOT_PRESENT = 0
    AGENT_PRESENT = 1

class VictimPresence(IntEnum):
    VICTIM_NOT_PRESENT = 0
    VICTIM_PRESENT = 1
