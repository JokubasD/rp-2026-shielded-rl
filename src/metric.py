from enum import Enum
from .agent import Agent


class RunOutcome(Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    TIMEOUT = "timeout"


class Metric:
    pass