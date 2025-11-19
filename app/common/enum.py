from enum import Enum


class TaskStatus(str, Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"