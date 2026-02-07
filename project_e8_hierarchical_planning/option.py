"""
Option Definition (Skill / Temporal Abstraction)

An Option represents a temporally extended action, also known as a skill.

Instead of choosing primitive actions at every timestep, the agent can
select an option that:
- decides when it can start
- controls actions over multiple steps
- decides when it should terminate

This is the core building block of hierarchical planning.
"""

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Option:
    """
    A temporally extended action (skill).

    Components:
    - initiation(state) -> bool
        Determines whether the option is applicable in the current state.

    - policy(state) -> action
        Returns the primitive action to execute while the option is active.

    - termination(state) -> bool
        Determines when the option should stop execution.
    """
    name: str
    initiation: Callable[[Any], bool]
    policy: Callable[[Any], str]
    termination: Callable[[Any], bool]
