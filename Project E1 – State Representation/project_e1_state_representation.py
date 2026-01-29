# Project E1 – State Representation (Perception → Belief)
# Single-cell Colab bootstrap: creates files + runs a demo.

import os, textwrap, json, random
from dataclasses import asdict

ROOT = "project_e1_state_representation"
os.makedirs(ROOT, exist_ok=True)

def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).lstrip())

# -------------------------
# state_schema.py
# -------------------------
write(f"{ROOT}/state_schema.py", """
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

Pos = Tuple[int, int]

@dataclass
class Entity:
    entity_id: str
    entity_type: str  # e.g., "goal", "obstacle", "item"
    position: Optional[Pos] = None
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    position: Pos
    orientation: str = "N"  # N, E, S, W (optional)
    energy: float = 1.0

@dataclass
class EnvironmentState:
    grid_size: Tuple[int, int]
    # Known map is an explicit belief (can contain unknowns)
    # cell values: "unknown", "empty", "obstacle", "goal"
    known_map: List[List[str]]
    # convenience lists
    known_obstacles: List[Pos] = field(default_factory=list)
    known_goals: List[Pos] = field(default_factory=list)

@dataclass
class UncertaintyState:
    unknown_cells: List[Pos] = field(default_factory=list)
    confidence_map: List[List[float]] = field(default_factory=list)  # 0.0 unknown, 1.0 fully known

@dataclass
class WorldState:
    agent: AgentState
    environment: EnvironmentState
    entities: Dict[str, Entity] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)  # e.g., {"impassable": [(x,y), ...]}
    uncertainty: UncertaintyState = field(default_factory=UncertaintyState)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # dataclasses -> dict with tuples preserved as lists (JSON-friendly)
        def convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                d = {}
                for k, v in obj.__dict__.items():
                    d[k] = convert(v)
                return d
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        return convert(self)
""")

# -------------------------
# perception.py
# -------------------------

