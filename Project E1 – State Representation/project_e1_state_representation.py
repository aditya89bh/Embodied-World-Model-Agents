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

write(f"{ROOT}/perception.py", """
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Pos = Tuple[int, int]

@dataclass
class Observation:
    # visible cells in agent's local window
    # mapping position -> cell_type ("empty", "obstacle", "goal")
    visible: Dict[Pos, str]
    agent_position: Pos
    grid_size: Tuple[int, int]
    timestep: int

def get_local_observation(grid: List[List[str]], agent_pos: Pos, radius: int, timestep: int) -> Observation:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    ax, ay = agent_pos
    visible: Dict[Pos, str] = {}

    for y in range(max(0, ay - radius), min(h, ay + radius + 1)):
        for x in range(max(0, ax - radius), min(w, ax + radius + 1)):
            cell = grid[y][x]
            # Normalize to our observation vocabulary
            if cell == "#":
                visible[(x, y)] = "obstacle"
            elif cell == "G":
                visible[(x, y)] = "goal"
            else:
                visible[(x, y)] = "empty"

    return Observation(
        visible=visible,
        agent_position=agent_pos,
        grid_size=(w, h),
        timestep=timestep,
    )
""")

# -------------------------
# encoder.py
# -------------------------
write(f"{ROOT}/encoder.py", """
from __future__ import annotations
from typing import Dict, List, Tuple
from state_schema import WorldState, AgentState, EnvironmentState, UncertaintyState, Entity

Pos = Tuple[int, int]

def init_belief(grid_size: Tuple[int, int]) -> Tuple[List[List[str]], List[List[float]]]:
    w, h = grid_size
    known_map = [["unknown" for _ in range(w)] for __ in range(h)]
    conf_map = [[0.0 for _ in range(w)] for __ in range(h)]
    return known_map, conf_map

def update_belief(known_map: List[List[str]], conf_map: List[List[float]], visible: Dict[Pos, str]) -> None:
    for (x, y), cell_type in visible.items():
        known_map[y][x] = cell_type
        conf_map[y][x] = 1.0

def derive_lists(known_map: List[List[str]]) -> Tuple[List[Pos], List[Pos], List[Pos]]:
    unknown = []
    obstacles = []
    goals = []
    h = len(known_map)
    w = len(known_map[0]) if h else 0
    for y in range(h):
        for x in range(w):
            v = known_map[y][x]
            if v == "unknown":
                unknown.append((x, y))
            elif v == "obstacle":
                obstacles.append((x, y))
            elif v == "goal":
                goals.append((x, y))
    return unknown, obstacles, goals

def build_world_state(
    *,
    agent_position: Pos,
    grid_size: Tuple[int, int],
    known_map: List[List[str]],
    conf_map: List[List[float]],
    timestep: int,
    source: str = "gridworld",
) -> WorldState:
    unknown_cells, known_obstacles, known_goals = derive_lists(known_map)

    env = EnvironmentState(
        grid_size=grid_size,
        known_map=known_map,
        known_obstacles=known_obstacles,
        known_goals=known_goals,
    )

    uncertainty = UncertaintyState(
        unknown_cells=unknown_cells,
        confidence_map=conf_map,
    )

    # Entities are optional at E1, but we add known goals as entities for downstream compatibility.
    entities: Dict[str, Entity] = {}
    for i, (x, y) in enumerate(known_goals):
        entities[f"goal_{i}"] = Entity(entity_id=f"goal_{i}", entity_type="goal", position=(x, y), properties={})

    constraints = {
        "impassable": list(known_obstacles),
        "grid_bounds": {"width": grid_size[0], "height": grid_size[1]},
    }

    ws = WorldState(
        agent=AgentState(position=agent_position),
        environment=env,
        entities=entities,
        constraints=constraints,
        uncertainty=uncertainty,
        metadata={"timestep": timestep, "source": source},
    )
    return ws
""")

# -------------------------
# examples.py
# -------------------------

write(f"{ROOT}/examples.py", """
from __future__ import annotations
from typing import List, Tuple
from perception import get_local_observation
from encoder import init_belief, update_belief, build_world_state

Pos = Tuple[int, int]

def render_true_grid(grid: List[List[str]], agent_pos: Pos) -> str:
    h = len(grid)
    w = len(grid[0]) if h else 0
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            if (x, y) == agent_pos:
                row.append("A")
            else:
                row.append(grid[y][x])
        out.append("".join(row))
    return "\\n".join(out)

def render_known_map(known_map: List[List[str]], agent_pos: Pos) -> str:
    # known_map uses: unknown/empty/obstacle/goal
    h = len(known_map)
    w = len(known_map[0]) if h else 0
    glyph = {"unknown": "?", "empty": ".", "obstacle": "#", "goal": "G"}
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            if (x, y) == agent_pos:
                row.append("A")
            else:
                row.append(glyph.get(known_map[y][x], "?"))
        out.append("".join(row))
    return "\\n".join(out)

def step_agent(agent_pos: Pos, action: str, grid: List[List[str]]) -> Pos:
    # Actions: "up", "down", "left", "right", "stay"
    x, y = agent_pos
    dx, dy = 0, 0
    if action == "up": dy = -1
    elif action == "down": dy = 1
    elif action == "left": dx = -1
    elif action == "right": dx = 1
    nx, ny = x + dx, y + dy

    h = len(grid)
    w = len(grid[0]) if h else 0
    if not (0 <= nx < w and 0 <= ny < h):
        return agent_pos
    if grid[ny][nx] == "#":
        return agent_pos
    return (nx, ny)

def demo():
    # True grid:
    # . empty, # obstacle, G goal
    grid = [
        list("..#..."),
        list("..#G.."),
        list("..#..."),
        list("......"),
        list("...#.."),
        list("......"),
    ]
    agent_pos = (0, 0)
    radius = 1

    known_map, conf_map = init_belief((len(grid[0]), len(grid)))

    actions = ["right", "right", "down", "down", "down", "right", "right", "up", "up", "right"]

    for t, a in enumerate(actions):
        obs = get_local_observation(grid, agent_pos, radius=radius, timestep=t)
        update_belief(known_map, conf_map, obs.visible)

        ws = build_world_state(
            agent_position=agent_pos,
            grid_size=obs.grid_size,
            known_map=known_map,
            conf_map=conf_map,
            timestep=t,
            source="gridworld_demo",
        )

        print("=" * 60)
        print(f"t={t} | action={a} | agent_pos={agent_pos}")
        print("\\nTRUE WORLD")
        print(render_true_grid(grid, agent_pos))
        print("\\nAGENT BELIEF (KNOWN MAP)")
        print(render_known_map(known_map, agent_pos))
        print("\\nBELIEF SUMMARY")
        print(f"Known obstacles: {len(ws.environment.known_obstacles)} | Known goals: {len(ws.environment.known_goals)} | Unknown cells: {len(ws.uncertainty.unknown_cells)}")

        # move at end of step
        agent_pos = step_agent(agent_pos, a, grid)

    print("=" * 60)
    print("Demo complete.")

if __name__ == "__main__":
    demo()
""")

# -------------------------
# tests.py
# -------------------------

write(f"{ROOT}/tests.py", """
from __future__ import annotations
from perception import get_local_observation
from encoder import init_belief, update_belief, build_world_state

def test_belief_updates():
    grid = [
        list(".."),
        list(".#"),
    ]
    agent_pos = (0, 0)
    known_map, conf_map = init_belief((2, 2))

    obs = get_local_observation(grid, agent_pos, radius=1, timestep=0)
    update_belief(known_map, conf_map, obs.visible)

    ws = build_world_state(
        agent_position=agent_pos,
        grid_size=obs.grid_size,
        known_map=known_map,
        conf_map=conf_map,
        timestep=0,
        source="test",
    )

    assert ws.environment.grid_size == (2, 2)
    assert conf_map[0][0] == 1.0
    assert ws.uncertainty.confidence_map[0][0] == 1.0
    assert (1, 1) in ws.environment.known_obstacles

if __name__ == "__main__":
    test_belief_updates()
    print("✅ tests passed")
""")

# -------------------------
# Run demo + tests
# -------------------------
print(f"✅ Project files created in: {ROOT}/")
print("Running quick tests...")
import subprocess, sys, pathlib

# Run tests
subprocess.check_call([sys.executable, f"{ROOT}/tests.py"])

print("\nRunning demo...\n")
subprocess.check_call([sys.executable, f"{ROOT}/examples.py"])
