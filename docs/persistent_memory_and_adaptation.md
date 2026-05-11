# Persistent Memory and Adaptation

## Core idea

An embodied agent should not forget everything after a single execution.

If the agent discovers an obstacle, learns that a transition is unreliable, or builds a partial map of the environment, that experience should survive across runs.

This repository introduces persistent memory so the agent can accumulate experience over time.

---

# What gets persisted

The system persists two main forms of memory:

| Memory type | Meaning |
|---|---|
| Belief map | What the agent has discovered about the world |
| World model | Learned transition outcomes and counts |

Together, these allow the agent to reload prior experience and continue adapting.

---

# Belief map persistence

The belief map stores the agent's internal representation of the environment.

It includes:

- explored cells
- unknown cells
- discovered obstacles
- known goal location

Example:

```json
{
  "width": 5,
  "height": 5,
  "goal": [4, 4],
  "cells": [
    ["?", "?", "?", "?", "?"],
    ["?", ".", "X", "?", "?"],
    ["?", ".", ".", "?", "?"],
    ["?", "?", "?", "?", "?"],
    ["?", "?", "?", "?", "G"]
  ]
}
```

---

# World model persistence

The world model stores transition statistics.

Example:

```json
{
  "1,1|right": {
    "1,1": 4
  },
  "1,2|right": {
    "2,2": 6,
    "1,3": 2,
    "1,2": 1
  }
}
```

This means the agent remembers how actions behaved in specific states.

---

# Cross-session adaptation

Without persistence:

```text
run starts -> agent learns -> run ends -> memory disappears
```

With persistence:

```text
run starts -> previous memory loads -> agent continues learning -> updated memory saves
```

This is the foundation of deployment adaptation.

---

# Demo trace

The demo exposes persistent memory explicitly.

Example:

```text
Persistent memory status:
  - persistent memory detected
  - explored_cells=18
  - known_obstacles=2
  - learned_transitions=37
```

This makes accumulated experience visible.

---

# Robotics relevance

| Persistent memory concept | Robotics equivalent |
|---|---|
| Explored cells | Known workspace regions |
| Discovered obstacles | Fixtures, machine boundaries, blocked zones |
| Transition counts | Reliability of motion primitives |
| Persistent belief map | Robot's remembered workspace model |
| Persistent world model | Deployment memory across cycles |

A real robot deployed in a factory should not start from zero every time.

It should remember:

- which motions are reliable
- which areas are risky
- where failures occurred
- what the workspace looked like
- how the environment changed over time

---

# Why this matters

Persistent memory moves the system from:

```text
single-session intelligence
```

to:

```text
long-term adaptation
```

That shift is central to embodied AI and robotics.

---

# Future directions

Planned extensions:

- memory decay
- confidence decay
- stale memory detection
- environment drift adaptation
- memory compression
- episodic deployment logs
- robotics skill memory

The long-term direction is to study how embodied agents accumulate operational intelligence through repeated interaction.
