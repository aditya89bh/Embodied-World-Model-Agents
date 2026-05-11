# Belief vs Reality

## Core idea

An embodied agent should not directly know the full world.

Instead, the system should maintain:

- an external reality
- an internal belief about reality

Those two representations are not always identical.

This repository separates them explicitly.

---

# Reality map

The reality map represents the actual environment.

Example:

```text
REALITY MAP
. . . . .
. . X . .
. S . . .
. . . . .
. . . . G
```

Reality exists independently of the agent.

The agent cannot directly access the full map.

---

# Belief map

The belief map represents:

```text
what the agent currently thinks is true
```

Example:

```text
BELIEF MAP
? ? ? ? ?
? A X ? ?
? . . ? ?
? ? ? ? ?
? ? ? ? G
```

Symbols:

| Symbol | Meaning |
|---|---|
| ? | unknown |
| . | known free cell |
| X | discovered obstacle |
| G | known goal |
| A | current agent position |

---

# Partial observability

The agent only receives local observations.

Example:

```python
observations = env.observe(position, radius=1)
```

This means:

- distant regions remain unknown
- the agent must explore
- knowledge grows through interaction

This is much closer to real embodied systems.

---

# Belief updates

Observations update the internal belief map.

Example:

```text
belief_update:
  discovered obstacle at (2,1)
```

The agent therefore constructs an internal world representation over time.

---

# Why this matters

Many toy agent systems assume:

```text
perfect world access
```

Real embodied systems rarely have that.

Robots operate with:

- limited sensing
- noisy perception
- incomplete information
- changing environments

Embodied intelligence therefore requires:

- internal belief states
- uncertainty handling
- memory-guided updates
- exploration

---

# Cognitive significance

Belief maps are foundational for:

- cognitive mapping
- exploration behavior
- active inference
- POMDP-style reasoning
- memory-based navigation
- embodied learning

This repository implements a small and interpretable version of those ideas.

---

# Future directions

Planned extensions:

- exploration reward
- information gain planning
- uncertainty-aware belief updates
- persistent memory
- dynamic environments
- deployment drift
- robotics skill abstraction

The long-term goal is to study how embodied agents maintain and adapt internal models of the world over time.
