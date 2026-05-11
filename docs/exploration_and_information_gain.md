# Exploration and Information Gain

## Core idea

An intelligent embodied agent should not only pursue goals.

It should also seek useful information.

This repository introduces:

- frontier detection
- information gain estimation
- exploration-aware planning
- curiosity-style action selection

---

# Exploration vs exploitation

A planning system must balance two competing objectives:

| Objective | Meaning |
|---|---|
| Exploitation | pursue known goals |
| Exploration | gather new information |

Pure exploitation can trap the agent in incomplete or incorrect beliefs.

Pure exploration can prevent useful task completion.

Intelligent behavior requires balancing both.

---

# Frontiers

A frontier is:

```text
a known region adjacent to unknown space
```

Example:

```text
BELIEF MAP
? ? ? ? ?
? A . ? ?
? . ? ? ?
? ? ? ? ?
? ? ? ? G
```

The known cells near the `?` regions become exploration frontiers.

These frontiers are informative because acting near them may reveal new information.

---

# Information gain

The repository estimates information gain using:

```text
number of adjacent unknown cells
```

Example:

```text
information_gain((1,2)) = 3
```

This means the state is expected to reveal information about three unknown neighboring regions.

---

# Exploration-aware planning

The rollout planner includes exploration directly inside scoring.

```text
score =
  distance_reward
+ confidence_bonus
+ movement_bonus
+ exploration_bonus
- uncertainty_penalty
```

Where:

```text
exploration_bonus = information_gain(next_state)
```

This means the planner values:

- reaching goals
- stable predictions
- informative actions

---

# Curiosity traces

The demo exposes exploration reasoning explicitly.

Example:

```text
Exploration motive:
  selected action 'right' moves toward (3,2)
  expected information_gain=4
```

This makes exploratory cognition inspectable.

---

# Why this matters

Many toy planning systems only optimize:

```text
shortest path to goal
```

Real embodied systems often need to:

- discover environments
- resolve uncertainty
- gather information
- update internal models
- adapt dynamically

Exploration is therefore a core part of embodied intelligence.

---

# Cognitive significance

This phase introduces ideas related to:

- curiosity-driven agents
- active inference
- intrinsic motivation
- cognitive exploration
- information-seeking behavior
- frontier-based planning

The implementation here is intentionally small and interpretable.

---

# Future directions

Planned extensions:

- adaptive exploration weight
- uncertainty-aware curiosity
- persistent memory across runs
- dynamic environments
- active forgetting
- robotics deployment exploration
- embodied skill discovery

The long-term goal is to study how embodied agents build, maintain, and improve internal world understanding through interaction.
