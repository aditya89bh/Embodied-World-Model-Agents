# Stochastic World Models

## Deterministic vs stochastic prediction

A deterministic world model assumes:

```text
(state, action) -> one next state
```

Example:

```text
(1,2) + right -> (2,2)
```

This is simple but unrealistic for embodied systems.

Real environments contain:

- sensor noise
- motion uncertainty
- partial observability
- changing conditions
- inconsistent outcomes

A stochastic world model instead predicts:

```text
(state, action) -> probability distribution over next states
```

Example:

```text
(1,2) + right:
  (2,2): 0.65
  (1,3): 0.20
  (1,2): 0.15
```

---

# Why stochasticity matters

Embodied systems interact with reality.

Reality is noisy.

A robot action does not always produce the same exact result.

Examples:

| Situation | Source of uncertainty |
|---|---|
| Grasping | object slip |
| Navigation | wheel slip |
| Manipulation | fixture tolerance |
| Vision | sensor noise |
| Human interaction | unpredictable movement |

A useful embodied agent therefore needs:

- uncertainty estimation
- probabilistic prediction
- adaptive planning
- confidence-aware reasoning

---

# Transition distributions

The world model stores transition statistics from experience.

Example:

```text
(1,2) + right:
  (2,2): 13
  (1,3): 4
  (1,2): 3
```

These counts are normalized into probabilities.

---

# Confidence

Confidence is defined as:

```text
max(probability distribution)
```

Example:

```text
(2,2): 0.65
(1,3): 0.20
(1,2): 0.15

confidence = 0.65
```

High confidence means the model expects a dominant outcome.

---

# Uncertainty

Uncertainty is measured using entropy.

High entropy means:

- outcomes are spread across multiple futures
- the environment is less predictable
- planning becomes riskier

Low entropy means:

- outcomes are concentrated
- the environment is stable
- planning is easier

---

# Uncertainty-aware planning

The rollout planner evaluates:

```text
score =
  distance_reward
+ confidence_bonus
+ movement_bonus
- uncertainty_penalty
```

This means the planner prefers:

- goal progress
- stable transitions
- predictable actions

and avoids:

- highly uncertain futures
- unstable outcomes

---

# Robotics relevance

| Toy concept | Robotics equivalent |
|---|---|
| Slippery cell | wheel slip / unstable manipulation |
| Transition distribution | uncertain robot outcome |
| Confidence | reliability estimate |
| Entropy | operational uncertainty |
| Uncertainty penalty | risk-aware planning |

This connects world-model planning to real embodied systems.

---

# Why this repo uses interpretable models

The goal of this repository is not benchmark performance.

The goal is to make:

- prediction
- memory
- uncertainty
- adaptation
- planning

visible and understandable.

The systems are intentionally small so the internal cognition loop can be inspected directly.
