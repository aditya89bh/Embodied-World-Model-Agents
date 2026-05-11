# Embodied World Model Architecture

## Core idea

The agent does not directly operate on reality.

It operates on an internal model of reality.

That distinction is critical for embodied intelligence.

A world-model agent:

1. maintains an internal belief state
2. predicts action outcomes
3. acts in the environment
4. observes real outcomes
5. measures prediction error
6. stores experience
7. updates future predictions

This repo implements a minimal version of that loop.

---

# System components

## 1. GridWorld environment

The environment defines the actual world dynamics.

In the demo:

- the world is a 5x5 grid
- movement actions exist
- a hidden blocked cell exists
- the agent initially does not know about it

This creates prediction mismatch.

---

## 2. Tabular world model

The world model predicts:

```text
(state, action) -> predicted next state
```

Initially the model uses a fallback assumption.

After repeated experience:

```text
(state=(1,1), action=right) -> blocked
```

becomes part of memory.

The model adapts through stored transition counts.

---

## 3. WorldModelAgent

The agent performs the main loop:

```text
predict
-> act
-> observe
-> compare
-> update
```

This creates learning through interaction.

---

# Prediction error

Prediction error is the gap between:

```text
predicted next state
vs
actual next state
```

This is the core learning signal.

Without prediction error:

- the system cannot adapt
- the system cannot improve its internal model
- the system cannot learn embodiment constraints

Prediction error is one of the central mechanisms behind embodied intelligence.

---

# Memory

The world model stores transition statistics.

Example:

```text
(1,1) + right -> (1,1)
count = 5
```

This memory changes future predictions.

The system therefore becomes:

```text
experience dependent
```

instead of purely reactive.

---

# Robotics mapping

| Toy environment | Robotics equivalent |
|---|---|
| Grid position | Robot pose |
| Hidden block | Collision / fixture / workspace constraint |
| Action | Robot motion primitive |
| Prediction | Expected robot outcome |
| Observation | Sensor feedback |
| Prediction error | Unexpected robot behavior |
| Transition memory | Deployment memory |
| Adaptation | Updated operational model |

---

# Why this matters

Most agent systems today focus heavily on:

- prompting
- reasoning
- tool calling

Embodied systems require something deeper:

```text
predictive interaction with reality
```

That requires:

- world models
- memory
- adaptation
- uncertainty handling
- feedback loops

This repository studies those primitives in a small and interpretable form.
