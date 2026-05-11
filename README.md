# Embodied World Model Agents

A minimal, interpretable repository for studying how embodied agents build internal models of the world, predict action outcomes, compare predictions with reality, store experience, and adapt over time.

This repo is not a chatbot-agent playground. It focuses on the core loop required for physical intelligence:

```text
belief -> prediction -> action -> observation -> prediction error -> memory -> model update
```

## Why this matters

An embodied agent does not only need to reason. It needs to understand how its actions change the world.

For robots and physical AI systems, intelligence depends on the ability to:

- maintain a belief state
- predict what should happen next
- act under uncertainty
- detect mismatch between prediction and reality
- remember experience
- update future predictions

This is the foundation of world-model based embodied intelligence.

## Current demo

The repo includes a simple grid-world demo where an agent tries to move right. The agent initially predicts that the move will succeed. The real environment contains a hidden blocked cell, so the action fails. The agent stores the actual outcome and updates its tabular world model. On later attempts, its prediction becomes more accurate.

Run:

```bash
pip install -r requirements.txt
python examples/run_world_model_demo.py
```

Run tests:

```bash
pytest
```

## Core architecture

```text
GridWorld environment
        |
        v
WorldModelAgent
        |
        | predicts using
        v
TabularWorldModel
        |
        | acts in environment
        v
Actual outcome
        |
        | compare predicted vs actual
        v
Prediction error
        |
        | update model
        v
Improved future prediction
```

## Repo structure

```text
embodied_world_models/
├── __init__.py
├── state.py
├── environment.py
├── world_model.py
└── agent.py

examples/
└── run_world_model_demo.py

tests/
└── test_world_model_demo.py

project_e4_imagination_rollouts/
project_e5_experience_memory/
project_e6_world_model_adaptation/
```

The `embodied_world_models/` package is the clean runnable layer.

The numbered `project_e*` folders preserve the earlier learning modules and experiments.

## Main concepts

| Concept | Meaning in this repo |
|---|---|
| Belief state | What the agent currently thinks about the world |
| World model | Internal prediction system for action outcomes |
| Action | A movement command such as `right`, `left`, `up`, `down` |
| Observation | What actually happened after acting |
| Prediction error | Mismatch between predicted and actual outcome |
| Experience memory | Stored transitions from past action attempts |
| Adaptation | Updating the model so future predictions improve |

## Robotics mapping

| Grid-world concept | Robotics equivalent |
|---|---|
| Agent position | Robot TCP or end-effector pose |
| Grid cell | Workspace region |
| Action | Motion primitive or skill |
| Hidden block | Fixture, collision zone, machine constraint, or unexpected obstacle |
| Prediction | Expected result of a robot action |
| Actual outcome | Sensor/state feedback after execution |
| Prediction error | Difference between expected and observed result |
| World model update | Robot learning from deployment experience |

This toy system is intentionally small. The point is not grid-world performance. The point is to make the embodied prediction loop visible.

## Learning progression

The repository is designed to grow through staged exercises:

1. Represent the agent's state.
2. Separate belief from reality.
3. Predict the next state from an action.
4. Run imagination rollouts before acting.
5. Store experience from real outcomes.
6. Adapt the transition model from prediction errors.
7. Extend the loop toward richer embodied systems.

## What this repo demonstrates

This repo demonstrates the difference between a reactive agent and a world-model agent.

A reactive agent only acts.

A world-model agent predicts, acts, compares, remembers, and adapts.

That distinction is central to embodied AI, robotics, and physical AGI.

## Roadmap

Planned improvements:

- add ASCII grid visualization
- add prediction error metrics across episodes
- add experience memory module
- add imagination rollout planner
- add richer stochastic transitions
- add robotics-focused documentation
- add diagrams and demo traces
- add CI for tests

## Positioning

This repository is part of a broader research direction around memory-enabled robotics, embodied agents, and physical AGI.

The long-term goal is to study how agents can build useful internal models from real-world interaction rather than relying only on static prompts or fixed policies.
