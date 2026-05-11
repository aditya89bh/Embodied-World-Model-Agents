# Embodied World Model Agents

A minimal, interpretable repository for studying how embodied agents build internal models of the world, predict action outcomes, compare predictions with reality, store experience, reason under uncertainty, seek information, and adapt over time.

This repo is not a chatbot-agent playground. It focuses on the core loop required for physical intelligence:

```text
observe -> belief update -> frontier detection -> imagination -> prediction -> action -> observation -> prediction error -> memory -> model update
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
- reason over probabilistic outcomes
- separate internal belief from external reality
- seek useful information through exploration

This is the foundation of world-model based embodied intelligence.

## Current demo

The repo includes a stochastic grid-world demo where an agent observes locally, updates an internal belief map, detects exploration frontiers, imagines possible futures, and acts under uncertainty.

The environment contains:

- a hidden blocked cell
- a slippery stochastic cell
- probabilistic movement outcomes
- partial observability through local sensing
- unknown regions that motivate exploration

The agent learns transition distributions from experience and uses confidence, uncertainty, and information gain to score imagined rollouts.

Run:

```bash
pip install -r requirements.txt
python examples/run_world_model_demo.py
```

Run tests:

```bash
pytest
```

## Belief vs reality

The demo explicitly separates what actually exists from what the agent currently knows.

```text
BELIEF MAP
? ? ? ? ?
? A X ? ?
? . . ? ?
? ? ? ? ?
? ? ? ? G

REALITY MAP
. . . . .
. A X . .
. S . . .
. . . . .
. . . . G
```

This makes partial observability visible.

The agent does not receive the full map. It observes nearby cells, updates belief, and gradually constructs an internal model of the environment.

## Exploration and information gain

The agent now identifies exploration frontiers:

```text
known cells adjacent to unknown cells
```

It estimates information gain using the number of unknown neighboring cells and adds an exploration bonus to rollout scoring.

```text
score =
  distance_reward
+ confidence_bonus
+ movement_bonus
+ exploration_bonus
- uncertainty_penalty
```

The demo exposes this as an exploration motive:

```text
Exploration motive:
  selected action 'right' moves toward (3,2) with expected information_gain=4
```

This makes curiosity-driven planning visible.

## Stochastic world model demo

The demo shows candidate action rollouts like:

```text
Action: right | total_score=-2.31
  distance=-3.00 confidence=0.65 movement=0.20 exploration=0.75 uncertainty_penalty=0.54

  (1,2) --right--> expected (2,2)
    distribution:
      (2,2): 0.65
      (1,3): 0.20
      (1,2): 0.15
    confidence: 0.65
    uncertainty: 0.82
    information_gain: 3
```

This makes the internal prediction process inspectable.

## Core architecture

```text
Environment
    |
    | local observation
    v
BeliefMap
    |
    | frontier detection + information gain
    v
WorldModelAgent
    |
    | imagines futures using
    v
RolloutPlanner
    |
    | queries
    v
TabularWorldModel
    |
    | predicts probability distributions
    v
Action execution
    |
    | returns observation
    v
Prediction error + memory update
    |
    v
Improved future prediction
```

## Repo structure

```text
embodied_world_models/
├── __init__.py
├── state.py
├── environment.py
├── stochastic_gridworld.py
├── belief_map.py
├── world_model.py
├── rollout_planner.py
├── visualization.py
└── agent.py

examples/
└── run_world_model_demo.py

tests/
├── test_world_model_demo.py
├── test_stochastic_world_model.py
├── test_belief_map.py
└── test_exploration.py

docs/
├── architecture.md
├── stochastic_world_models.md
├── belief_vs_reality.md
└── exploration_and_information_gain.md

project_e4_imagination_rollouts/
project_e5_experience_memory/
project_e6_world_model_adaptation/
```

The `embodied_world_models/` package is the clean runnable layer.

The numbered `project_e*` folders preserve the earlier learning modules and experiments.

## Main concepts

| Concept | Meaning in this repo |
|---|---|
| Reality map | The actual environment |
| Belief map | What the agent currently thinks is true |
| Unknown cell | A region the agent has not observed yet |
| Local observation | Limited sensing around the agent |
| Frontier | Known region adjacent to unknown space |
| Information gain | Expected new knowledge from visiting a state |
| Exploration bonus | Reward for moving toward informative states |
| World model | Internal prediction system for action outcomes |
| Action | A movement command such as `right`, `left`, `up`, `down` |
| Observation | What the agent locally perceives after acting |
| Prediction error | Mismatch between predicted and actual outcome |
| Experience memory | Stored transitions from past action attempts |
| Adaptation | Updating the model so future predictions improve |
| Confidence | Dominant probability in a transition distribution |
| Uncertainty | Entropy of possible action outcomes |
| Rollout | Imagined future trajectory before acting |

## Robotics mapping

| Grid-world concept | Robotics equivalent |
|---|---|
| Agent position | Robot TCP or end-effector pose |
| Grid cell | Workspace region |
| Action | Motion primitive or skill |
| Hidden block | Fixture, collision zone, machine constraint, or unexpected obstacle |
| Slippery cell | Motion uncertainty, wheel slip, object slip, tolerance error |
| Belief map | Robot's internal workspace model |
| Local observation | Camera, force, proximity, or state feedback |
| Frontier | Boundary between known and unknown workspace |
| Information gain | Value of sensing or testing a region |
| Prediction | Expected result of a robot action |
| Actual outcome | Sensor/state feedback after execution |
| Prediction error | Difference between expected and observed result |
| Transition distribution | Learned reliability of a skill in context |
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
7. Estimate confidence and uncertainty.
8. Plan under probabilistic outcomes.
9. Maintain an internal belief map from local observations.
10. Seek information through frontier-based exploration.
11. Extend the loop toward richer embodied systems.

## What this repo demonstrates

This repo demonstrates the difference between a reactive agent and a world-model agent.

A reactive agent only acts.

A world-model agent observes, updates belief, detects frontiers, predicts, imagines, acts, compares, remembers, and adapts.

That distinction is central to embodied AI, robotics, and physical AGI.

## Roadmap

Planned improvements:

- add persistent memory serialization
- add memory decay and confidence decay
- add robotics-focused skill abstractions
- add dynamic environment drift
- add diagrams and demo traces
- add CI for tests

## Positioning

This repository is part of a broader research direction around memory-enabled robotics, embodied agents, and physical AGI.

The long-term goal is to study how agents can build useful internal models from real-world interaction rather than relying only on static prompts or fixed policies.
