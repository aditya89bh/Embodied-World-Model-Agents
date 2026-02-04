# Project E3 – Action Space & Constraints
_What the Body Can and Cannot Do_

This project defines the agent’s **action system**: the set of actions available to the agent, the constraints that restrict them, and the costs and risks attached to each action.

In E2, actions were simple strings and dynamics were applied in the environment.
In E3, actions become a first-class concept with structure, semantics, and safety.

The output of this project is an **Action Model** that downstream modules can use for:
- planning under constraints
- cost-aware decision making
- failure-aware behavior
- safety checks before execution

---

## Why This Project Exists

Most agents fail in real environments because they:
- assume actions always succeed
- ignore time and energy costs
- plan impossible moves
- do not model constraints explicitly

Embodied intelligence requires an action layer that knows:
- what is possible
- what is invalid
- what is expensive
- what is risky

This project turns “actions” into **grounded, constrained operators**.

---

## Core Idea

Action =  
**Intent + Preconditions + Effects + Cost + Risk + Latency**

Instead of “move_right”, an action has meaning:
- it may be invalid due to obstacles
- it may fail with some probability
- it may take time
- it may consume energy
- it may have side effects

---

## What This Project Builds

### 1) Action Definitions
A structured action schema that includes:
- action name
- parameters (optional)
- preconditions
- effects
- cost model (time/energy)
- risk model (failure probability)

### 2) Constraint System
A constraint layer that can determine:
- allowed actions given a state
- invalid actions and reasons
- soft constraints (penalties)
- hard constraints (forbidden)

### 3) Action Evaluation
A scoring interface that helps planners compare actions:
- expected utility
- cost tradeoffs
- risk-aware ranking

This becomes the bridge between E2 dynamics and E4 imagination rollouts.

---

## Project Structure

```text
project_e3_action_space_constraints/
├── README.md
├── action_schema.py        # Action data structures
├── constraints.py          # Hard + soft constraints
├── cost_models.py          # Time/energy cost functions
├── risk_models.py          # Failure and uncertainty models
├── action_selector.py      # Choose actions under constraints
├── demo.py                 # Run examples in gridworld
└── tests.py                # Sanity tests
