# Project E4 – Imagination & Rollouts
_Thinking Before Acting_

This project implements the agent’s “imagination loop”: the ability to simulate possible futures internally before taking real actions.

E2 gave us dynamics:
(State, Action) → Next State

E3 gave us grounded actions:
preconditions + cost + risk + effects

E4 combines them into rollouts:
simulate multiple action sequences, score outcomes, and choose the best plan.

---

## Why This Project Exists

Most agents act first and explain later.

Embodied intelligence requires:
- predicting consequences
- comparing alternatives
- avoiding risky or costly paths
- selecting actions that optimize long-term outcomes

Imagination is planning via internal simulation, not prompt chaining.

---

## Core Idea

Rollout =  
start_state → (a1, a2, ... ak) → predicted_future

Then score:
- expected reward
- total cost (time/energy)
- risk accumulation
- uncertainty penalties

Pick the best first action (receding-horizon planning).

---

## What This Project Builds

### 1) Rollout Simulator
Given a belief state and a transition model, simulate k-step futures.

### 2) Candidate Generation
Generate candidate action sequences using:
- random sampling
- breadth-limited search
- constraint-filtered actions

### 3) Scoring Function
Score a rollout using:
- goal achievement
- cost models (E3)
- risk models (E3)
- optional exploration bonuses

### 4) Action Selection via Imagination
Choose the next action by comparing simulated futures.

---

## Project Structure

```text
project_e4_imagination_rollouts/
├── README.md
├── env_gridworld.py           # Ground-truth world (for evaluation)
├── state_adapter.py           # Belief state (minimal)
├── transition_rule_based.py   # Deterministic dynamics (baseline model)
├── action_schema.py           # Action abstraction
├── constraints.py             # Hard constraints
├── cost_models.py             # Cost functions
├── risk_models.py             # Risk functions
├── action_library.py          # Default action set
├── rollout.py                 # Rollout simulator + scoring
├── planner.py                 # Choose action via imagined rollouts
├── render.py                  # Truth vs belief render
├── demo.py                    # Run imagination planner end-to-end
└── tests.py                   # Sanity tests
