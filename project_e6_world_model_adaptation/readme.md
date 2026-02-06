# Project E6 – World Model Adaptation
_Updating Imagination Using Experience_

This project upgrades the agent from “I can plan” to “I can improve my planning.”

E4 introduced imagination (rollouts).
E5 introduced prediction error and experience memory.
E6 uses those experiences to **adapt the world model** so future predictions become
more accurate and rollouts become calibrated.

---

## Why This Project Exists

Without adaptation:
- the agent keeps using a flawed internal model
- imagination stays overconfident
- mistakes repeat, only with penalties

With adaptation:
- the agent’s transition predictions shift toward reality
- uncertainty increases in unfamiliar or error-prone regions
- planning becomes more reliable over time

This is where learning becomes structural, not cosmetic.

---

## Core Idea

We maintain an internal transition model:

(State, Action) → Distribution over Next State

When reality disagrees, we update:
- probability mass moves toward actual outcomes
- confidence drops when outcomes are inconsistent
- risk increases where prediction error is frequent

---

## What This Project Builds

### 1) Adaptive Transition Model
A lightweight model that supports:
- update(state, action, actual_next_state)
- predict_distribution(state, action)
- confidence(state, action)

### 2) Model Update Rules
Update strategies such as:
- count-based frequency updates (tabular)
- exponential moving average (EMA) updates
- error-weighted updates (bigger surprises update more)

### 3) Imagination Integration
Rollouts will use:
- the updated transition distribution
- uncertainty penalties during scoring
- sampled next-states for stochastic rollouts

### 4) Diagnostics
Simple dashboards/logs:
- top uncertain transitions
- biggest recurring surprises
- before/after prediction accuracy

---

## Project Structure

```text
project_e6_world_model_adaptation/
├── README.md
├── transition_model_tabular.py     # Adaptive (state,action)->next_state distribution
├── adaptation_rules.py            # Update rules (counts / EMA / error-weighted)
├── uncertainty.py                 # Confidence + uncertainty scoring
├── planner_rollout_stochastic.py  # Rollouts that sample from transition distribution
├── demo.py                        # Run: predict → act → update → improve
└── tests.py                       # Sanity checks
