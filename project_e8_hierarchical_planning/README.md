# Project E8 – Hierarchical Planning (Options & Skills)
_From Actions to Intentional Behavior_

This project introduces **temporal abstraction** into the agent.

Instead of planning over primitive actions (up, down, left…),
the agent learns and uses **options / skills**:
multi-step behaviors with intent, duration, and termination.

E8 completes the agent stack by enabling long-horizon reasoning
without exponential planning cost.

---

## Why This Project Exists

Up to E7, the agent can:
- model the world
- adapt from experience
- reason about uncertainty
- explore intelligently

But it still plans at the **wrong level of abstraction**.

Problems without hierarchy:
- long-horizon planning is expensive
- plans are brittle
- the agent cannot reuse experience efficiently
- behavior looks reactive, not intentional

E8 fixes this by introducing **behavioral primitives**.

---

## Core Idea

An **Option (Skill)** is defined as:

- **Initiation condition**: when the option is applicable
- **Policy**: what actions to execute while the option is active
- **Termination condition**: when the option ends
- **Expected outcome**: predicted effect on state, cost, risk

The planner chooses **options**, not actions.
Actions are executed inside options.

---

## What This Project Builds

### 1) Option / Skill Abstraction
A formal representation of skills such as:
- “navigate to region”
- “avoid obstacle corridor”
- “approach goal safely”

Each option encapsulates:
- multi-step behavior
- internal control loop
- learned outcome statistics

---

### 2) Option-Level World Model
Instead of:
(State, Action) → Next State

We now model:
(State, Option) → Outcome Distribution

This drastically reduces planning depth.

---

### 3) Hierarchical Planner
A two-level planner:

**High level**
- plans over options
- reasons about long-term outcomes and uncertainty

**Low level**
- executes the selected option
- handles primitive actions
- monitors termination

---

### 4) Skill Learning Hooks
Options can be:
- hand-defined (v1)
- learned from repeated trajectories
- promoted from frequent action sequences

This makes behavior compositional and reusable.

---

## Project Structure

```text
project_e8_hierarchical_planning/
├── README.md
├── option.py                   # Option / skill definition
├── option_library.py           # Collection of available options
├── option_policy.py            # How an option executes actions
├── option_model.py             # (state, option) → outcome distribution
├── option_planner.py           # High-level planner over options
├── controller.py               # Bridges option execution to actions
├── demo.py                     # Long-horizon task solved via options
└── tests.py                    # Sanity tests for options + planner
