# Project E5 – Experience Memory & Prediction Error
_Learning From When Reality Disagrees_

This project introduces **experience memory**: the ability of an agent to
compare imagined outcomes with real outcomes, detect mismatch, and update
its internal world model.

E4 allowed the agent to imagine futures.
E5 allows the agent to **learn when those imaginations were wrong**.

This is where adaptation begins.

---

## Why This Project Exists

Planning alone is not intelligence.

An intelligent agent must:
- notice when reality deviates from expectation
- remember those deviations
- adjust future predictions and behavior

Without this, agents hallucinate competence.

---

## Core Idea

For every step:

Imagined Outcome (E4)  
vs  
Actual Outcome (Reality)

→ **Prediction Error**

Prediction error is stored as experience:
- what state the agent was in
- what action it took
- what it expected
- what actually happened
- how wrong it was

This experience is then used to:
- update transition confidence
- increase perceived risk
- bias future rollouts away from failures

---

## What This Project Builds

### 1) Experience Record
A structured memory entry:
- state snapshot
- action
- predicted next state
- actual next state
- error magnitude

### 2) Prediction Error Detector
Logic to compare belief-space rollouts with real execution.

### 3) Experience Memory Store
A growing memory that:
- indexes failures
- tracks surprises
- stores rare events

### 4) Model Update Hooks
Interfaces for:
- adjusting rollout scores
- penalizing overconfident actions
- flagging unreliable transitions

(No learning yet — just signal routing.)

---

## Project Structure

```text
project_e5_experience_memory/
├── README.md
├── experience.py            # Experience data structure
├── error_metrics.py         # How wrong was the prediction?
├── experience_store.py      # Memory of mismatches
├── update_hooks.py          # How experience influences planning
├── demo.py                  # Imagination vs reality loop
└── tests.py                 # Sanity checks
