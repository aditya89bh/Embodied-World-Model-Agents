# Project E7 – Uncertainty & Calibration
_When Confidence Actually Means Something_

This project separates **knowing**, **not knowing**, and **being wrong**.

E6 taught the agent to adapt its world model.
E7 teaches the agent to understand **how much it should trust that model**.

This is the difference between:
- learning
- and knowing when learning is incomplete

---

## Why This Project Exists

Most agents:
- output probabilities
- but don’t understand confidence
- and cannot distinguish noise from ignorance

As a result:
- they overcommit to bad plans
- explore blindly
- or become falsely conservative

E7 fixes this by making uncertainty explicit, structured, and actionable.

---

## Core Idea

We decompose uncertainty into two types:

### 1) Epistemic Uncertainty (Knowledge Uncertainty)
- Comes from lack of data
- Decreases with experience
- Signals “I don’t know yet”

Example:
> This (state, action) has only been tried once.

### 2) Aleatoric Uncertainty (World Noise)
- Comes from inherent randomness
- Does NOT disappear with more data
- Signals “the world is unstable here”

Example:
> The same action leads to different outcomes even after many trials.

E7 teaches the agent to tell these apart.

---

## What This Project Builds

### 1) Uncertainty Decomposition
For each (state, action):
- outcome variance
- sample count
- entropy of transition distribution

Mapped into:
- epistemic score
- aleatoric score

---

### 2) Calibration Metrics
Measures whether:
- predicted confidence matches empirical accuracy
- high-confidence predictions fail rarely
- low-confidence predictions fail often

This prevents overconfidence.

---

### 3) Uncertainty-Aware Planning Hooks
Planning can now:
- avoid high epistemic regions when risk-averse
- seek high epistemic regions when exploring
- tolerate aleatoric uncertainty differently than ignorance

---

### 4) Exploration Policies
Policies such as:
- curiosity-driven exploration (reduce epistemic uncertainty)
- safety-aware avoidance (limit aleatoric risk)
- mixed strategies based on task context

---

## Project Structure

```text
project_e7_uncertainty_calibration/
├── README.md
├── uncertainty_metrics.py        # entropy, variance, confidence intervals
├── decomposition.py              # epistemic vs aleatoric split
├── calibration.py                # confidence vs accuracy tracking
├── exploration_policy.py         # how uncertainty shapes action choice
├── planner_uncertainty_aware.py  # planning with calibrated uncertainty
├── demo.py                       # visualize confidence over time
└── tests.py                      # sanity + calibration tests
