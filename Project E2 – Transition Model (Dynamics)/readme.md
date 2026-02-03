# Project E2 – Transition Model (Dynamics)
_State + Action → Next State_

This project implements **world dynamics**: how an agent’s believed world changes when it takes an action.

Project E1 answered:
“What does the agent believe the world looks like?”

Project E2 answers:
“If I do this, what will happen next?”

This is the foundation of prediction, imagination, and planning.

---

## Why Transition Models Matter

Most agents today:
- choose actions without predicting consequences
- assume the world is static
- cannot reason about failure, delay, or cost
- hallucinate outcomes

Real intelligence requires a **model of change**.

A transition model allows an agent to:
- predict outcomes before acting
- learn dynamics from experience
- estimate uncertainty when knowledge is missing
- support imagination rollouts in planning

---

## Core Idea

World Dynamics =  
**Current State + Action → Next State + Outcome**

This project formalizes that transformation.

---

## What This Project Builds

Two complementary transition models:

### 1. Rule-Based Transition Model
A deterministic, environment-grounded transition function that:
- applies physical constraints (walls, boundaries)
- updates agent position
- reveals new observations
- serves as a trusted baseline

This acts as the agent’s initial “physics engine”.

---

### 2. Learned Tabular Transition Model
An experience-based world model that:
- learns from `(state, action) → next_state`
- stores transition frequencies
- predicts the most likely next state
- exposes uncertainty when data is sparse

This is the agent’s first learned world-model.

---

