# Project E1 – State Representation  
_Perception → Belief_

This project implements the foundational layer of embodied intelligence: **how an agent represents the world it believes it is in**.

Before planning, memory, or imagination, an agent must answer a simple question:
“What does the world look like right now?”

This project builds a **structured world-state representation** that converts raw observations into an explicit, uncertainty-aware belief about reality.

---

## Why State Representation Matters

Most AI agents:
- operate directly on pixels or text
- lack a stable notion of “world”
- confuse observation with belief
- fail under partial or missing information

Real-world intelligence requires a **state abstraction**:
- what exists
- where it exists
- what is known vs unknown
- what constraints apply

This project treats state as a **first-class cognitive object**, not a hidden tensor.

---

## Core Idea

Perception does not produce truth.  
Perception produces **belief**.

This project separates:
- observation (what is sensed)
- state (what is believed)
- uncertainty (what is unknown)

---

## What This Project Builds

A **WorldState** abstraction that includes:
- agent state (position, energy, orientation)
- environment layout
- entities and their properties
- constraints and impassable regions
- uncertainty and partial observability
- metadata such as timestep and source

This state becomes the shared interface for:
- planning agents
- memory systems
- world-model dynamics
- imagination rollouts

---
## Scope of Project E1

Included:
- grid-based or simple 2D environments
- deterministic perception
- partial observability
- explicit belief updates
- no learning or optimization

Explicitly excluded:
- reinforcement learning
- neural world-models
- robotics hardware
- end-to-end policies

This is a **representation-first** project.

---

## Repository Structure

```text
project_e1_state_representation/
├── README.md
├── state_schema.py      # WorldState definitions
├── perception.py        # Observation extraction
├── encoder.py           # Observation → WorldState
├── examples.py          # Usage and demos
└── tests.py             # Sanity checks
