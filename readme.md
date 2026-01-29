# 🌍 Embodied & World-Model Agents  
_From Perception to Imagination_

This repository is a **systematic exploration of world-models and embodiment for AI agents**.

The goal is not robotics demos or game-playing agents.

The goal is to understand and implement **how an agent models reality**, predicts consequences, imagines futures, and grounds its decisions in a causal world.

This repo builds on prior work in:
- Memory Agents (continuity, learning, identity)
- Reasoning & Planning Agents (goals, strategies, failure awareness)

Here, intelligence is no longer abstract.

It is **situated**.

---

## Why World Models?

Most AI agents today:
- reason in text
- plan in symbols
- hallucinate actions
- ignore physics, cost, delay, and failure

Real intelligence requires a **model of the world**:
- what exists
- how it changes
- what actions are possible
- what actions are costly or irreversible

A world-model allows an agent to:
- predict before acting
- imagine futures
- learn from surprise
- ground planning in reality

This repository treats **world-modeling as the bridge between cognition and embodiment**.

---

## Core Idea

World Models =  
**Perception → State → Dynamics → Imagination → Action → Feedback → Memory**

Every project in this repository implements one piece of that loop.

Nothing is skipped. Nothing is assumed.

---

## Project Structure

### Project E1 – State Representation (World → Latent)
**Perception as Belief**

Build a state encoder that converts raw observations into a compact, structured representation of the world.

Focus:
- what the agent believes exists
- what is known vs unknown
- uncertainty and partial observability

Output is not pixels or text, but a **world state** the agent reasons over.

---
