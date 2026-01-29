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
