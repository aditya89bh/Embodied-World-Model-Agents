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
