# Project E3 – Action Space & Constraints
_What the Body Can and Cannot Do_

This project defines the agent’s **action system**: the set of actions available to the agent, the constraints that restrict them, and the costs and risks attached to each action.

In E2, actions were simple strings and dynamics were applied in the environment.
In E3, actions become a first-class concept with structure, semantics, and safety.

The output of this project is an **Action Model** that downstream modules can use for:
- planning under constraints
- cost-aware decision making
- failure-aware behavior
- safety checks before execution

---

## Why This Project Exists

Most agents fail in real environments because they:
- assume actions always succeed
- ignore time and energy costs
- plan impossible moves
- do not model constraints explicitly

Embodied intelligence requires an action layer that knows:
- what is possible
- what is invalid
- what is expensive
- what is risky

This project turns “actions” into **grounded, constrained operators**.

---
