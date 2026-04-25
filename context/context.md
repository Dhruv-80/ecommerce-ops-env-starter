# CommerceOps-Env Grand Finale Handoff

This document is the single source of truth for continuing the hackathon project in Cursor. It contains the full product context, build direction, training plan, stack, constraints, deliverables, and hackathon rules relevant to implementation.

## Project Summary

Build an OpenEnv-compliant reinforcement learning environment where an LLM acts as a fulfillment operations manager and learns to make judgment-heavy decisions under inventory scarcity, SLA pressure, competing customer priorities, and routing tradeoffs.[cite:17][cite:72]

The project should not be a chatbot, prompt wrapper, or plain classifier. It must be a stateful environment with structured observations, constrained actions, rewards, and episode transitions so the model can be trained with verifiable outcomes and later shown to improve measurably.[cite:72][cite:70]

## Core Story

The core claim of the project is:

> Human fulfillment managers make judgment calls that cannot be covered cleanly with a giant if/else rules engine. This environment captures that judgment, and GRPO training should help an LLM improve at making those tradeoff decisions.[cite:72]

This is the main story to preserve in code, demo, and pitch.

## Why This Version

Earlier thinking was centered around refunds and stock workflows from Round 1. The current version improves that by shifting the spotlight toward fulfillment judgment, which is more interesting and more defensible as a real RL environment.[cite:17]

The new version should still reuse the same OpenEnv philosophy from the prior e-commerce environment work: deterministic world state, typed contracts, stable reset/step/state behavior, simple verifiers, and a reward design that is hard to game.[cite:72]

## What Is Being Improved From Round 1

Round 1 already had the basic OpenEnv-style e-commerce environment idea with task scaffolding and grader logic.[cite:17]

Round 2 must improve on that in the following ways:

- Add actual RL training evidence instead of stopping at a static environment.[cite:72]
- Show measurable before/after performance improvement, not just environment correctness.[cite:17][cite:72]
- Use stronger reward engineering with multiple verifiable components instead of only one pass/fail style check.[cite:72]
- Add anti-reward-hacking safeguards and inspect rollouts manually during training.[cite:72]
- Deploy the environment early on Hugging Face Spaces so it can be used consistently by the team and shown to judges.[cite:72][cite:70]
- Keep the environment small enough that training produces non-zero reward quickly.[cite:72]

## Theme Fit

Primary fit: **World Modeling — Professional Tasks**. The project is a professional workflow environment where an agent must act inside a partially constrained business system and update state over multiple steps.[cite:17]

This is also compatible with the enterprise workflow framing from the hackathon themes, especially because the environment simulates real workflow tradeoffs rather than only text generation.[cite:17]

## Hackathon Rules And Requirements

The following are hard requirements and must be satisfied:

- Use the **latest OpenEnv release**.[cite:17]
- Show a **minimal training script** for the environment using **Unsloth** or **Hugging Face TRL** in **Colab**.[cite:17]
- Publish either a **mini-blog on Hugging Face** or a **mini-video on YouTube** about the submission, under 2 minutes if video.[cite:17]
- The environment should be deployable and usable through the OpenEnv pattern, which is designed around FastAPI apps and Hugging Face Spaces.[cite:72]

Judging criteria to optimize for:

| Criterion | Weight | What the project must show |
|---|---:|---|
| Environment Innovation | 40% | A stateful, realistic environment that tests meaningful behavior rather than static prompting.[cite:17] |
| Storytelling | 30% | A simple, convincing story: human-like fulfillment judgment under pressure.[cite:17] |
| Showing Improvement in Rewards | 20% | Baseline vs trained behavior, reward curves, and obvious behavior improvement.[cite:17][cite:72] |
| Reward and Training Setup | 10% | Coherent verifier design, training script, and meaningful optimization loop.[cite:17][cite:72] |

## The Final Locked Idea

The final environment is a **CommerceOps fulfillment judgment environment**.

An agent receives structured observations about orders, warehouses, stock, shipping distance, SLAs, and customer importance. It must choose actions that resolve the business situation while optimizing service quality, cost, and policy compliance.[cite:72]

The environment should feel like an operations dashboard world model, not a toy classification task.

## The Three Tasks

### Task 1 — Warehouse Assignment Bootstrap

One incoming order must be assigned to the best warehouse based on stock, distance, shipping method availability, and SLA urgency.[cite:72]

This task exists to bootstrap action format, environment interaction, and non-zero reward. It should be easy enough that the model occasionally succeeds early in training.[cite:72]

This is not the flagship task. It is the curriculum starter.[cite:72]

### Task 2 — Multi-Order Fulfillment Triage (Headline Task)

Several orders compete for limited stock across multiple warehouses. The agent must decide which orders get fulfilled, delayed, split, or deprioritized while balancing customer tier, SLA breach risk, distance, and operational cost.[cite:72]

This is the core interesting task. It represents the kind of judgment a human operations manager makes that is difficult to encode with simple rules.[cite:72]

This should be the primary task shown in the final demo and pitch.

### Task 3 — Cascade Recovery (Stretch)

A shipment or supplier failure occurs after earlier allocation decisions have already changed the system state. The agent must recover by rerouting, compensating, delaying, or escalating without causing downstream failures.[cite:72]

This task is long-horizon and should be considered stretch only. It should not block T1 and T2 from being finished.[cite:72]

## Environment Design Principles

The environment must satisfy these principles:

- It must be **stateful**: actions change inventory, order status, step count, and available options.[cite:70][cite:72]
- It must be **verifiable**: success should be checked from environment state or explicit business rules, not from model explanations.[cite:72]
- It must be **trainable**: early tasks must be easy enough for the model to achieve non-zero reward.[cite:72]
- It must be **hard to game**: reward hacking opportunities should be minimized with explicit checks and protected state.[cite:72]
- It must be **small enough to finish**: no giant maps, no real routing engines, no overbuilt UI.[cite:72]

## Recommended Stack

| Layer | Choice | Notes |
|---|---|---|
| Environment framework | OpenEnv latest | Mandatory for hackathon compliance.[cite:17][cite:72] |
| API layer | FastAPI | OpenEnv environments are described as FastAPI apps that can run locally or on Spaces.[cite:72] |
| Deployment | Hugging Face Spaces | Recommended early deployment target.[cite:72][cite:70] |
| Training library | TRL | Supports environment-based RL and GRPO workflows.[cite:72][cite:70] |
| RL algorithm | GRPO | Recommended in the participant materials as a practical, memory-efficient post-training method.[cite:72] |
| Efficiency layer | Unsloth | Recommended for efficient RL fine-tuning and faster experimentation.[cite:72] |
| Base model | Qwen2.5-3B Instruct | Recommended as the default starter choice from the participant materials and suitable for fast iteration.[cite:72] |
| Fallback model | Gemma 3 1B | Useful only if compute becomes a bottleneck.[cite:72] |
| Language | Python | Required by stack and tutorials.[cite:72] |

## Model Choice

Default model: **Qwen2.5-3B Instruct**.[cite:72]

Why this model:

- Small enough to iterate quickly in a hackathon setting.[cite:72]
- Explicitly recommended by the participant FAQ among GRPO starter recipes.[cite:72]
- Strong enough to occasionally solve structured workflow tasks, which matters because RL needs some successful rollouts to improve.[cite:72]

Fallback only if compute is tight: **Gemma 3 1B**.[cite:72]

Do not switch models repeatedly unless blocked by infrastructure.

## Environment Interface

The environment should expose the standard OpenEnv pattern:

- `reset()` to initialize a fresh episode.[cite:72]
- `step(action)` to apply an action, update state, and return reward and terminal status.[cite:72]
- `state()` or equivalent internal state tracking to inspect the current world model.[cite:72]

The API should be packaged as an OpenEnv-compatible FastAPI application and deployable on Hugging Face Spaces.[cite:72][cite:70]

## Observation Design

Observations should be structured, typed, and explicit. They must include only the information the agent should be allowed to reason over.

Recommended fields:

- task type
- current step
- max steps
- list of orders
- per-order quantity requested
- per-order customer tier
- per-order SLA remaining
- per-order destination or distance bucket
- warehouse stock levels
- warehouse-to-order cost or distance estimates
- allowed actions
- any policy flags relevant to the current episode

Avoid freeform hidden logic. The observation should make the decision hard because of tradeoffs, not because of hidden information.

## Action Design

Actions should be constrained and schema-validatable. Examples:

- `assign_warehouse`
- `split_shipment`
- `delay_order`
- `prioritize_order`
- `reroute_order`
- `escalate_supplier`
- `refund_or_compensate`

Each action should include only the minimum fields needed to change the state, such as order IDs, warehouse IDs, quantity allocations, and reason codes.

Do not allow arbitrary text actions to control the system. Actions should be structured so schema validation can be part of reward and safety.[cite:72]

## Reward Design

Reward design should stay simple and layered.[cite:72]

Recommended components:

- **Task success reward**: binary outcome-based reward when the environment reaches a correct final state.[cite:72]
- **Schema compliance reward**: small positive reward for valid action format.[cite:72]
- **Invalid action penalty**: negative reward for invalid, impossible, or policy-violating actions.[cite:72]
- **Efficiency penalty**: small step penalty for unnecessary actions or excessive path length.[cite:72]
- **Optional business penalty**: split-shipment or SLA-breach penalty only if it is cleanly verifiable and does not overcomplicate training.[cite:72]

Reward engineering rules:

- Start simple.[cite:72]
- Reward outcomes first, then add minimal shaping only when necessary.[cite:72]
- Avoid many competing reward terms.[cite:72]
- Never optimize a reward you have not tried to break yourself first.[cite:72]

## Anti-Reward-Hacking Rules

These must be explicitly implemented or checked:

- Action whitelist only.[cite:72]
- Strict schema validation before applying actions.[cite:72]
- Protected environment state: model cannot mutate inventory or metadata directly.[cite:72]
- Timeout and max-step enforcement.[cite:72]
- Manual inspection of sampled generations during training.[cite:72]
- Holdout sanity checks to ensure rising reward matches better behavior.[cite:72]

## Training Strategy

The participant guidance strongly recommends the following order:[cite:72]

1. Build the environment first.[cite:72]
2. Build the verifier before the trainer.[cite:72]
3. Run scripted baselines.[cite:72]
4. Run a frozen instruct model.[cite:72]
5. Run a tiny GRPO experiment.[cite:72]
6. Inspect outputs for reward hacking.[cite:72]
7. Only then scale or add harder tasks.[cite:72]

Important training principles:

- Do not rely on RL from scratch; start from a capable instruct model.[cite:72]
- Use light SFT or format priming only if needed for valid action syntax.[cite:72]
- Monitor more than average reward: look at pass rate, invalid actions, timeout rate, and actual trajectories.[cite:72]
- Training should demonstrate measurable improvement, not huge scale.[cite:17][cite:72]

## Minimal Training Goal

The minimum success bar is not “state-of-the-art training.”

The minimum success bar is:

- A working OpenEnv environment.[cite:17][cite:72]
- A verifiable reward function.[cite:72]
- A baseline model that fails often enough to make improvement visible.[cite:72]
- A small GRPO run that produces some measurable improvement in reward or task success.[cite:17][cite:72]

## Deployment Requirements

The environment should be deployed early on Hugging Face Spaces.[cite:72][cite:70]

Why this matters:

- Shared source of truth for the team.[cite:72]
- Easier testing across local and remote workflows.[cite:72]
- Easier demoing to judges.[cite:72]

Deployment rules:

- Use the OpenEnv/FastAPI pattern.[cite:72]
- Make sure the environment responds cleanly to reset/step/state style interactions.[cite:72]
- Catch packaging issues early before training begins.[cite:72]

## Suggested File Structure

```text
commerceops-env/
├── models.py
├── environment.py
├── tasks.py
├── verifier.py
├── reward.py
├── server/
│   └── app.py
├── train/
│   └── grpo_train.ipynb
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

### File Responsibilities

- `models.py`: freeze all typed action, observation, and state contracts early.
- `environment.py`: implement reset/step/state logic.
- `tasks.py`: generate episodes for T1, T2, and optional T3.
- `verifier.py`: compute correctness and rule-based evaluation.
- `reward.py`: combine verifier outputs into reward components.
- `server/app.py`: expose the environment through FastAPI/OpenEnv.
- `train/grpo_train.ipynb`: minimal Colab-compatible GRPO training script using TRL + Unsloth.

## Team Split

Recommended split for two people:

### Dhruv

- Freeze contracts in `models.py`
- Implement `environment.py`
- Build `tasks.py`
- Handle `server/app.py` and deployment
- Own demo story and end-to-end integration

### Tendulkar

- Implement `verifier.py`
- Implement `reward.py`
- Add anti-hacking checks and timeout logic
- Build `train/grpo_train.ipynb`
- Log metrics and produce reward curves

Rule: after `models.py` is frozen, do not casually rename keys or fields. Stable contracts are what allow parallel work.[cite:17]

## Mandatory Deliverables

The final submission must include:

- OpenEnv environment using the latest release.[cite:17]
- Minimal training script using Unsloth or HF TRL in Colab.[cite:17]
- Hugging Face mini-blog or YouTube mini-video under 2 minutes.[cite:17]
- A live or reproducible demo showing before/after behavior and measurable improvement.[cite:17][cite:72]

## Demo Story

The best demo structure is:

1. Show the baseline model failing or making poor tradeoffs on a fulfillment scenario.[cite:72]
2. Show verifier output and reward breakdown.[cite:72]
3. Show the trained model making a better decision on a similar task.[cite:17][cite:72]
4. Show a reward curve or simple table with measurable improvement.[cite:17][cite:72]

The story should emphasize that the agent learned fulfillment judgment under pressure, not just output formatting.

## Build Order For Cursor

Cursor should continue in this exact order:

1. Create `models.py` first and freeze all contracts.
2. Implement Task 1 episode generation.
3. Implement Task 1 verifier and reward.
4. Implement Task 1 environment reset/step loop.
5. Add Task 2 episode generation and verifier.
6. Add FastAPI/OpenEnv server wrapper.
7. Test locally.
8. Deploy early to Hugging Face Spaces.
9. Add training notebook.
10. Run baseline.
11. Run tiny GRPO training.
12. Capture metrics and demo artifacts.

## Important Do-Not-Do List

- Do not turn this into a chatbot UI project.[cite:72]
- Do not add a full map UI, truck routing engine, or courier simulator.[cite:72]
- Do not overcomplicate reward shaping.[cite:72]
- Do not use only an LLM judge without hard checks.[cite:72]
- Do not wait until the end to deploy.[cite:72]
- Do not keep redesigning the idea once coding starts.
- Do not prioritize Task 3 over shipping T1 and T2.

## Definition Of Success

The project is successful if by the end of the hackathon it has all of the following:

- A real, stateful OpenEnv environment.[cite:72][cite:70]
- At least T1 and T2 working.[cite:72]
- Verifiable rewards with anti-hacking checks.[cite:72]
- A minimal GRPO run that shows measurable improvement.[cite:17][cite:72]
- A demo and pitch that clearly explain the human-judgment problem being solved.[cite:17]

## One-Sentence Pitch

**CommerceOps-Env trains an LLM to make human-like fulfillment decisions under inventory scarcity, SLA pressure, and competing customer priorities inside a verifiable OpenEnv business workflow environment.**[cite:17][cite:72]

## Cursor Instruction

Continue from this plan without redefining the project. Start by implementing `models.py` for the fulfillment-judgment environment, centered around Task 2 but with Task 1 kept simple enough for early training success.
