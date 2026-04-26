# Teaching an LLM to Run a Fulfillment Desk

If you've ever watched an operations manager triage orders during a supplier outage, you'll know the job isn't really about following a rulebook. It's about reading a messy situation — three orders, two warehouses, half the inventory you expected — and making a call that keeps the most important customers happy without breaking promises you've already made.

That's the role we tried to give a language model.

## The Setup

CommerceOps-Env is a small e-commerce world the model can act inside. Every episode it sees a snapshot — the open orders, what's sitting in each warehouse, who the customer is, how long until the SLA clock runs out — and it picks one structured action. Assign a warehouse. Split a shipment across two. Delay an order. Reroute when something fails upstream. That's it.

The model never gets to "explain" itself out of a bad outcome. The environment grades it from state alone. If an order ends the episode in the wrong status, the score reflects that.

We give it three jobs:

**Warehouse assignment.** One order, three warehouses. Send it to the closest one that has stock and supports the requested shipping method. It's the kind of decision that looks trivial until you notice the closest warehouse doesn't carry the SKU.

**Multi-order triage.** Two orders showing up at the same time, one SKU, two warehouses with one unit each, and competing customer tiers. There's a clear right answer here — each order has a "near" warehouse — but the catch is that picking the wrong one consumes the only inventory available and leaves the other order stranded. A loyalty customer waiting on a delayed shipment is the worst possible outcome, so tier matters when it goes wrong.

**Cascade recovery.** A supplier failure has zeroed out stock at one warehouse. There's a pending order that was always going to ship from somewhere — now the model has to recognise the broken state and reroute. It's the situation that breaks rule engines: the "right" answer changes the moment the world changes.

## Why This Isn't a Chatbot

A lot of LLM demos are really just prompt engineering with a clever wrapper. We wanted this to feel like an actual operating environment. So:

- The action space is a fixed JSON schema. The model can't make up an action verb. It can't sneak natural language into a field. If it tries, the request is rejected and a small penalty applies.
- Inventory is protected. Only the environment can decrement stock — the model has no path to mutate state directly. There's no way to "fake" fulfillment.
- The grader looks at the world, not at the model's reasoning. It doesn't care how the order got assigned, only whether the order ended up in the right place.
- Repeat actions get penalised. Hammering the same order with the same verb to farm reward gets you nowhere.

These are small things, but they're what make the reward signal honest. You can't game your way to a high score by being verbose or persuasive.

## What the Baseline Could and Couldn't Do

When we ran a frozen Qwen2.5-1.5B against the environment without any training, the picture wasn't uniform.

It was already pretty competent at the multi-order triage task — averaging 0.81 across four eval seeds. Distance buckets are intuitive: if an order is "near" warehouse W2, the model picks W2. That alone gets you most of the way on a clean two-order scenario, and the baseline reflected that.

The single-order assignment task was a different story. Mean score 0.55, with an invalid-action rate above 57%. The model would sometimes pick a warehouse that didn't actually carry the SKU, or it would generate an action with a malformed field that bounced off the schema validator entirely. Format compliance was the bottleneck more than judgment.

And on the cascade recovery task, the baseline simply didn't know what to do. It hadn't seen the `reroute_order` verb in any meaningful way during pretraining, so it would emit invalid actions and time out the episode. Mean score 0.01. Invalid-action rate 100%. Across every seed.

That's the kind of split you actually want before training. There's a clear gap to close on two of the three tasks, and a steady performance on the third that you don't want to destroy in the process.

## What Training Actually Did

We used GRPO — group-relative policy optimisation — with LoRA adapters on top of the base model. The reward function is layered: a small bonus for outputting valid JSON, more for picking the right entity, more for using the right verb, and the largest share for the final state matching what a sensible plan would have produced. Penalties show up for repeating actions, going over the step budget, and causing collateral damage like assigning to a warehouse that's already empty.

We deliberately train on all three tasks together rather than one at a time. The reasoning is practical: if you only train on the headline task, the model forgets the others. Mixing them keeps the things the baseline already does well from regressing while the harder tasks catch up.

The shape of the training run matched the hypothesis. Reward climbed steadily, format compliance jumped from "sometimes" to "always", and the model picked up the new verbs quickly because the per-step gradient is dense — getting the verb right is worth real reward even before the warehouse choice settles. Eighty steps in, training loss settled around 0.012.

## What Changed After Training

The numbers came back the way we'd hoped:

- **Single-order assignment** went from 0.55 to 0.795 mean score. The 57% invalid-action rate dropped to zero. Format errors went away, and the model started picking warehouses that actually carry the SKU and support the shipping method.
- **Cascade recovery** — which the baseline completely failed at — moved from 0.01 to 0.50. The 100% invalid-action rate dropped to zero. The model learned that a supplier failure means reroute, and it learned to read the stock table to figure out where to reroute *to*.
- **Multi-order triage** held its ground at 0.81. The competence the baseline already had didn't get washed out by training pressure on the other two tasks.

That last point matters more than it might sound. A common failure mode in multi-task RL is that gains on the hard tasks come at the cost of the easy ones. We didn't see that here. The mix of fast-forwarded states and dense per-step rewards seems to keep the model from overfitting to any single task's reward shape.

## Why Any of This Matters

The honest answer is that operations work doesn't reduce neatly to rules. The decisions an experienced fulfillment manager makes — which order to delay when stock is short, when to absorb a shipping cost to keep a tier-1 customer happy, when to escalate versus when to wait — are judgment calls. They're learned from many small situations, and they don't compress into a flowchart.

If language models are going to be useful inside operational software (and not just sitting next to it answering questions), they need to act inside environments where the consequences are real and the rewards reflect business outcomes. That's the experiment this is. Not "can the model talk about fulfillment", but "can the model do fulfillment, badly at first, and get better".

The early answer is yes — at least at this scale, in this kind of environment, with this style of training. The harder question, which we're not pretending to answer here, is what happens when you scale the world up and the exceptions get weirder.

## Try It

The whole project is open under MIT. If you want to poke at it:

- **Trained model**: [huggingface.co/Gloomytarsier3/ecommerce-ops-grpo](https://huggingface.co/Gloomytarsier3/ecommerce-ops-grpo)
- **Training results & logs**: [huggingface.co/datasets/Gloomytarsier3/ecommerce-ops-results](https://huggingface.co/datasets/Gloomytarsier3/ecommerce-ops-results)
- **Live OpenEnv space**: [huggingface.co/spaces/Gloomytarsier3/e-com-r2](https://huggingface.co/spaces/Gloomytarsier3/e-com-r2)
- **Training notebook (Colab)**: [colab.research.google.com/drive/1zsdtNfhN8_pstougqh66K8bMmSRaeZrZ](https://colab.research.google.com/drive/1zsdtNfhN8_pstougqh66K8bMmSRaeZrZ?usp=sharing)
- **GitHub repo**: [github.com/Dhruv-80/ecommerce-ops-env-starter](https://github.com/Dhruv-80/ecommerce-ops-env-starter)

If you want to look at the two files that matter most:

- The **environment** lives in [`environment.py`](https://github.com/Dhruv-80/ecommerce-ops-env-starter/blob/main/environment.py). That's the `reset / step / final_score` loop — every action the model takes runs through here, gets schema-validated, mutates state under the anti-hacking guards, and produces the next observation and reward.
- The **training script** lives in [`train/hf_train.py`](https://github.com/Dhruv-80/ecommerce-ops-env-starter/blob/main/train/hf_train.py) (and the same code lives in the Colab notebook). It wires the environment up to GRPO via Hugging Face TRL, runs the baseline eval, trains the LoRA adapters, runs the post-training eval, and pushes the trained model and `results.json` + `reward_curves.png` to the two HF repos linked above.

The Colab notebook captures the full run end-to-end — baseline eval, training loop, post-training eval, and the before/after comparison — and you can re-run it on a single GPU.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026.*
