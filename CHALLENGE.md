# What Makes HelixDesk Hard for LLM Agents

## Delayed Rewards
Resolution events fire steps after assignment — 
the agent that assigns a ticket doesn't see the 
resolution reward until later steps.

## Partial Observability  
The agent sees normalized floats, not raw email text.
It must infer category and urgency from dim 0-9 alone.

## Competing Objectives
Maximizing CSAT conflicts with fast assignment.
Catching trends requires alert_gm which costs secondary action slots.

## The Expert Task
All 5 criteria must be met simultaneously.
Optimizing one metric often degrades another.
A frontier LLM scoring above 0.5 on expert_full 
would be a strong result.

## Benchmark Targets
| Agent | easy | medium | hard | expert |
|-------|------|--------|------|--------|
| random | 0.000 | 0.448 | 0.415 | 0.000 |
| rule | 1.000 | 0.882 | 0.652 | 0.935 |
| target LLM | ~0.90 | ~0.75 | ~0.50 | ~0.25 |
