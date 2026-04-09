import numpy as np

def grade(env, agent, seed=42) -> float:
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    score_penalty = 0.0
    steps = 0
    done = False
    while not done:
        action = agent.act(obs)
        obs, r, term, trunc, info = env.step(action)
        if info.get("overdue_count", 0) > 0:
            score_penalty += 0.01

        steps += 1
        done = term or trunc
    
    return max(0.0, 1.0 - score_penalty)
