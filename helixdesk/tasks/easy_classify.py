import numpy as np

def grade(env, agent, seed=42) -> float:
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    correct = 0
    total = 0
    for _ in range(20):
        # True keyword flag is at index 1
        is_keyword = obs[1] > 0.5
        action = agent.act(obs)
        if is_keyword:
            total += 1
            if action[0] == 1 and action[1] == 0:  # classify=complaint, priority=critical
                correct += 1
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    if total == 0: return 1.0
    return float(correct) / float(total)
