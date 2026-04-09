"""Medium task: SLA compliance rate.

Run one full episode (100 emails). Score = fraction of tickets
resolved within their SLA deadline.
"""

from __future__ import annotations

def grade(env, agent) -> float:
    obs, info = env.reset(seed=42)
    env.action_space.seed(42)
    agent.reset()
    
    resolved_on_time = 0
    total_assigned = 0
    done = False
    
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Accurately track assignments
        if int(action[2]) < 5:  # 5 employees: indices 0-4
            total_assigned += 1
            
        breakdown = info.get("reward_breakdown", [])
        for event in breakdown:
            if event["type"] == "resolve_on_time":
                resolved_on_time += 1
    
    return resolved_on_time / max(total_assigned, 1)
