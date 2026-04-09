import numpy as np

def grade(env, agent, seed=42) -> float:
    """
    Hard task: Trend detection + CSAT + Penalize naive sentiment rules.
    Made harder for rule agents.
    """
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    
    misclassifications = 0
    steps = 0
    done = False
    
    while not done:
        # Make the task harder: inject fake high sentiment queries to trick naive rule agents
        is_trick = False
        if steps % 4 == 0:
            obs[0] = 0.95  # extremely high sentiment
            obs[1] = 0.0   # no keyword
            is_trick = True
            
        action = agent.act(obs)
        
        # A simple rule agent says: if sentiment > 0.7, it's a complaint (action[0]==1).
        # But high sentiment queries exist!
        if is_trick and action[0] == 1:
            misclassifications += 1
            
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc
        
    trick_cases = max(steps // 4, 1)
    score = 1.0 - (misclassifications / trick_cases)
    
    # Mix with CSAT if present
    csat = info.get("csat_score", 5.0)
    final_score = (score + (csat / 5.0)) / 2.0
    
    return float(max(0.0, min(1.0, final_score)))
