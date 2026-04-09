import numpy as np

def grade(env, agent, seed=42) -> float:
    """
    Expert task: Evaluate full behavior using geometric mean of components.
    Made adversarial to evaluate beyond simple rules.
    """
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    
    score_penalty = 0.0
    steps = 0
    done = False
    
    while not done:
        # Adversarial workload injection:
        # Make Employee 0 severely overloaded.
        obs[15] = 1.0 
        
        action = agent.act(obs)
        
        # If the agent ignores the workload balance and assigns to Employee 0 anyway
        if action[0] == 1 and action[2] == 0:
            score_penalty += 0.05
            
        obs, r, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc
        
    return float(max(0.0, 1.0 - score_penalty))
