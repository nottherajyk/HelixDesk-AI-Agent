from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent

def test():
    env = HelixDeskEnv()
    for agent_class, name in [(RandomAgent, "random"), (RuleAgent, "rule")]:
        agent = agent_class(env.observation_space, env.action_space)
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
            
            # User's literal snippet:
            breakdown = info.get("reward_breakdown", {})
            for event in breakdown:
                if event == "resolve_on_time":
                    resolved_on_time += 1
                if event in ("resolve_on_time", "missed_deadline"):
                    total_assigned += 1
                    
        print(f"--- {name} ---")
        print(f"User Snippet result -> {resolved_on_time / max(total_assigned, 1):.3f}")
test()
