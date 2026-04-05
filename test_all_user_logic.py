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

        trend_alerts_caught = 0
        total_surge_events = 0
        csat_scores = []
        overdue_count = 0
        total_tickets = 0
        
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # User snippet for medium_sla
            breakdown = info.get("reward_breakdown", {})
            for event in breakdown:
                if event == "resolve_on_time":
                    resolved_on_time += 1
                if event in ("resolve_on_time", "missed_deadline"):
                    total_assigned += 1
            
            # User snippet for hard_trend
            for event in breakdown:
                if event == "trend_prevented":
                    trend_alerts_caught += 1
                if event == "resolve_on_time":
                    total_surge_events += 1
            
            if info.get("csat_score") is not None:
                csat_scores.append(info.get("csat_score"))
            
            overdue_count = info.get("overdue_count", 0)
            total_tickets = info.get("step", 1)
                    
        print(f"--- {name} ---")
        
        # calculate medium sla
        med_score = resolved_on_time / max(total_assigned, 1)
        print(f"Medium SLA -> {med_score:.3f}")
        
        # calculate hard trend
        trend_score = trend_alerts_caught / max(total_surge_events, 1)
        csat_score = min(
            (sum(csat_scores) / max(len(csat_scores), 1)) / 4.5, 1.0
        )
        overdue_score = max(
            0.0, 1.0 - (overdue_count / max(total_tickets * 0.10, 1))
        )
        
        hard_score = (trend_score + csat_score + overdue_score) / 3
        print(f"Hard Trend -> {hard_score:.3f}")
test()
