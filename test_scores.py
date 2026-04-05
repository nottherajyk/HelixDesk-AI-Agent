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
        overdue_count = 0
        total_tickets = 0
        trend_alerts_caught = 0
        total_surge_events = 0
        csat_scores = []
        
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Count the actual values from reward breakdown values if possible
            breakdown = info.get("reward_breakdown", {})
            
            # The event strings in breakdown are keys, but wait!
            # Could we use overdue_count for medium_sla?
            overdue_count = info.get("overdue_count", 0)
            total_tickets = info.get("step", 1)
            
            for event in breakdown:
                if event == "resolve_on_time":
                    resolved_on_time += 1 # wait, this just counts steps. 
                if event == "trend_prevented":
                    trend_alerts_caught += 1
            
            # Actually, total_surge_events should be active_alerts logic from before?
            active_alerts = info.get("trend_alerts_active", 0)
            if active_alerts > 0:
                total_surge_events += 1
                
            if info.get("csat_score") is not None:
                csat_scores.append(info.get("csat_score"))
                
        # let's try different metric denominators
        print(f"--- {name} ---")
        print(f"resolved_on_time: {resolved_on_time}, overdue_count (peak): {overdue_count}")
        print(f"total_tickets: {total_tickets}")
        sla_1 = resolved_on_time / max(total_tickets, 1)
        sla_2 = 1.0 - (overdue_count / max(total_tickets, 1))
        print(f"SLA if resolved/total: {sla_1:.3f}")
        print(f"SLA if 1 - overdue/total: {sla_2:.3f}")
        
        # Hard trend
        trend_score = trend_alerts_caught / max(total_surge_events, 1)
        csat_score = min(
            (sum(csat_scores) / max(len(csat_scores), 1)) / 4.7, 1.0
        )
        overdue_score = max(0.0, 1.0 - (overdue_count / max(total_tickets * 0.10, 1)))

        print(f"Trend: caught={trend_alerts_caught}, surges={total_surge_events} -> {trend_score:.3f}")
        print(f"CSAT: mean_csat={sum(csat_scores)/max(len(csat_scores),1):.2f} -> {csat_score:.3f}")
        print(f"Overdue penalty -> {overdue_score:.3f}")
        print(f"Hard Trend Product: {trend_score * csat_score * overdue_score:.3f}")
        print(f"Hard Trend Average: {(trend_score + csat_score + overdue_score) / 3:.3f}")
        
test()
