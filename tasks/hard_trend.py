"""Hard task: trend detection + CSAT quality + overdue control.

Run one full episode (100 emails). Score = average of three components:
  1. trend_score  = trend alerts caught / total surge events
  2. csat_score   = min(avg_csat / 4.5, 1.0)
  3. overdue_score = max(0, 1 - overdue_count / (steps * 0.10))
"""

from __future__ import annotations

from helixdesk import HelixDeskEnv

def grade(env, agent) -> float:
    obs, info = env.reset(seed=42)
    env.action_space.seed(42)
    agent.reset()
    
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
        
        breakdown = info.get("reward_breakdown", {})
        for event in breakdown:
            if event == "trend_prevented":
                trend_alerts_caught += 1
                
        # Track true surge events instead of resolve_on_time
        if info.get("trend_alerts_active", 0) > 0:
            total_surge_events += 1
        
        if info.get("csat_score") is not None:
            csat_scores.append(info.get("csat_score"))
        
        # Accumulate ongoing overdue backlog across steps
        overdue_count += info.get("overdue_count", 0)
        total_tickets = info.get("step", 1)
    
    trend_score = trend_alerts_caught / max(total_surge_events, 1)
    csat_score = min(
        (sum(csat_scores) / max(len(csat_scores), 1)) / 4.5, 1.0
    )
    overdue_score = max(
        0.0, 1.0 - (overdue_count / max(total_tickets * 0.10, 1))
    )
    
    return (trend_score + csat_score + overdue_score) / 3
