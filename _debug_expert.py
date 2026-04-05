from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent
from tasks.expert_full import grade

env = HelixDeskEnv()
rule = RuleAgent(env.observation_space, env.action_space)

obs, info = env.reset(seed=42)
env.action_space.seed(42)
rule.reset()

total_keyword_emails = 0
keyword_misses = 0
resolved_on_time = 0
total_assigned = 0
trend_alerts_caught = 0
total_surge_events = 0
csat_scores = []
misclassifications = 0
total_classified = 0

done = False

while not done:
    has_keyword = obs[1] > 0.5
    if has_keyword:
        total_keyword_emails += 1
        
    action = rule.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    breakdown = info.get("reward_breakdown", {})
    
    if has_keyword:
        if "keyword_flag_missed" in breakdown or "keyword_not_critical" in breakdown:
            keyword_misses += 1
            
    if int(action[2]) < 4:
        total_assigned += 1
    if "resolve_on_time" in breakdown:
        resolved_on_time += 1
        
    if "trend_prevented" in breakdown:
        trend_alerts_caught += 1
    if info.get("trend_alerts_active", 0) > 0:
        total_surge_events += 1
        
    if info.get("csat_score") is not None:
        csat_scores.append(info.get("csat_score"))
        
    if "misclassification" in breakdown:
        misclassifications += 1
    total_classified += 1

print(f"keyword_misses: {keyword_misses} / {total_keyword_emails}")
print(f"resolved_on_time: {resolved_on_time} / {total_assigned}")
print(f"trend_alerts_caught: {trend_alerts_caught} / {total_surge_events}")
print(f"csat_score length: {len(csat_scores)}")
print(f"misclassifications: {misclassifications} / {total_classified}")
