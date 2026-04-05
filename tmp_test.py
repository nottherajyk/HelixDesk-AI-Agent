from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent

env = HelixDeskEnv()
agent = RuleAgent(env.observation_space, env.action_space)

obs, info = env.reset(seed=42)
env.action_space.seed(42)
agent.reset()

metrics = {
    'kw_emails': 0, 'kw_misses': 0, 
    'assigned': 0, 'on_time': 0, 
    'surges': 0, 'trend_caught': 0, 
    'csat': [], 'misclass': 0, 
    'total': 0, 'trend_active_prev': 0
}

for _ in range(100):
    has_kw = obs[1] > 0.5
    if has_kw: 
        metrics['kw_emails'] += 1
    
    act = agent.act(obs)
    obs, r, term, trunc, info = env.step(act)
    bd = info.get('reward_breakdown', {})
    
    if has_kw and ('keyword_flag_missed' in bd or 'keyword_not_critical' in bd):
        metrics['kw_misses'] += 1
        
    if int(act[2]) < 4: 
        metrics['assigned'] += 1
        
    if 'resolve_on_time' in bd: 
        metrics['on_time'] += 1
    
    if 'trend_prevented' in bd: 
        metrics['trend_caught'] += 1
    
    active = info.get('trend_alerts_active', 0)
    if active > metrics['trend_active_prev']:
        metrics['surges'] += (active - metrics['trend_active_prev'])
    metrics['trend_active_prev'] = active
    
    if info.get('csat_score') is not None: 
        metrics['csat'].append(info.get('csat_score'))
        
    if 'misclassification' in bd: 
        metrics['misclass'] += 1
        
    metrics['total'] += 1
    
print(f"on_time: {metrics['on_time']}, assigned: {metrics['assigned']}")
print(f"surges: {metrics['surges']}, trend_caught: {metrics['trend_caught']}")
print(f"misclass: {metrics['misclass']}, total: {metrics['total']}")
