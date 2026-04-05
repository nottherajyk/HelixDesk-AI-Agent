from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent
from tasks.expert_full import grade

env = HelixDeskEnv()
rule = RuleAgent(env.observation_space, env.action_space)
rand = RandomAgent(env.observation_space, env.action_space)

rule_score = grade(env, rule)
rand_score = grade(env, rand)
print(f'Rule agent:   {rule_score:.4f}')
print(f'Random agent: {rand_score:.4f}')
assert rule_score <= 0.35, f'Too easy for rule agent: {rule_score}'
assert rand_score <= 0.08, f'Too easy for random agent: {rand_score}'
assert rule_score > rand_score, 'Rule must beat random'
print('Expert task check PASSED')
