from helixdesk import HelixDeskEnv
env = HelixDeskEnv()
obs, info = env.reset(seed=42)
rewards = []
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    rewards.append(reward)
    assert -1.0 <= reward <= 1.0, f"Reward {reward} out of bounds"
    assert obs.shape == (42,), f"Observation shape {obs.shape} != (42,)"
    assert "reward_breakdown" in info, "reward_breakdown missing from info"
assert terminated == True, "Episode did not terminate at 100 steps"
print(f"All checks passed. Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
