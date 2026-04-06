"""
HelixDesk OpenEnv — Training entry point.

Usage:
  python train.py --agent rule       # runs rule-based agent, no learning
  python train.py --agent random     # runs random agent baseline
  python train.py --agent sb3        # trains with Stable-Baselines3 PPO (must be installed)
  python train.py --episodes 500
"""

import argparse
import sys

from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent
from helixdesk.monitor import EpisodeLogger, TerminalDashboard


try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


def run(agent_type: str, n_episodes: int) -> None:
    """Run training loop with the specified agent.

    Args:
        agent_type: One of 'rule', 'random', 'sb3'.
        n_episodes: Number of episodes to run.
    """
    env = HelixDeskEnv()
    logger = EpisodeLogger()
    dashboard = TerminalDashboard()

    if agent_type == "sb3":
        if not SB3_AVAILABLE:
            print("ERROR: stable-baselines3 not installed.")
            print("Install with: pip install stable-baselines3")
            sys.exit(1)

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=n_episodes * 100)
        model.save("helixdesk_ppo")
        print(f"\nModel saved to helixdesk_ppo.zip")
        env.close()
        logger.close()
        return

    if agent_type == "rule":
        agent = RuleAgent(env.observation_space, env.action_space)
    elif agent_type == "random":
        agent = RandomAgent(env.observation_space, env.action_space)
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)

    episode_rewards: list[float] = []

    with dashboard.live():
        for ep in range(n_episodes):
            obs, info = env.reset()
            agent.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.learn(obs, action, reward, obs, terminated, info)
                logger.log(ep, info, action=action, reward=reward)
                dashboard.update(ep, info, episode_rewards, action=action, reward=reward)
                ep_reward += reward
                done = terminated or truncated

            episode_rewards.append(ep_reward)

    # Final summary
    if episode_rewards:
        last_50 = episode_rewards[-50:]
        avg = sum(last_50) / len(last_50)
        print(f"\nFinal avg reward over last {len(last_50)} episodes: {avg:.3f}")
    else:
        print("\nNo episodes completed.")

    env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HelixDesk OpenEnv Trainer")
    parser.add_argument(
        "--agent",
        default="rule",
        choices=["rule", "random", "sb3"],
        help="Agent type to use (default: rule)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of episodes to run (default: 200)",
    )
    args = parser.parse_args()
    run(args.agent, args.episodes)
