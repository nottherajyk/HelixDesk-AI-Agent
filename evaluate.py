"""
HelixDesk OpenEnv — Evaluation script.

Runs a trained agent for N episodes and prints:
- Mean episode reward ± std
- Mean overdue rate (overdue tickets / total tickets)
- Mean CSAT score on auto-replies
- Keyword flag miss rate
- Workload balance score (mean std dev of employee loads per episode)
- Misclassification rate

Usage:
  python evaluate.py --agent rule --episodes 100
  python evaluate.py --agent random --episodes 50
"""

import argparse
import sys

import numpy as np
from rich.console import Console
from rich.table import Table

from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent


def evaluate(agent_type: str, n_episodes: int) -> None:
    """Run evaluation loop and display results.

    Args:
        agent_type: One of 'rule', 'random'.
        n_episodes: Number of evaluation episodes.
    """
    env = HelixDeskEnv()
    console = Console()

    if agent_type == "rule":
        agent = RuleAgent(env.observation_space, env.action_space)
    elif agent_type == "random":
        agent = RandomAgent(env.observation_space, env.action_space)
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)

    # Accumulators
    episode_rewards: list[float] = []
    episode_overdue_rates: list[float] = []
    episode_csat_scores: list[float] = []
    episode_keyword_miss_rates: list[float] = []
    episode_workload_stds: list[float] = []
    episode_misclass_rates: list[float] = []

    console.print(f"\n[bold cyan]Evaluating {agent_type} agent for {n_episodes} episodes...[/bold cyan]\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()
        ep_reward = 0.0
        done = False

        step_overdue_counts: list[int] = []
        step_csat_scores: list[float] = []
        keyword_misses = 0
        keyword_total = 0
        misclassifications = 0
        total_steps = 0
        load_stds: list[float] = []

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

            # Track overdue
            step_overdue_counts.append(info.get("overdue_count", 0))

            # Track CSAT
            csat = info.get("csat_score")
            if csat is not None:
                step_csat_scores.append(float(csat))

            # Track keyword misses & misclassifications from reward breakdown
            breakdown = info.get("reward_breakdown", [])
            for event in breakdown:
                if event["type"] == "keyword_flag_missed":
                    keyword_misses += 1
                if event["type"] == "keyword_not_critical":
                    keyword_total += 1
                if event["type"] == "misclassification":
                    misclassifications += 1

            # Track workload balance
            employee_loads = [obs[15 + i * 2] for i in range(5)]
            load_stds.append(float(np.std(employee_loads)))

        episode_rewards.append(ep_reward)

        # Overdue rate
        avg_overdue = np.mean(step_overdue_counts) if step_overdue_counts else 0.0
        overdue_rate = avg_overdue / max(total_steps, 1)
        episode_overdue_rates.append(overdue_rate)

        # CSAT
        if step_csat_scores:
            episode_csat_scores.append(np.mean(step_csat_scores))

        # Keyword miss rate
        episode_keyword_miss_rates.append(keyword_misses / max(total_steps, 1))

        # Workload balance
        episode_workload_stds.append(np.mean(load_stds) if load_stds else 0.0)

        # Misclassification rate
        episode_misclass_rates.append(misclassifications / max(total_steps, 1))

    env.close()

    # --- Build results table ---
    table = Table(title="HelixDesk Evaluation Results", border_style="bright_green")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="white", justify="right")

    reward_arr = np.array(episode_rewards)
    table.add_row(
        "Mean Episode Reward",
        f"{reward_arr.mean():+.3f} ± {reward_arr.std():.3f}",
    )
    table.add_row(
        "Mean Overdue Rate",
        f"{np.mean(episode_overdue_rates):.4f}",
    )
    table.add_row(
        "Mean CSAT Score",
        f"{np.mean(episode_csat_scores):.2f}" if episode_csat_scores else "N/A",
    )
    table.add_row(
        "Keyword Flag Miss Rate",
        f"{np.mean(episode_keyword_miss_rates):.4f}",
    )
    table.add_row(
        "Workload Balance (mean load std)",
        f"{np.mean(episode_workload_stds):.4f}",
    )
    table.add_row(
        "Misclassification Rate",
        f"{np.mean(episode_misclass_rates):.4f}",
    )
    table.add_row("", "")
    table.add_row("Agent", agent_type)
    table.add_row("Episodes", str(n_episodes))

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HelixDesk OpenEnv Evaluator")
    parser.add_argument(
        "--agent",
        default="rule",
        choices=["rule", "random"],
        help="Agent type to evaluate (default: rule)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)",
    )
    args = parser.parse_args()
    evaluate(args.agent, args.episodes)
