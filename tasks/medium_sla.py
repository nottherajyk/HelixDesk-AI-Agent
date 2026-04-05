"""Medium task: SLA compliance rate.

Run one full episode (100 emails). Score = fraction of tickets
resolved within their SLA deadline.
"""

from __future__ import annotations

from helixdesk import HelixDeskEnv


def grade(env: HelixDeskEnv | None = None, agent=None) -> float:
    """Grade the agent on SLA compliance.

    Args:
        env: Optional pre-built env. If None, creates one with default config.
        agent: Must implement .act(obs) -> action array.

    Returns:
        Score in [0.0, 1.0]: fraction of tickets resolved within SLA.
    """
    if env is None:
        env = HelixDeskEnv()

    obs, info = env.reset(seed=42)
    total_resolutions = 0
    on_time_resolutions = 0

    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Count resolution events from reward breakdown
        breakdown = info.get("reward_breakdown", {})
        if "resolve_on_time" in breakdown:
            on_time_resolutions += 1
            total_resolutions += 1
        if "missed_deadline" in breakdown:
            total_resolutions += 1

    env.close()

    if total_resolutions == 0:
        return 0.0  # No resolutions at all = 0 score

    return on_time_resolutions / total_resolutions
