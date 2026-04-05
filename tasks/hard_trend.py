"""Hard task: trend detection + CSAT quality.

Run one full episode (100 emails).
Score = 0.5 × (trend alerts caught / total surge events)
       + 0.5 × min(avg_csat / 4.0, 1.0)
"""

from __future__ import annotations

import numpy as np

from helixdesk import HelixDeskEnv


def grade(env: HelixDeskEnv | None = None, agent=None) -> float:
    """Grade the agent on trend alerting and CSAT quality.

    Args:
        env: Optional pre-built env. If None, creates one with default config.
        agent: Must implement .act(obs) -> action array.

    Returns:
        Score in [0.0, 1.0]: composite of trend catch rate + CSAT quality.
    """
    if env is None:
        env = HelixDeskEnv()

    obs, info = env.reset(seed=42)

    total_surge_steps = 0       # steps where trend_alerts > 0
    trend_alerts_caught = 0     # steps where agent alerted GM during a surge
    csat_scores: list[float] = []

    done = False
    while not done:
        # Check trend state in obs: dims 29-36 are growth rates
        trend_obs = obs[29:37]
        has_active_surge = bool(np.any(np.abs(trend_obs) > 0.3))

        action = agent.act(obs)
        action_arr = np.asarray(action, dtype=np.int64)

        obs, reward, terminated, truncated, info = env.step(action_arr)
        done = terminated or truncated

        # Track surge detection
        active_alerts = info.get("trend_alerts_active", 0)
        if active_alerts > 0 or has_active_surge:
            total_surge_steps += 1
            secondary = int(action_arr[3])
            if secondary == 1:  # alert_gm
                trend_alerts_caught += 1

        # Track CSAT
        csat = info.get("csat_score")
        if csat is not None:
            csat_scores.append(float(csat))

    env.close()

    # Trend catch rate component (50%)
    if total_surge_steps > 0:
        trend_score = trend_alerts_caught / total_surge_steps
    else:
        trend_score = 1.0  # No surges = perfect

    # CSAT quality component (50%)
    if csat_scores:
        avg_csat = np.mean(csat_scores)
        csat_component = min(avg_csat / 4.0, 1.0)
    else:
        csat_component = 0.0

    return 0.5 * trend_score + 0.5 * csat_component
