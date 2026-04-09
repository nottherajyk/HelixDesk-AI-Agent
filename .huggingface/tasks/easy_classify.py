"""Easy task: keyword-flag classification accuracy.

Run 20 emails with a fixed seed. Score = fraction of keyword-flagged
emails that the agent correctly classifies as complaint with critical priority.
"""

from __future__ import annotations

import numpy as np


def grade(env, agent) -> float:
    """Grade the agent on keyword-flag classification.

    Args:
        env: A HelixDesk environment instance.
        agent: Must implement .act(obs) -> action array.

    Returns:
        Score in [0.0, 1.0]: fraction of keyword-flagged emails correctly
        classified as complaint with critical priority.
    """
    obs, info = env.reset(seed=42)
    env.action_space.seed(42)  # seed action space for deterministic random agent
    keyword_total = 0
    keyword_correct = 0

    for step in range(20):
        # Read keyword flag from obs (dim 1)
        has_keyword = obs[1] > 0.5

        action = agent.act(obs)
        action = np.asarray(action, dtype=np.int64)

        if has_keyword:
            keyword_total += 1
            classify = int(action[0])
            priority = int(action[1])
            # Correct = classified as complaint (1) with critical priority (0)
            if classify == 1 and priority == 0:
                keyword_correct += 1

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break



    if keyword_total == 0:
        return 1.0  # No keyword emails to miss = perfect score

    return keyword_correct / keyword_total
