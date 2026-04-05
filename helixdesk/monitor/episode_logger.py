"""EpisodeLogger — writes per-step metrics to CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, TextIO

import numpy as np


class EpisodeLogger:
    """Logs one CSV row per environment step for post-training analysis.

    CSV columns:
        episode, step, sim_time, action_classify, action_priority,
        action_assign, action_secondary, reward, episode_reward,
        queue_depth, overdue_count, trend_alerts, ticket_type, priority_assigned
    """

    _FIELDNAMES = [
        "episode",
        "step",
        "sim_time",
        "action_classify",
        "action_priority",
        "action_assign",
        "action_secondary",
        "reward",
        "episode_reward",
        "queue_depth",
        "overdue_count",
        "trend_alerts",
        "ticket_type",
        "priority_assigned",
    ]

    def __init__(self, log_dir: str = "./logs/", enabled: bool = True):
        self._enabled = enabled
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None

        if self._enabled:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            filepath = log_path / "episode_log.csv"
            self._file = open(filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDNAMES)
            self._writer.writeheader()

    def log(self, episode: int, info: dict[str, Any], action: Any = None, reward: float = 0.0) -> None:
        """Write one row for a single step.

        Args:
            episode: Current episode number.
            info: The info dict returned by env.step().
            action: The action taken (optional, for logging action components).
            reward: The step reward.
        """
        if not self._enabled or self._writer is None:
            return

        row = {
            "episode": episode,
            "step": info.get("step", ""),
            "sim_time": info.get("sim_time_minutes", ""),
            "action_classify": "",
            "action_priority": "",
            "action_assign": "",
            "action_secondary": "",
            "reward": reward,
            "episode_reward": info.get("episode_reward_so_far", ""),
            "queue_depth": info.get("queue_depth", ""),
            "overdue_count": info.get("overdue_count", ""),
            "trend_alerts": info.get("trend_alerts_active", ""),
            "ticket_type": info.get("ticket_type", ""),
            "priority_assigned": info.get("priority", ""),
        }

        if action is not None:
            action = np.asarray(action)
            if len(action) >= 4:
                row["action_classify"] = int(action[0])
                row["action_priority"] = int(action[1])
                row["action_assign"] = int(action[2])
                row["action_secondary"] = int(action[3])

        self._writer.writerow(row)
        if self._file:
            self._file.flush()

    def close(self) -> None:
        """Close the underlying CSV file."""
        if self._file:
            self._file.close()
            self._file = None
