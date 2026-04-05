"""TerminalDashboard — Rich live dashboard for training visualisation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout


class TerminalDashboard:
    """Real-time terminal dashboard using Rich during training.

    Displays episode progress, queue state, team loads, overdue tickets,
    trend alerts, and the last action/reward breakdown.
    """

    def __init__(self, refresh_hz: float = 2.0):
        self._console = Console()
        self._refresh_hz = refresh_hz
        self._live: Live | None = None
        self._last_action_str: str = "—"
        self._last_reward_str: str = "—"

    @contextmanager
    def live(self) -> Generator[None, None, None]:
        """Context manager that keeps the Rich Live display running."""
        self._live = Live(
            self._build_display(0, {}, []),
            console=self._console,
            refresh_per_second=self._refresh_hz,
            transient=False,
        )
        with self._live:
            yield

    def update(
        self,
        episode: int,
        info: dict[str, Any],
        episode_rewards: list[float],
        action: Any = None,
        reward: float = 0.0,
    ) -> None:
        """Refresh the dashboard with the latest step data.

        Args:
            episode: Current episode index.
            info: The info dict from env.step().
            episode_rewards: List of all completed episode rewards so far.
            action: The action taken this step (optional).
            reward: The reward received this step.
        """
        if action is not None:
            act = np.asarray(action)
            classify_map = {0: "query", 1: "complaint", 2: "review"}
            priority_map = {0: "critical", 1: "high", 2: "medium", 3: "normal"}
            assign_str = f"emp_{act[2]}" if act[2] < 5 else "none"
            secondary_map = {0: "auto_reply", 1: "alert_gm", 2: "none"}
            self._last_action_str = (
                f"classify={classify_map.get(int(act[0]), '?')}  "
                f"priority={priority_map.get(int(act[1]), '?')}  "
                f"assign={assign_str}  "
                f"secondary={secondary_map.get(int(act[3]), '?')}"
            )

        # Format reward breakdown
        breakdown = info.get("reward_breakdown", {})
        if breakdown:
            parts = [f"{k} {v:+.2f}" for k, v in breakdown.items()]
            self._last_reward_str = f"{reward:+.2f}  [{', '.join(parts)}]"
        else:
            self._last_reward_str = f"{reward:+.2f}"

        if self._live:
            self._live.update(self._build_display(episode, info, episode_rewards))

    def _build_display(
        self, episode: int, info: dict[str, Any], episode_rewards: list[float]
    ) -> Panel:
        """Build the Rich renderable for the current state."""
        step = info.get("step", 0)
        total_steps = 100  # from config
        sim_minutes = info.get("sim_time_minutes", 0)
        sim_hours = int(sim_minutes // 60)
        sim_mins = int(sim_minutes % 60)
        ep_reward = info.get("episode_reward_so_far", 0.0)

        # Running average
        last_n = episode_rewards[-10:] if episode_rewards else []
        running_avg = sum(last_n) / max(len(last_n), 1)

        # Build main table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan", width=14)
        table.add_column("Value", style="white")

        table.add_row(
            "Episode",
            f"{episode}    Step: {step}/{total_steps}    Sim time: {sim_hours}h {sim_mins:02d}m",
        )
        table.add_row(
            "Ep reward",
            f"{ep_reward:+.2f}    Running avg (last 10): {running_avg:+.1f}",
        )
        table.add_row("", "")

        # Queue state
        queue_depth = info.get("queue_depth", 0)
        overdue = info.get("overdue_count", 0)
        trends = info.get("trend_alerts_active", 0)
        table.add_row("Queue depth", str(queue_depth))
        table.add_row("Overdue", str(overdue))
        table.add_row("Trend alerts", str(trends))
        table.add_row("", "")

        # Last action
        table.add_row("Last action", self._last_action_str)
        table.add_row("Last reward", self._last_reward_str)

        return Panel(
            table,
            title="[bold bright_green]HelixDesk OpenEnv[/bold bright_green]",
            border_style="bright_green",
        )
