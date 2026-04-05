"""HelixDeskEnv — the main Gymnasium environment for email queue management."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import yaml

from helixdesk.rewards import RewardFunction
from helixdesk.simulator.clock import SimClock
from helixdesk.simulator.email_gen import EmailEvent, EmailGenerator
from helixdesk.simulator.employee_sim import EmployeeSimulator
from helixdesk.simulator.knowledge_base import KnowledgeBase
from helixdesk.simulator.trend_watchdog import TrendWatchdog
from helixdesk.spaces import (
    CATEGORY_ENCODING,
    OBS_SIZE,
    REVIEW_FORCED_ACTION,
    build_action_space,
    build_observation_space,
    encode_category,
    encode_customer_tier,
)

logger = logging.getLogger(__name__)


class HelixDeskEnv(gymnasium.Env):
    """Gymnasium environment simulating a customer email helpdesk.

    An RL agent processes incoming emails one at a time, deciding how to
    classify, prioritise, assign, and respond to each. The environment
    simulates employee resolution, SLA tracking, trend detection, and
    knowledge base management.

    Compatible with gymnasium >= 0.29 and Stable-Baselines3 / RLlib / CleanRL.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, config_path: str = "config.yaml", render_mode: str | None = None):
        super().__init__()

        # --- Load config ---
        config_file = Path(config_path)
        if not config_file.is_absolute():
            # Search relative to this file's directory, then CWD
            candidates = [
                Path(__file__).parent.parent / config_path,
                Path.cwd() / config_path,
            ]
            for c in candidates:
                if c.exists():
                    config_file = c
                    break

        with open(config_file, "r") as f:
            self._config: dict = yaml.safe_load(f)

        env_cfg = self._config["env"]
        sla_cfg = self._config["sla"]

        self._episode_emails: int = env_cfg["episode_emails"]
        self._n_employees: int = env_cfg["n_employees"]
        self._categories: list[str] = self._config["email_gen"]["categories"]
        self._seed_val: int = env_cfg["seed"]

        # SLA deadlines in minutes
        self._sla_deadlines: dict[str, float] = {
            "critical": sla_cfg["critical_hours"] * 60.0,
            "high": sla_cfg["high_hours"] * 60.0,
            "medium": sla_cfg["medium_hours"] * 60.0,
            "normal": sla_cfg["normal_hours"] * 60.0,
        }
        self._max_employee_load: int = sla_cfg["max_employee_load"]

        # --- Spaces ---
        self.observation_space = build_observation_space()
        self.action_space = build_action_space()
        self.render_mode = render_mode

        # --- Simulators ---
        self._clock = SimClock(self._seed_val)
        self._email_gen = EmailGenerator(self._config, self._seed_val + 1)
        self._employee_sim = EmployeeSimulator(self._config, self._seed_val + 2)
        self._kb = KnowledgeBase()
        self._trend = TrendWatchdog(self._config)
        self._reward_fn = RewardFunction(self._config)

        # --- Internal state ---
        self._queue: list[dict[str, Any]] = []  # open tickets
        self._current_email: EmailEvent | None = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._last_obs: np.ndarray | None = None
        self._prev_employee_loads: list[int] = [0] * self._n_employees
        self._last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode (simulated inbox day).

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Unused, reserved for Gymnasium compatibility.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)

        if seed is not None:
            self._seed_val = seed
            self._clock = SimClock(seed)
            self._email_gen = EmailGenerator(self._config, seed + 1)
            self._employee_sim = EmployeeSimulator(self._config, seed + 2)

        # Reset all simulators
        self._clock.reset()
        self._employee_sim.reset()
        self._trend.reset()
        self._kb = KnowledgeBase()

        # Reset internal state
        self._queue = []
        self._step_count = 0
        self._episode_reward = 0.0
        self._prev_employee_loads = [0] * self._n_employees

        # Generate first email
        self._current_email = self._email_gen.next(self._clock.minutes)

        # Build observation
        obs = self._build_observation()
        self._last_obs = obs.copy()

        info = {
            "step": 0,
            "email_id": self._current_email.email_id,
            "sim_time": 0.0,
        }
        self._last_info = info

        return obs, info

    def step(
        self, action: np.ndarray | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Process the current email with the given action and advance simulation.

        Args:
            action: 4-element array [classify, priority, assign, secondary].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.int64)
        assert self._current_email is not None, "Must call reset() before step()"

        email = self._current_email
        self._step_count += 1

        # 1. Decode & enforce action rules
        classify = int(action[0])
        priority = int(action[1])
        assign = int(action[2])
        secondary = int(action[3])

        # Force action when flagging for review
        if classify == 2:  # flag_for_human_review
            priority, assign, secondary = REVIEW_FORCED_ACTION

        # 2. Apply action to current email
        kb_updated = False
        priority_labels = {0: "critical", 1: "high", 2: "medium", 3: "normal"}
        priority_label = priority_labels[priority]

        if classify == 0:  # query
            ticket_entry = {
                "ticket_id": email.email_id,
                "type": "query",
                "priority": "normal",
                "status": "auto_replied",
                "created_at": self._clock.minutes,
                "sla_deadline": self._clock.minutes + self._sla_deadlines["normal"],
            }
            # Auto-reply from KB
            if secondary == 0:
                kb_entry, similarity = self._kb.lookup(email.category, email.sentiment_intensity)
                if similarity >= 0.5 and kb_entry is not None:
                    ticket_entry["status"] = "resolved_auto"
                    # Learn from this resolution
                    self._kb.add_entry(
                        email.category,
                        [email.category.replace("_", " ")],
                        f"Auto-resolved: {email.category}",
                    )
                    kb_updated = True
            self._queue.append(ticket_entry)

        elif classify == 1:  # complaint
            sla_deadline = self._clock.minutes + self._sla_deadlines[priority_label]
            ticket_entry = {
                "ticket_id": email.email_id,
                "type": "complaint",
                "priority": priority_label,
                "status": "open",
                "created_at": self._clock.minutes,
                "sla_deadline": sla_deadline,
                "assigned_to": None,
            }

            if assign < self._n_employees:
                try:
                    self._employee_sim.assign(assign, email.email_id, sla_deadline)
                    ticket_entry["assigned_to"] = assign
                    ticket_entry["status"] = "assigned"
                except ValueError:
                    # Employee at max load — goes to unassigned queue
                    ticket_entry["status"] = "unassigned"
            else:
                ticket_entry["status"] = "unassigned"

            # Record for trend tracking
            self._trend.record(email.category, self._clock.minutes)
            self._queue.append(ticket_entry)

        else:  # flag_for_human_review
            ticket_entry = {
                "ticket_id": email.email_id,
                "type": "pending_review",
                "priority": "normal",
                "status": "pending_review",
                "created_at": self._clock.minutes,
                "sla_deadline": self._clock.minutes + self._sla_deadlines["normal"],
            }
            self._queue.append(ticket_entry)

        # 3. Advance SimClock
        self._clock.tick()

        # 4. Employee tick — resolve or miss tickets
        resolution_events = self._employee_sim.tick(self._clock.minutes)

        # Remove resolved/missed tickets from queue
        resolved_ids = {ev.ticket_id for ev in resolution_events}
        self._queue = [t for t in self._queue if t["ticket_id"] not in resolved_ids]

        # 5. Trend watchdog tick
        trend_alerts = self._trend.tick(self._clock.minutes)

        # 6. SLA watchdog pass — mark overdue tickets
        overdue_count = 0
        near_deadline_count = 0
        critical_overdue = False
        for ticket in self._queue:
            if ticket["status"] in ("resolved_auto", "pending_review"):
                continue
            remaining = ticket["sla_deadline"] - self._clock.minutes
            if remaining <= 0:
                ticket["status"] = "overdue"
                overdue_count += 1
                if ticket["priority"] == "critical":
                    critical_overdue = True
            elif remaining < 120.0:  # less than 2 hours
                near_deadline_count += 1

        # 7. Compute reward
        employee_loads = self._employee_sim.get_loads()
        queue_state = {
            "overdue_count": overdue_count,
            "near_deadline_count": near_deadline_count,
        }
        total_reward, reward_events = self._reward_fn.compute(
            action=action,
            email=email,
            resolution_events=resolution_events,
            trend_alerts=trend_alerts,
            queue_state=queue_state,
            kb_updated=kb_updated,
            employee_loads=employee_loads,
            prev_employee_loads=self._prev_employee_loads,
        )
        self._prev_employee_loads = list(employee_loads)
        self._episode_reward += total_reward

        # 8. Generate next email
        self._current_email = self._email_gen.next(self._clock.minutes)

        # 9. Build observation
        obs = self._build_observation()
        self._last_obs = obs.copy()

        # 10/11. Termination
        terminated = self._step_count >= self._episode_emails
        truncated = False

        # 12. Info dict
        classification_map = {0: "query", 1: "complaint", 2: "pending_review"}
        info: dict[str, Any] = {
            "step": self._step_count,
            "sim_time_minutes": self._clock.minutes,
            "email_id": email.email_id,
            "ticket_type": classification_map[classify],
            "priority": priority_label,
            "assigned_to": assign if assign < self._n_employees and classify == 1 else None,
            "reward_breakdown": {ev.event_type: ev.value for ev in reward_events},
            "queue_depth": len(self._queue),
            "overdue_count": overdue_count,
            "trend_alerts_active": len(trend_alerts),
            "csat_score": (
                next(
                    (ev.csat_score for ev in resolution_events if ev.resolved and ev.csat_score),
                    None,
                )
            ),
            "episode_reward_so_far": self._episode_reward,
        }
        self._last_info = info

        if self.render_mode == "human":
            self.render()

        return obs, total_reward, terminated, truncated, info

    def state(self) -> np.ndarray:
        """Return current observation without advancing simulation.

        Idempotent — calling state() N times gives the same result until
        step() is called.

        Returns:
            The last computed observation vector (42-dim float32).
        """
        if self._last_obs is not None:
            return self._last_obs.copy()
        return self._build_observation()

    def render(self) -> str | None:
        """Render the current environment state.

        Returns:
            String representation if mode='ansi', None if mode='human' (prints to stdout).
        """
        info = self._last_info
        line = (
            f"Step {info.get('step', '?')}/{self._episode_emails} | "
            f"Reward: {self._episode_reward:+.2f} | "
            f"Queue: {info.get('queue_depth', '?')} | "
            f"Overdue: {info.get('overdue_count', 0)} | "
            f"Trends: {info.get('trend_alerts_active', 0)}"
        )

        if self.render_mode == "human":
            print(line)
            return None
        elif self.render_mode == "ansi":
            return line
        return None

    def close(self) -> None:
        """Clean up any open resources."""
        pass  # No persistent resources; CSV logger is managed externally

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """Build the 42-dimensional observation vector from current state."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        email = self._current_email

        if email is None:
            return obs

        # --- Current email features (dims 0–9) ---
        obs[0] = email.sentiment_intensity
        obs[1] = 1.0 if email.has_keyword_flag else 0.0

        tier = encode_customer_tier(email.customer_tier)
        obs[2] = tier[0]
        obs[3] = tier[1]
        obs[4] = tier[2]

        cat_enc = encode_category(email.category, self._categories)
        obs[5:10] = cat_enc

        # --- Queue state (dims 10–14) ---
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "normal": 0}
        review_count = 0
        for ticket in self._queue:
            if ticket["status"] == "pending_review":
                review_count += 1
            elif ticket["priority"] in priority_counts:
                priority_counts[ticket["priority"]] += 1

        obs[10] = priority_counts["critical"] / max(self._max_employee_load, 1)
        obs[11] = priority_counts["high"] / max(self._max_employee_load, 1)
        obs[12] = priority_counts["medium"] / max(self._max_employee_load, 1)
        obs[13] = priority_counts["normal"] / max(self._max_employee_load, 1)
        obs[14] = review_count / 10.0

        # --- Team state (dims 15–24) ---
        loads = self._employee_sim.get_loads()
        resolve_times = self._employee_sim.get_avg_resolve_times()
        sla_normal_minutes = self._sla_deadlines["normal"]

        for i in range(self._n_employees):
            obs[15 + i * 2] = loads[i] / max(self._max_employee_load, 1)
            obs[16 + i * 2] = resolve_times[i] / max(sla_normal_minutes, 1)

        # --- SLA state (dims 25–28) ---
        total_open = max(len(self._queue), 1)
        overdue = sum(
            1 for t in self._queue
            if t.get("status") == "overdue"
        )
        near_deadline = sum(
            1 for t in self._queue
            if t.get("status") not in ("resolved_auto", "pending_review", "overdue")
            and (t["sla_deadline"] - self._clock.minutes) < 120.0
            and (t["sla_deadline"] - self._clock.minutes) > 0
        )

        obs[25] = overdue / total_open
        obs[26] = near_deadline / total_open

        # SLA pressure: inverted normalized time remaining
        sla_pressure = 0.0
        active_tickets = [
            t for t in self._queue
            if t.get("status") not in ("resolved_auto", "pending_review")
        ]
        if active_tickets:
            for t in active_tickets:
                remaining = max(t["sla_deadline"] - self._clock.minutes, 0)
                deadline_window = self._sla_deadlines.get(t["priority"], self._sla_deadlines["normal"])
                sla_pressure += 1.0 - (remaining / max(deadline_window, 1))
            sla_pressure /= len(active_tickets)
        obs[27] = float(np.clip(sla_pressure, -1.0, 1.0))

        obs[28] = 1.0 if any(
            t.get("status") == "overdue" and t.get("priority") == "critical"
            for t in self._queue
        ) else 0.0

        # --- Trend state (dims 29–36) ---
        growth_rates = self._trend.get_growth_rates(self._clock.minutes)
        for i, cat in enumerate(self._categories):
            obs[29 + i] = growth_rates.get(cat, 0.0)

        # --- Time state (dims 37–38) ---
        obs[37] = self._clock.hour_of_day / 24.0
        obs[38] = self._clock.day_of_week / 7.0

        # --- Episode progress (dims 39–41) ---
        obs[39] = (self._episode_emails - self._step_count) / max(self._episode_emails, 1)
        obs[40] = float(np.clip(
            self._episode_reward / max(self._episode_emails, 1), -1.0, 1.0
        ))
        obs[41] = 1.0  # agent_confidence — default, can be set externally

        # Clip entire obs to [-1, 1]
        obs = np.clip(obs, -1.0, 1.0)

        return obs

    # ------------------------------------------------------------------
    # Typed API (Pydantic wrappers)
    # ------------------------------------------------------------------

    def typed_reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], "HelixObservation"]:
        """Reset the environment and return typed observation alongside numpy.

        Returns:
            (obs_numpy, info, obs_typed) — the third element is a Pydantic model.
        """
        from helixdesk.models import HelixObservation

        obs, info = self.reset(seed=seed, options=options)
        return obs, info, HelixObservation.from_numpy(obs)

    def typed_step(
        self, action: np.ndarray | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], "HelixObservation", "HelixReward"]:
        """Step the environment and return typed models alongside numpy.

        Returns:
            (obs_numpy, reward, terminated, truncated, info, obs_typed, reward_typed)
        """
        from helixdesk.models import HelixObservation, HelixReward

        obs, reward, terminated, truncated, info = self.step(action)
        obs_typed = HelixObservation.from_numpy(obs)
        reward_typed = HelixReward.from_info(reward, info)
        return obs, reward, terminated, truncated, info, obs_typed, reward_typed

