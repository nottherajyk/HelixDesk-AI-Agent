"""Pydantic-typed wrappers for HelixDesk observations, actions, and rewards."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Observation model — mirrors the 42-dim vector with named fields
# ---------------------------------------------------------------------------

class EmailFeatures(BaseModel):
    """Dims 0–9: current email features."""
    sentiment_intensity: float = Field(ge=-1.0, le=1.0)
    has_keyword_flag: float = Field(ge=0.0, le=1.0)
    customer_tier_enterprise: float = Field(ge=0.0, le=1.0)
    customer_tier_standard: float = Field(ge=0.0, le=1.0)
    customer_tier_free: float = Field(ge=0.0, le=1.0)
    category_0: float = Field(ge=0.0, le=1.0)
    category_1: float = Field(ge=0.0, le=1.0)
    category_2: float = Field(ge=0.0, le=1.0)
    category_3: float = Field(ge=0.0, le=1.0)
    category_4_overflow: float = Field(ge=0.0, le=1.0)


class QueueState(BaseModel):
    """Dims 10–14: queue priority counts (normalized)."""
    critical_count_norm: float = Field(ge=-1.0, le=1.0)
    high_count_norm: float = Field(ge=-1.0, le=1.0)
    medium_count_norm: float = Field(ge=-1.0, le=1.0)
    normal_count_norm: float = Field(ge=-1.0, le=1.0)
    review_queue_count_norm: float = Field(ge=-1.0, le=1.0)


class EmployeeState(BaseModel):
    """Two dims per employee: load + avg resolve time (normalized)."""
    load_norm: float = Field(ge=-1.0, le=1.0)
    avg_resolve_norm: float = Field(ge=-1.0, le=1.0)


class SLAState(BaseModel):
    """Dims 25–28: SLA pressure metrics."""
    overdue_count_norm: float = Field(ge=-1.0, le=1.0)
    near_deadline_count_norm: float = Field(ge=-1.0, le=1.0)
    sla_pressure: float = Field(ge=-1.0, le=1.0)
    critical_overdue_flag: float = Field(ge=0.0, le=1.0)


class TrendState(BaseModel):
    """Dims 29–36: per-category growth rates."""
    growth_rates: list[float] = Field(min_length=8, max_length=8)


class TimeState(BaseModel):
    """Dims 37–38: normalized time of day and day of week."""
    hour_of_day_norm: float = Field(ge=0.0, le=1.0)
    day_of_week_norm: float = Field(ge=0.0, le=1.0)


class EpisodeProgress(BaseModel):
    """Dims 39–41: episode-level progress indicators."""
    steps_remaining_norm: float = Field(ge=-1.0, le=1.0)
    episode_reward_norm: float = Field(ge=-1.0, le=1.0)
    agent_confidence: float = Field(ge=-1.0, le=1.0)


class HelixObservation(BaseModel):
    """Full typed observation — mirrors the 42-dim float32 vector."""
    email: EmailFeatures
    queue: QueueState
    employees: list[EmployeeState] = Field(min_length=5, max_length=5)
    sla: SLAState
    trends: TrendState
    time: TimeState
    progress: EpisodeProgress

    @classmethod
    def from_numpy(cls, obs: np.ndarray) -> "HelixObservation":
        """Parse a 42-dim numpy array into a typed HelixObservation."""
        assert obs.shape == (42,), f"Expected (42,), got {obs.shape}"
        o = obs.tolist()
        return cls(
            email=EmailFeatures(
                sentiment_intensity=o[0], has_keyword_flag=o[1],
                customer_tier_enterprise=o[2], customer_tier_standard=o[3],
                customer_tier_free=o[4],
                category_0=o[5], category_1=o[6], category_2=o[7],
                category_3=o[8], category_4_overflow=o[9],
            ),
            queue=QueueState(
                critical_count_norm=o[10], high_count_norm=o[11],
                medium_count_norm=o[12], normal_count_norm=o[13],
                review_queue_count_norm=o[14],
            ),
            employees=[
                EmployeeState(load_norm=o[15 + i * 2], avg_resolve_norm=o[16 + i * 2])
                for i in range(5)
            ],
            sla=SLAState(
                overdue_count_norm=o[25], near_deadline_count_norm=o[26],
                sla_pressure=o[27], critical_overdue_flag=o[28],
            ),
            trends=TrendState(growth_rates=o[29:37]),
            time=TimeState(hour_of_day_norm=o[37], day_of_week_norm=o[38]),
            progress=EpisodeProgress(
                steps_remaining_norm=o[39], episode_reward_norm=o[40],
                agent_confidence=o[41],
            ),
        )


# ---------------------------------------------------------------------------
# Action model — mirrors MultiDiscrete([3, 4, 6, 3])
# ---------------------------------------------------------------------------

class HelixAction(BaseModel):
    """Typed representation of an agent action."""
    classification: Literal["query", "complaint", "flag_for_review"]
    priority: Literal["critical", "high", "medium", "normal"]
    assignment: int = Field(ge=0, le=5, description="0-4 = employee, 5 = no assignment")
    secondary: Literal["auto_reply_from_kb", "alert_gm", "none"]

    _CLASSIFY_MAP = {"query": 0, "complaint": 1, "flag_for_review": 2}
    _PRIORITY_MAP = {"critical": 0, "high": 1, "medium": 2, "normal": 3}
    _SECONDARY_MAP = {"auto_reply_from_kb": 0, "alert_gm": 1, "none": 2}

    _CLASSIFY_INV = {0: "query", 1: "complaint", 2: "flag_for_review"}
    _PRIORITY_INV = {0: "critical", 1: "high", 2: "medium", 3: "normal"}
    _SECONDARY_INV = {0: "auto_reply_from_kb", 1: "alert_gm", 2: "none"}

    def to_numpy(self) -> np.ndarray:
        """Convert to a 4-element numpy action array."""
        return np.array([
            self._CLASSIFY_MAP[self.classification],
            self._PRIORITY_MAP[self.priority],
            self.assignment,
            self._SECONDARY_MAP[self.secondary],
        ], dtype=np.int64)

    @classmethod
    def from_numpy(cls, action: np.ndarray) -> "HelixAction":
        """Parse a 4-element numpy array into a typed HelixAction."""
        a = action.tolist()
        return cls(
            classification=cls._CLASSIFY_INV[a[0]],
            priority=cls._PRIORITY_INV[a[1]],
            assignment=a[2],
            secondary=cls._SECONDARY_INV[a[3]],
        )


# ---------------------------------------------------------------------------
# Reward model — structured breakdown of reward signals
# ---------------------------------------------------------------------------

class RewardSignal(BaseModel):
    """A single reward component."""
    signal_type: str
    value: float
    ticket_id: str | None = None
    details: str = ""


class HelixReward(BaseModel):
    """Full reward breakdown for one step."""
    total: float = Field(ge=-1.0, le=1.0)
    signals: list[RewardSignal]

    @classmethod
    def from_info(cls, total: float, info: dict[str, Any]) -> "HelixReward":
        """Build from step reward and info dict."""
        breakdown = info.get("reward_breakdown", {})
        signals = [
            RewardSignal(signal_type=k, value=v)
            for k, v in breakdown.items()
        ]
        return cls(total=total, signals=signals)
