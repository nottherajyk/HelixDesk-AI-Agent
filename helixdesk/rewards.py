"""RewardFunction — computes multi-component rewards for HelixDesk agent actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helixdesk.simulator.email_gen import EmailEvent
from helixdesk.simulator.employee_sim import TickResolutionEvent


@dataclass
class RewardEvent:
    """A single reward component that fired during a step."""
    event_type: str
    value: float
    ticket_id: str | None
    details: str


class RewardFunction:
    """Computes the composite reward signal for each environment step.

    The reward is a weighted sum of up to 12 distinct signals, each
    configurable in config.yaml. The total is clipped to [-1.0, +1.0].
    """

    def __init__(self, config: dict):
        self.cfg = config["rewards"]

    def compute(
        self,
        action: np.ndarray,
        email: EmailEvent,
        resolution_events: list[TickResolutionEvent],
        trend_alerts: list[str],
        queue_state: dict,
        kb_updated: bool,
        employee_loads: list[int],
        prev_employee_loads: list[int],
    ) -> tuple[float, list[RewardEvent]]:
        """Compute total reward for this step.

        Args:
            action: The 4-element action array [classify, priority, assign, secondary].
            email: The current email being processed.
            resolution_events: Outcomes from EmployeeSimulator.tick().
            trend_alerts: Categories with active surge alerts.
            queue_state: Dict with queue metrics (unused for now, reserved).
            kb_updated: Whether the KB was updated this step.
            employee_loads: Current employee load counts.
            prev_employee_loads: Employee loads from previous step.

        Returns:
            (total_reward, list_of_reward_events). Total is clipped to [-1, 1].
        """
        events: list[RewardEvent] = []

        # 1. Correct classification vs ground truth
        classification_map = {0: "query", 1: "complaint", 2: "pending_review"}
        predicted_type = classification_map[int(action[0])]

        if predicted_type == email.ticket_type:
            events.append(RewardEvent(
                "correct_classification",
                self.cfg["correct_priority"],
                email.email_id,
                f"Correctly classified as {predicted_type}",
            ))
        elif predicted_type == "pending_review":
            # Flag for review is not exactly wrong, but not rewarded for classification
            pass
        else:
            events.append(RewardEvent(
                "misclassification",
                self.cfg["misclassification"],
                email.email_id,
                f"Predicted {predicted_type}, actual {email.ticket_type}",
            ))

        # 2. Keyword flag check
        if email.has_keyword_flag and predicted_type != "complaint":
            events.append(RewardEvent(
                "keyword_flag_missed",
                self.cfg["keyword_flag_missed"],
                email.email_id,
                "Failed to classify keyword-flagged email as complaint",
            ))

        if email.has_keyword_flag and int(action[1]) != 0:  # not critical priority
            events.append(RewardEvent(
                "keyword_not_critical",
                self.cfg["keyword_flag_missed"],
                email.email_id,
                "Keyword-flagged email not assigned critical priority",
            ))

        # 3. Resolution outcomes (from EmployeeSimulator tick results)
        for ev in resolution_events:
            if ev.resolved:
                events.append(RewardEvent(
                    "resolve_on_time",
                    self.cfg["resolve_on_time"],
                    ev.ticket_id,
                    "Ticket resolved within SLA",
                ))
                if ev.csat_score is not None and ev.csat_score >= 4:
                    events.append(RewardEvent(
                        "csat_high",
                        self.cfg["csat_high"],
                        ev.ticket_id,
                        f"High CSAT score: {ev.csat_score}",
                    ))
                elif ev.csat_score is not None and ev.csat_score <= 2:
                    events.append(RewardEvent(
                        "bad_autoreply",
                        self.cfg["bad_autoreply"],
                        ev.ticket_id,
                        f"Low CSAT score: {ev.csat_score}",
                    ))
            else:
                events.append(RewardEvent(
                    "missed_deadline",
                    self.cfg["missed_deadline"],
                    ev.ticket_id,
                    "Ticket missed SLA deadline",
                ))

        # 4. Trend alerts: reward agent for alerting during surges
        if int(action[3]) == 1:  # alert_gm action
            for category in trend_alerts:
                events.append(RewardEvent(
                    "trend_prevented",
                    self.cfg["trend_prevented"],
                    None,
                    f"GM alerted during {category} surge",
                ))

        # 5. Workload balance
        if len(employee_loads) > 0 and len(prev_employee_loads) > 0:
            std_now = float(np.std(employee_loads))
            std_prev = float(np.std(prev_employee_loads))
            if std_now < std_prev:
                events.append(RewardEvent(
                    "balanced_assignment",
                    self.cfg["balanced_assignment"],
                    None,
                    f"Workload std improved: {std_prev:.2f} → {std_now:.2f}",
                ))

        # 6. KB updated
        if kb_updated:
            events.append(RewardEvent(
                "kb_updated",
                self.cfg["kb_updated"],
                email.email_id,
                "Knowledge base updated from resolved query",
            ))

        # 7. Unnecessary escalation
        if int(action[0]) == 2:  # flag_for_human_review
            # If the email is clearly classifiable, this is wasteful
            if email.sentiment_intensity < 0.5 and not email.has_keyword_flag:
                events.append(RewardEvent(
                    "unnecessary_escalation",
                    self.cfg["unnecessary_escalation"],
                    email.email_id,
                    "Flagged for review despite low complexity",
                ))

        # 8. Agentic behavior: Conflicting Signals / Ambiguity Resolution
        is_ambiguous = (email.sentiment_intensity > 0.6 and not email.has_keyword_flag) or \
                       (email.has_keyword_flag and email.sentiment_intensity > 0.8)
        
        if is_ambiguous:
            if int(action[0]) == 2:  # safely flagged for review
                events.append(RewardEvent(
                    "ambiguity_resolved",
                    0.5,
                    email.email_id,
                    "Successfully flagged ambiguous/conflicting email for review."
                ))
            else:
                events.append(RewardEvent(
                    "acted_on_ambiguity",
                    -0.5,
                    email.email_id,
                    "Agent acted on conflicting signals without escalating."
                ))

        # 9. Escalation tradeoff / Delayed consequences risk
        if int(action[1]) == 0 and int(action[0]) == 1: # critical complaint
            if int(action[2]) != 0: # not assigned to senior (emp 0)
                events.append(RewardEvent(
                    "risky_assignment",
                    -0.3,
                    email.email_id,
                    "Critical complaint assigned to junior employee."
                ))
            if int(action[3]) != 1: # didn't alert GM
                events.append(RewardEvent(
                    "gm_not_alerted",
                    -0.2,
                    email.email_id,
                    "Failed to alert GM for critical complaint."
                ))

        # Sum and clip
        total = sum(e.value for e in events)
        total = float(np.clip(total, -1.0, 1.0))

        return total, events
