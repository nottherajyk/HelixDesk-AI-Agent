"""
HelixDesk OpenEnv — GPT-4o Baseline.

Uses the OpenAI Python client to run GPT-4o as an agent against all 3 tasks.
Reads OPENAI_API_KEY from environment variables.

Usage:
  export OPENAI_API_KEY=sk-...
  python baseline.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from rich.console import Console
from rich.table import Table

from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent
from tasks import easy_classify, medium_sla, hard_trend


# ---------------------------------------------------------------------------
# GPT-4o Agent
# ---------------------------------------------------------------------------

class GPT4oAgent:
    """Agent that calls GPT-4o for each action decision."""

    SYSTEM_PROMPT = """You are HelixDesk AI, a customer support triage agent.

For each email observation you receive, respond with a JSON object:
{
  "classification": 0|1|2,  // 0=query, 1=complaint, 2=flag_for_review
  "priority": 0|1|2|3,      // 0=critical, 1=high, 2=medium, 3=normal
  "assignment": 0|1|2|3|4|5, // 0-4=employee, 5=no assignment
  "secondary": 0|1|2         // 0=auto_reply_from_kb, 1=alert_gm, 2=none
}

Decision rules:
- If keyword flag (obs[1]>0.5): classify=1, priority=0 (complaint, critical)
- High sentiment (obs[0]>0.7) without keyword: classify=1, priority=1
- Enterprise tier (obs[2]>0.5): assign to least loaded employee, priority <= 1
- For queries (low sentiment, no keyword): classify=0, secondary=0 (auto reply)
- If trend growth rates are high (obs[29-36]), consider secondary=1 (alert GM)
- Assign to the employee with lowest load (obs[15,17,19,21,23])

Respond ONLY with the JSON object, no other text."""

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai>=1.0.0")
            sys.exit(1)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("Set it with: set OPENAI_API_KEY=sk-...")
            sys.exit(1)

        self._client = OpenAI(api_key=api_key)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Call GPT-4o to decide the action given the observation vector."""
        obs_summary = (
            f"Observation vector (42 dims):\n"
            f"  sentiment={obs[0]:.2f}, keyword_flag={obs[1]:.1f}\n"
            f"  tier=[enterprise={obs[2]:.0f}, standard={obs[3]:.0f}, free={obs[4]:.0f}]\n"
            f"  category_encoding={obs[5:10].tolist()}\n"
            f"  queue=[crit={obs[10]:.2f}, high={obs[11]:.2f}, med={obs[12]:.2f}, norm={obs[13]:.2f}, review={obs[14]:.2f}]\n"
            f"  employee_loads=[{obs[15]:.2f}, {obs[17]:.2f}, {obs[19]:.2f}, {obs[21]:.2f}, {obs[23]:.2f}]\n"
            f"  sla=[overdue={obs[25]:.2f}, near_deadline={obs[26]:.2f}, pressure={obs[27]:.2f}]\n"
            f"  trend_growth={obs[29:37].tolist()}\n"
            f"  time=[hour={obs[37]:.2f}, day={obs[38]:.2f}]\n"
            f"  progress=[remaining={obs[39]:.2f}, ep_reward={obs[40]:.2f}]"
        )

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": obs_summary},
                ],
                temperature=0.0,
                max_tokens=100,
            )

            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            data = json.loads(text)
            return np.array([
                int(data.get("classification", 1)),
                int(data.get("priority", 2)),
                int(data.get("assignment", 5)),
                int(data.get("secondary", 2)),
            ], dtype=np.int64)

        except Exception as e:
            # Fallback to safe defaults on any API/parse error
            return np.array([1, 2, 0, 2], dtype=np.int64)

    def reset(self):
        """No state to reset."""
        pass


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def run_baseline():
    """Run all 3 tasks with random, rule, and GPT-4o agents; print results."""
    console = Console()
    env = HelixDeskEnv()

    # Build agents
    agents: dict[str, object] = {
        "random": RandomAgent(env.observation_space, env.action_space),
        "rule": RuleAgent(env.observation_space, env.action_space),
    }

    # Only add GPT-4o if API key is set
    has_gpt = bool(os.environ.get("OPENAI_API_KEY"))
    if has_gpt:
        agents["gpt-4o"] = GPT4oAgent()
    else:
        console.print("[yellow]OPENAI_API_KEY not set — skipping GPT-4o baseline[/yellow]")

    tasks = {
        "easy_classify": easy_classify,
        "medium_sla": medium_sla,
        "hard_trend": hard_trend,
    }

    # Results matrix
    results: dict[str, dict[str, float]] = {}

    for agent_name, agent in agents.items():
        results[agent_name] = {}
        for task_name, task_module in tasks.items():
            console.print(f"  Running [cyan]{task_name}[/cyan] with [green]{agent_name}[/green]...", end=" ")
            score = task_module.grade(env=None, agent=agent)
            results[agent_name][task_name] = score
            console.print(f"[bold]{score:.3f}[/bold]")

    # Build results table
    table = Table(title="HelixDesk Baseline Scores", border_style="bright_green")
    table.add_column("Agent", style="cyan", width=12)
    table.add_column("easy_classify", justify="right")
    table.add_column("medium_sla", justify="right")
    table.add_column("hard_trend", justify="right")

    for agent_name, scores in results.items():
        table.add_row(
            agent_name,
            f"{scores['easy_classify']:.3f}",
            f"{scores['medium_sla']:.3f}",
            f"{scores['hard_trend']:.3f}",
        )

    console.print()
    console.print(table)
    env.close()


if __name__ == "__main__":
    run_baseline()
