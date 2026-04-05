"""
HelixDesk OpenEnv — Standard API Server + Gradio Dashboard.

Required for hackathon submission:
1. Provides a visual Gradio UI at root "/" for manual review.
2. Exposes /api/health, /reset, /step, /state endpoints for automated evaluation.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import gradio as gr
from pydantic import BaseModel
from typing import List, Optional, Any

from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent

# Initialize global env for API
env = HelixDeskEnv()

# --- FastAPI Setup ---
app = FastAPI(title="HelixDesk OpenEnv API")

# Health endpoint — NOT on "/" so Gradio can own the root
@app.get("/api/health")
@app.get("/health")
async def health():
    return {"status": "ok", "env": "HelixDesk OpenEnv", "version": "1.0.0"}

@app.post("/reset")
async def reset():
    obs, info = env.reset()
    return {
        "observation": obs.tolist(),
        "info": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in info.items()}
    }

class StepRequest(BaseModel):
    action: List[int]

@app.post("/step")
async def step(req: StepRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in info.items()}
    }

@app.get("/state")
async def get_state():
    obs = env.state()
    return {"observation": obs.tolist()}

# --- Chart styling ---
plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#dee2e6",
    "axes.labelcolor": "#495057",
    "xtick.color": "#495057",
    "ytick.color": "#495057",
    "text.color": "#212529",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#adb5bd",
    "font.size": 11,
})

# --- Gradio UI Logic ---
def run_episode(agent_type: str) -> tuple:
    """Run a full episode and return chart figures + summary markdown."""
    local_env = HelixDeskEnv()
    if agent_type == "rule":
        agent = RuleAgent(local_env.observation_space, local_env.action_space)
    else:
        agent = RandomAgent(local_env.observation_space, local_env.action_space)

    obs, info = local_env.reset(seed=42)
    agent.reset()

    steps, cumulative_rewards = [], []
    queue_depths, overdue_counts, csat_scores = [], [], []
    ep_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = local_env.step(action)
        ep_reward += reward
        done = terminated or truncated
        step_num = info.get("step", 0)
        steps.append(step_num)
        cumulative_rewards.append(ep_reward)
        queue_depths.append(info.get("queue_depth", 0))
        overdue_counts.append(info.get("overdue_count", 0))
        csat = info.get("csat_score")
        if csat is not None:
            csat_scores.append(float(csat))

    local_env.close()

    # --- Reward chart ---
    fig_reward, ax_reward = plt.subplots(figsize=(7, 3.5))
    ax_reward.plot(steps, cumulative_rewards, color="#4f46e5", linewidth=2)
    ax_reward.fill_between(steps, cumulative_rewards, alpha=0.1, color="#4f46e5")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Cumulative Reward")
    ax_reward.set_title("Cumulative Reward per Step", fontweight="bold")
    fig_reward.tight_layout()

    # --- Queue depth chart ---
    fig_queue, ax_queue = plt.subplots(figsize=(7, 3.5))
    ax_queue.plot(steps, queue_depths, color="#0891b2", linewidth=2)
    ax_queue.fill_between(steps, queue_depths, alpha=0.1, color="#0891b2")
    ax_queue.set_xlabel("Step")
    ax_queue.set_ylabel("Queue Depth")
    ax_queue.set_title("Queue Depth over Time", fontweight="bold")
    fig_queue.tight_layout()

    # --- Summary ---
    avg_csat = np.mean(csat_scores) if csat_scores else 0.0
    final_overdue = overdue_counts[-1] if overdue_counts else 0
    max_overdue = max(overdue_counts) if overdue_counts else 0

    summary = f"""### Episode Summary

| Metric | Value |
|---|---|
| **Agent** | {agent_type} |
| **Total Reward** | {ep_reward:+.2f} |
| **Steps** | {len(steps)} |
| **Final Queue Depth** | {queue_depths[-1] if queue_depths else 0} |
| **Final Overdue** | {final_overdue} |
| **Peak Overdue** | {max_overdue} |
| **Avg CSAT** | {avg_csat:.2f} / 5.0 |
| **CSAT Samples** | {len(csat_scores)} |
"""

    return fig_reward, fig_queue, summary


# ---------------------------------------------------------------------------
# Startup verification
# ---------------------------------------------------------------------------
print("HelixDesk OpenEnv — verifying environment...")
_env = HelixDeskEnv()
_agent = RuleAgent(_env.observation_space, _env.action_space)
_obs, _ = _env.reset(seed=0)
for _ in range(5):
    _action = _agent.act(_obs)
    _obs, _, _terminated, _, _ = _env.step(_action)
    if _terminated:
        break
_env.close()
print("Environment verified ✓")

# ---------------------------------------------------------------------------
# Create Gradio interface
# ---------------------------------------------------------------------------
initial_reward, initial_queue, initial_summary = run_episode("rule")

with gr.Blocks(
    title="HelixDesk OpenEnv",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
) as demo:
    gr.Markdown(
        """
# 📧 HelixDesk OpenEnv

**Gymnasium-compatible RL environment for AI-powered customer email queue management.**

Select an agent and click **Run Episode** to watch it process 100 emails.
The rule-based agent uses 6 deterministic business rules; the random agent samples uniformly.
        """
    )

    with gr.Row():
        agent_dropdown = gr.Dropdown(
            choices=["rule", "random"],
            value="rule",
            label="Agent",
            scale=1,
        )
        run_btn = gr.Button("▶ Run Episode", variant="primary", scale=1)

    with gr.Row():
        reward_plot = gr.Plot(value=initial_reward, label="Reward")
        queue_plot = gr.Plot(value=initial_queue, label="Queue Depth")

    summary_md = gr.Markdown(value=initial_summary)

    run_btn.click(
        fn=run_episode,
        inputs=[agent_dropdown],
        outputs=[reward_plot, queue_plot, summary_md],
    )

# Mount Gradio into FastAPI at root path
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
