"""
HelixDesk OpenEnv — inference.py
Mandatory hackathon inference script.
Emits [START], [STEP], [END] logs to stdout.
"""

import os
import json
import numpy as np
from typing import List, Optional

from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("HELIXDESK_TASK", "medium_sla")
BENCHMARK = "helixdesk-openenv"
MAX_STEPS = 100
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are HelixDesk, an AI agent managing a customer support email queue.
Each step you receive a summary of the current state and must output a JSON action.

The action has 4 fields:
- classify: 0=query, 1=complaint, 2=flag_for_review
- priority: 0=critical, 1=high, 2=medium, 3=normal
- assign: 0-4=employee index, 5=no_assignment
- secondary: 0=auto_reply_from_kb, 1=alert_gm, 2=none

Key rules:
- keyword_flag=1.0: always classify=1, priority=0
- sentiment > 0.85: classify=1, priority=1
- enterprise tier: classify=1, priority=1
- Assign to least loaded employee
- trend growth > 0.5: secondary=1 (alert_gm)

Respond ONLY with valid JSON: {"classify": 1, "priority": 0, "assign": 2, "secondary": 1}"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_llm_action(client, obs: np.ndarray, step: int) -> np.ndarray:
    obs_summary = {
        "sentiment": round(float(obs[0]), 2),
        "keyword_flag": float(obs[1]),
        "enterprise": float(obs[2]),
        "critical_queue": round(float(obs[10]), 2),
        "overdue": round(float(obs[25]), 2),
        "employee_loads": [round(float(obs[15 + i*2]), 2) for i in range(5)],
        "max_trend": round(float(max(obs[29:37])), 2),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Step {step}. State: {json.dumps(obs_summary)}. Output JSON action."},
            ],
            temperature=0.1,
            max_tokens=60,
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(text)
        return np.array([
            int(parsed.get("classify", 1)),
            int(parsed.get("priority", 2)),
            int(parsed.get("assign", 0)),
            int(parsed.get("secondary", 2)),
        ], dtype=np.int64)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Rule-based fallback
        if obs[1] == 1.0:
            return np.array([1, 0, 0, 1], dtype=np.int64)
        elif obs[0] > 0.7:
            return np.array([1, 1, int(np.argmin([obs[15+i*2] for i in range(5)])), 2], dtype=np.int64)
        else:
            return np.array([0, 3, 5, 0], dtype=np.int64)


def get_task_score(task_name: str, agent) -> float:
    env = HelixDeskEnv()
    if task_name == "easy_classify":
        from tasks.easy_classify import grade
    elif task_name == "medium_sla":
        from tasks.medium_sla import grade
    elif task_name == "hard_trend":
        from tasks.hard_trend import grade
    elif task_name == "expert_full":
        from tasks.expert_full import grade
    else:
        raise ValueError(f"Unknown task: {task_name}")
    return grade(env, agent)


def main() -> None:
    env = HelixDeskEnv()
    obs, info = env.reset(seed=42)

    use_llm = bool(API_KEY)
    if use_llm:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Build agent for task grading
    if use_llm:
        class LLMAgent:
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space
                self._step = 0
            def reset(self):
                self._step = 0
            def act(self, obs):
                self._step += 1
                return get_llm_action(client, obs, self._step)
        agent = LLMAgent(env.observation_space, env.action_space)
    else:
        agent = RuleAgent(env.observation_space, env.action_space)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        done = False
        step = 0
        while not done and step < MAX_STEPS:
            step += 1
            if use_llm:
                action = get_llm_action(client, obs, step)
            else:
                action = agent.act(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_str = f"[{action[0]},{action[1]},{action[2]},{action[3]}]"
            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=action_str, reward=float(reward), done=done, error=None)

        # Get final task score
        score = get_task_score(TASK_NAME, agent)
        score = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
