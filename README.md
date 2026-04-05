---
title: HelixDesk OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - gymnasium
  - customer-support
---

# HelixDesk OpenEnv

**HelixDesk OpenEnv** is a complete, real-world Gymnasium-style reinforcement learning environment where an AI agent named HelixDesk learns to manage customer email queues by interacting with a realistic simulation of a company's complaint management system. It is fully compatible with standard RL libraries including Stable-Baselines3, RLlib, and CleanRL.

---

## The RL Problem

**State**: A 42-dimensional observation vector encoding the current email's features (sentiment, category, customer tier, keyword flags), the support queue state (priority counts, overdue tickets), team workload (5 employees' loads and resolve times), SLA pressure, complaint volume trends, simulated time, and episode progress.

**Action**: A 4-part decision for each incoming email—classification (query / complaint / flag for review), priority assignment (critical / high / medium / normal), employee assignment (5 employees or none), and a secondary action (auto-reply from KB / alert GM / none).

**Reward**: A composite signal from 12 distinct components: correct classification (+0.5), timely resolution (+1.0), high CSAT (+0.8), trend prevention (+0.6), workload balance (+0.4), KB updates (+0.3), and penalties for missed deadlines (−1.0), bad auto-replies (−0.8), unnecessary escalations (−0.6), misclassification (−0.5), reopened complaints (−0.4), and missed keyword flags (−0.3). Total reward is clipped to [−1.0, +1.0] per step.

---

## Setup

```bash
cd helixdesk-openenv
pip install -r requirements.txt
```

---

## Quick Start

### Run rule-based agent (no learning)
```bash
python train.py --agent rule --episodes 100
```

### Run random baseline
```bash
python train.py --agent random --episodes 100
```

### Train with Stable-Baselines3 PPO
```bash
pip install stable-baselines3
python train.py --agent sb3 --episodes 500
```

### Evaluate an agent
```bash
python evaluate.py --agent rule --episodes 100
```

---

## Gymnasium Compatibility

HelixDeskEnv passes `gymnasium.utils.env_checker.check_env()` with 0 errors:

```python
from gymnasium.utils.env_checker import check_env
from helixdesk import HelixDeskEnv

env = HelixDeskEnv()
check_env(env)  # ✓ passes
```

Compatible with any Gymnasium-based training library:
- **Stable-Baselines3**: `PPO("MlpPolicy", HelixDeskEnv())`
- **CleanRL**: use the env like any standard gymnasium env
- **RLlib**: register with `gymnasium.register()`

---

## State Space (42 dimensions)

| Group | Dims | Description |
|---|---|---|
| **Current Email** | 0–9 | Sentiment, keyword flag, customer tier (3-hot), category (5-slot overflow encoding) |
| **Queue State** | 10–14 | Normalized counts: critical, high, medium, normal, pending review |
| **Team State** | 15–24 | 5 employees × (load_norm, avg_resolve_norm) |
| **SLA State** | 25–28 | Overdue norm, near-deadline norm, SLA pressure, critical overdue flag |
| **Trend State** | 29–36 | 8 categories × growth rate fraction [−1, 1] |
| **Time State** | 37–38 | Hour of day / 24, day of week / 7 |
| **Episode Progress** | 39–41 | Steps remaining norm, episode reward norm, agent confidence |

All values normalized to `[-1.0, 1.0]`. Observation space: `Box(low=-1, high=1, shape=(42,), dtype=float32)`.

---

## Action Space (MultiDiscrete[3, 4, 6, 3])

| Dim | Choices | Description |
|---|---|---|
| 0: Classification | 0=query, 1=complaint, 2=flag_for_review | How to classify the current email |
| 1: Priority | 0=critical, 1=high, 2=medium, 3=normal | Priority level (complaints only) |
| 2: Assignment | 0–4=employee_0..4, 5=no_assignment | Who handles it (complaints only) |
| 3: Secondary | 0=auto_reply_from_kb, 1=alert_gm, 2=none | Additional action |

**Rule**: If classification = `flag_for_review`, dims 1/2/3 are forced to `(normal, no_assignment, none)`.

---

## Reward Signals (12 components)

| Signal | Value | Condition |
|---|---|---|
| `resolve_on_time` | +1.0 | Employee resolves ticket within SLA |
| `csat_high` | +0.8 | CSAT score ≥ 4 on resolved ticket |
| `trend_prevented` | +0.6 | GM alerted during category surge |
| `correct_classification` | +0.5 | Classification matches ground truth |
| `balanced_assignment` | +0.4 | Workload std decreased |
| `kb_updated` | +0.3 | Knowledge base learned new entry |
| `missed_deadline` | −1.0 | Ticket missed SLA deadline |
| `bad_autoreply` | −0.8 | CSAT score ≤ 2 |
| `unnecessary_escalation` | −0.6 | Flagged for review despite low complexity |
| `misclassification` | −0.5 | Classification doesn't match ground truth |
| `complaint_reopened` | −0.4 | Complaint reopened after resolution |
| `keyword_flag_missed` | −0.3 | Keyword-flagged email not treated as complaint/critical |

---

## How to Extend

### Add an employee
1. In `config.yaml`, set `env.n_employees: 6`
2. The observation space grows by 2 dims (new employee load + resolve time)
3. Update `spaces.py` accordingly (increase `OBS_SIZE` and add employee dims)
4. Update action space dim 2 to `7` (6 employees + no_assignment)

### Add a category
1. Add the category name to `email_gen.categories` in `config.yaml`
2. Add 5 query + 5 complaint templates in `email_gen.py`
3. Add 3 KB entries in `knowledge_base.py`
4. Trend state dims grow by 1

### Tune parameters
All parameters in `config.yaml` propagate through without code changes:
- Adjust `episode_emails` for longer/shorter episodes
- Modify reward weights to shape different agent behaviours
- Change `sla.*_hours` to tighten or relax deadlines
- Adjust `employee_sim.base_resolve_rate` for harder/easier simulation

---

## Project Structure

```
helixdesk-openenv/
├── helixdesk/
│   ├── __init__.py          # exports HelixDeskEnv
│   ├── env.py               # main environment class
│   ├── models.py            # Pydantic typed wrappers (HelixObservation, HelixAction, HelixReward)
│   ├── spaces.py            # observation & action space definitions
│   ├── rewards.py           # reward function
│   ├── simulator/           # simulation components
│   │   ├── clock.py         # simulated time
│   │   ├── email_gen.py     # synthetic email generation
│   │   ├── employee_sim.py  # employee behaviour model
│   │   ├── knowledge_base.py # KB lookup & auto-learn
│   │   └── trend_watchdog.py # volume surge detection
│   ├── agents/              # baseline agents
│   │   ├── base_agent.py    # abstract agent interface
│   │   ├── random_agent.py  # random baseline
│   │   └── rule_agent.py    # deterministic rule-based agent
│   └── monitor/             # logging & visualization
│       ├── episode_logger.py    # CSV per-step logger
│       └── terminal_dashboard.py # Rich live dashboard
├── tasks/                   # graded task definitions
│   ├── easy_classify.py     # keyword-flag classification (easy)
│   ├── medium_sla.py        # SLA compliance rate (medium)
│   ├── hard_trend.py        # trend detection + CSAT (hard)
│   └── expert_full.py       # full expert evaluation (expert)
├── tests/                   # pytest test suite
├── train.py                 # training entry point
├── evaluate.py              # evaluation with rich table output
├── baseline.py              # GPT-4o + rule + random baseline runner
├── inference.py             # mandatory hackathon inference script
├── config.yaml              # all configurable parameters
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # container image
├── requirements.txt         # Python dependencies
└── README.md                # this file
```

---

## Tasks

HelixDesk OpenEnv ships with 4 graded tasks of increasing difficulty. Each task's `grade(env, agent)` function returns a score in `[0.0, 1.0]`.

| Task | Difficulty | Scoring Criteria |
|---|---|---|
| `easy_classify` | 🟢 Easy | Run 20 emails. Score = fraction of keyword-flagged emails correctly classified as **complaint** with **critical** priority. |
| `medium_sla` | 🟡 Medium | Run 1 full episode (100 emails). Score = fraction of tickets resolved **within SLA deadline**. |
| `hard_trend` | 🔴 Hard | Run 1 full episode. Score = avg of (trend alerts caught / surge events, CSAT / 4.5, overdue control). |
| `expert_full` | ⚫ Expert | Geometric mean of keyword score × classification accuracy × review abuse rate. One weakness tanks the whole score. |

```bash
# Run all tasks against rule + random baselines
python baseline.py
```

---

## Baseline Scores

Scores are exact and reproducible with seed=42:

| Agent  | easy_classify | medium_sla | hard_trend | expert_full |
|--------|--------------|------------|------------|-------------|
| random | 0.000        | 0.337      | 0.415      | 0.000       |
| rule   | 1.000        | 0.882      | 0.652      | 0.935       |

Run `python baseline.py` to reproduce exactly.

---

## Docker

```bash
# Build
docker build -t helixdesk-openenv .

# Run rule agent for 10 episodes
docker run --rm helixdesk-openenv

# Run with custom args
docker run --rm helixdesk-openenv python train.py --agent random --episodes 50

# Run baseline (requires API key)
docker run --rm -e OPENAI_API_KEY=sk-... helixdesk-openenv python baseline.py
```

---

## OpenEnv Compliance

This environment follows the [OpenEnv](https://openenv.org) specification:

- **`openenv.yaml`** — declares environment name, version, entry point, and task IDs. Schema validated manually against the OpenEnv spec.
- **Typed models** — `helixdesk/models.py` defines `HelixObservation`, `HelixAction`, `HelixReward` as Pydantic models.
- **4 graded tasks** — `tasks/easy_classify.py`, `tasks/medium_sla.py`, `tasks/hard_trend.py`, `tasks/expert_full.py` each export `grade(env, agent) -> float` in `[0.0, 1.0]`.
- **Gymnasium compatible** — passes `gymnasium.utils.env_checker.check_env()` with 0 errors.
- **Docker** — `docker build -t helixdesk-openenv . && docker run --rm helixdesk-openenv` starts cleanly.
- **Baseline** — `python baseline.py` reproduces exact scores from the table above using seed=42.
```bash
# Validate manually
python -c "from helixdesk import HelixDeskEnv; from gymnasium.utils.env_checker import check_env; check_env(HelixDeskEnv())"
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## HuggingFace Space

Live demo: https://huggingface.co/spaces/nottherajyk/helixdesk-openenv

The Space runs the rule-based and random agents interactively in your browser.
No install required.

---

## License

MIT
