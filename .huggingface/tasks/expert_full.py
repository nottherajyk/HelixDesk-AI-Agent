import numpy as np


def grade(env, agent) -> float:
    """
    Expert task — scores only on metrics where agent behavior
    actually makes a difference:
      1. keyword_score  — must classify ALL keyword emails as complaint+critical
      2. classify_score — overall classification accuracy >= 95%
      3. no_review_abuse — flag_for_review used <= 5% of time
      4. load_balance   — std dev of employee loads kept low (< 2.0 avg)
      5. episode_reward — normalized total episode reward >= 0.5
    
    Uses weakest-link: min_score * 0.60 + avg_score * 0.40
    Target: rule <= 0.55, random <= 0.10
    """
    obs, info = env.reset(seed=42)
    agent.reset()

    keyword_flagged_total = 0
    keyword_missed = 0
    correct_classifications = 0
    total_classifications = 0
    review_flags = 0
    total_steps = 0
    load_stds = []
    episode_reward = 0.0
    done = False

    while not done:
        # Check if current email has a keyword flag BEFORE acting
        has_keyword = obs[1] > 0.5
        if has_keyword:
            keyword_flagged_total += 1

        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        total_steps += 1

        breakdown = info.get("reward_breakdown", [])
        for event in breakdown:
            if event["type"] == "keyword_flag_missed":
                keyword_missed += 1
            if event["type"] == "correct_classification":
                correct_classifications += 1
                total_classifications += 1
            if event["type"] == "misclassification":
                total_classifications += 1

        if action[0] == 2:
            review_flags += 1

        loads = [obs[15 + i*2] for i in range(5)]
        load_stds.append(float(np.std(loads)))

    keyword_score  = 1.0 if keyword_missed == 0 else max(0.0, 1.0 - (keyword_missed / max(keyword_flagged_total, 1)) * 4)
    classify_score = min(correct_classifications / max(total_classifications * 0.95, 1), 1.0)
    review_score   = max(0.0, 1.0 - (review_flags / max(total_steps * 0.05, 1)))

    # All three must be good
    final = (keyword_score * classify_score * review_score) ** 0.5
    return float(min(max(final, 0.0), 1.0))
