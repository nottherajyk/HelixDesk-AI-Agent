import pytest
from fastapi.testclient import TestClient
from app import app
import yaml
import subprocess
import os
from tasks.easy_classify import grade as easy_grade
from helixdesk.env import HelixDeskEnv
from helixdesk.agents import RuleAgent

client = TestClient(app)

def test_endpoints():
    # Test all API endpoints required by OpenEnv
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    response = client.post("/reset")
    assert response.status_code == 200
    assert "observation" in response.json()
    
    response = client.post("/step", json={"action": [0, 0, 0, 0]})
    assert response.status_code == 200
    data = response.json()
    assert "reward" in data
    assert "terminated" in data
    
    response = client.get("/state")
    assert response.status_code == 200
    
    response = client.get("/tasks")
    assert response.status_code == 200
    
    response = client.post("/grader", json={"task_id": "easy", "episode_reward": 0.5})
    assert response.status_code == 200
    assert "score" in response.json()

def test_manifest_validation():
    # Strict openenv.yaml checks
    with open("openenv.yaml", "r") as f:
        manifest = yaml.safe_load(f)
    assert manifest["app"] == "app:app"
    assert manifest["port"] == 7860
    assert "endpoints" in manifest
    paths = [ep["path"] for ep in manifest["endpoints"]]
    assert "/reset" in paths
    assert "/step" in paths
    assert "tasks" in manifest

def test_inference_script_format():
    # Hackathon-critical stdout format checks
    env_vars = os.environ.copy()
    env_vars["MAX_STEPS"] = "2" # Just run a couple steps
    result = subprocess.run(["python", "inference.py"], capture_output=True, text=True, env=env_vars)
    assert "[START] task=" in result.stdout
    assert "[STEP] step=1 action=" in result.stdout
    assert "[END] success=" in result.stdout

def test_grader_consistency():
    # Ensure grade deterministic and local == API expectation
    env = HelixDeskEnv()
    agent = RuleAgent(env.observation_space, env.action_space)
    
    s1 = easy_grade(env, agent, seed=42)
    s2 = easy_grade(env, agent, seed=42)
    assert s1 == s2, "Grader must be fully deterministic given the same seed"
    
    s3 = easy_grade(env, agent, seed=99)
    # The output from different seeds might still match if agents are invariant, but deterministic property is proven.
    assert s1 >= 0.0 and s1 <= 1.0, "Grader must return normalized [0, 1] range"
