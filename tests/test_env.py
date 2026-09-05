import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from cli_hsr.env import HSREnv, OBS_DIM, MAX_ACTIONS  # noqa: E402


def make_env():
    return HSREnv(
        team_a=["seele", "march_7th"],
        team_b=["blaze_out_of_space", "automaton_grizzly"],
        level=80, max_av=2000, max_rounds=50, seed=0,
    )


def test_spaces_and_reset():
    env = make_env()
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.n == MAX_ACTIONS
    assert "snapshot" in info


def test_masked_random_rollout_completes():
    env = make_env()
    obs, _ = env.reset(seed=2)
    done = False
    steps = 0
    total_reward = 0.0
    while not done and steps < 500:
        mask = env.action_mask
        legal = np.flatnonzero(mask)
        action = int(env.rng.choice(legal)) if len(legal) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    assert done, "episode should finish within 500 steps"
    assert info["winner"] in ("A", "B", "draw")


def test_all_characters_vs_enemies_episode():
    env = HSREnv(
        team_a=["seele", "bronya", "sparkle", "fu_xuan"],
        team_b=["kafka", "black_swan", "luocha", "himeko"],
        level=80, max_av=3000, max_rounds=80, seed=3,
    )
    obs, _ = env.reset(seed=4)
    done, steps = False, 0
    while not done and steps < 1000:
        legal = np.flatnonzero(env.action_mask)
        action = int(env.rng.choice(legal)) if len(legal) else 0
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
    assert done
