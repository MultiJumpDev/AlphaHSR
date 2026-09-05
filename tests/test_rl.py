import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")


from cli_hsr.env import HSREnv, battle_to_observation  # noqa: E402
from cli_hsr.engine import Battle  # noqa: E402
from cli_hsr.registry import (  # noqa: E402
    CheckpointAgent, ModelMetadata, load_model_bundle, save_model_bundle,
)
from cli_hsr.rl.rewards import RewardConfig, RewardShaper  # noqa: E402
from cli_hsr.rl.train import TrainConfig, evaluate_policy, train  # noqa: E402


def test_reward_shaper_scores_events():
    config = RewardConfig(break_reward=0.5, damage_dealt_scale=0.001)
    shaper = RewardShaper(config)
    rows_a = [{"id": "seele", "unit_type": "character"}]
    # minimal fake battle: use real battle from db
    from cli_hsr import db
    battle = Battle([db.get_unit("seele")], [db.get_unit("blaze_out_of_space")],
                    rng=np.random.RandomState(0) if False else None)
    shaper.reset(battle)
    battle.emit("damage", source="Seele", source_side="A", target="Blaze",
                target_side="B", damage=1000.0)
    battle.emit("weakness_break", target="Blaze")
    r = shaper.turn_reward(battle, "A")
    assert r > 0.5  # 1000*0.001 + 0.5
    # clipping guard
    battle.emit("damage", source="Seele", source_side="A", target="Blaze",
                target_side="B", damage=999999.0)
    r2 = shaper.turn_reward(battle, "A")
    assert r2 <= config.clip


def test_env_perspective_swap():
    teams_a = ["seele", "march_7th"]
    teams_b = ["blaze_out_of_space", "automaton_grizzly"]
    env_a = HSREnv(team_a=teams_a, team_b=teams_b, agent_side="A",
                   max_av=1000, max_rounds=20, seed=0)
    env_b = HSREnv(team_a=teams_a, team_b=teams_b, agent_side="B",
                   max_av=1000, max_rounds=20, seed=0)
    obs_a, _ = env_a.reset(seed=1)
    obs_b, _ = env_b.reset(seed=1)
    # Slot 0 of each perspective should describe a different unit, but both
    # must be valid observations with a mask.
    assert env_a.action_mask.any() or env_b.action_mask.any()
    assert obs_a.shape == obs_b.shape


def test_train_ppo_tiny_and_evaluate():
    config = TrainConfig(algo="ppo", total_timesteps=512, n_steps=64,
                         batch_size=64, n_epochs=2, verbose=0, seed=0)
    model, _ = train(["seele"], ["blaze_out_of_space"], config,
                     env_kwargs={"max_av": 1500, "max_rounds": 30})
    stats = evaluate_policy(model, ["seele"], ["blaze_out_of_space"],
                            n_games=2, opponent="random", seed=5,
                            env_kwargs={"max_av": 1500, "max_rounds": 30})
    assert 0.0 <= stats["win_rate"] <= 1.0


def test_save_and_load_bundle(tmp_path):
    config = TrainConfig(algo="dqn", total_timesteps=600, batch_size=64,
                         learning_starts=100, train_freq=4, verbose=0, seed=1)
    model, _ = train(["seele"], ["blaze_out_of_space"], config,
                     env_kwargs={"max_av": 1200, "max_rounds": 25})
    bundle = tmp_path / "testbundle"
    save_model_bundle(model, ModelMetadata(
        name="testbundle", algo="dqn", team=["seele"],
        trained_timesteps=600), bundle)
    loaded, meta = load_model_bundle(bundle)
    assert meta.team == ["seele"]
    # agent can choose in a real battle
    from cli_hsr import db
    battle = Battle([db.get_unit("seele")], [db.get_unit("blaze_out_of_space")],
                    rng=None)
    agent = CheckpointAgent(loaded, "test")
    actor = battle.side_a.units[0]
    legal = battle.legal_actions(actor)
    action = agent.choose(battle, actor, legal)
    assert action in legal


def test_checkpoint_agent_vs_greedy_completes():
    config = TrainConfig(algo="dqn", total_timesteps=400, batch_size=32,
                         learning_starts=50, verbose=0, seed=2)
    model, _ = train(["march_7th"], ["blaze_out_of_space"], config,
                     env_kwargs={"max_av": 1000, "max_rounds": 20})
    agent = CheckpointAgent(model, "tiny")
    from cli_hsr.agents import run_battle_between
    battle = run_battle_between(agent, __import__("cli_hsr.agents", fromlist=["GreedyAgent"]).GreedyAgent(0),
                                ["march_7th"], ["blaze_out_of_space"],
                                seed=9, max_av=1500)
    assert battle.winner in ("A", "B", "draw")
