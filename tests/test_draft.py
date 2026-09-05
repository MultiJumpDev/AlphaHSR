"""Tests for the draft/duel system: anyone-vs-anyone with per-unit levels."""

import numpy as np
import pytest

from cli_hsr import db
from cli_hsr.agents import GreedyAgent
from cli_hsr.draft import (
    DraftConfig,
    DraftEnv,
    FighterChoice,
    list_fighters,
    make_battle,
    run_duel,
)


@pytest.fixture(scope="module", autouse=True)
def seeded_db():
    db.seed()


class TestFighterPool:
    def test_all_unit_types_available(self):
        fighters = list_fighters()
        types = {f["unit_type"] for f in fighters}
        assert {"character", "normal", "elite", "boss"} <= types
        assert len(fighters) >= 100

    def test_characters_have_self_weakness_for_duels(self):
        f = {x["id"]: x for x in list_fighters()}["seele"]
        assert f["weaknesses"] == ["Quantum"]

    def test_every_fighter_can_be_built_as_unit(self):
        from cli_hsr.units import build_unit
        for f in list_fighters():
            row = db.get_unit(f["id"])
            u = build_unit(row, "A", 0, level=80)
            assert u.base.max_hp > 0
            assert u.base.atk > 0


class TestAnyVsAny:
    def test_character_vs_boss_resolves(self):
        w = run_duel(GreedyAgent("A"), GreedyAgent("B"),
                     FighterChoice(["seele"], [80]),
                     FighterChoice(["cocolia_boss"], [80]), seed=3)
        assert w.winner in ("A", "B", "draw")

    def test_boss_vs_boss_resolves(self):
        w = run_duel(GreedyAgent("A"), GreedyAgent("B"),
                     FighterChoice(["cocolia_boss"], [80]),
                     FighterChoice(["argenti_boss"], [80]), seed=3)
        assert w.winner in ("A", "B", "draw")

    def test_level_gap_decisive(self):
        w = run_duel(GreedyAgent("A"), GreedyAgent("B"),
                     FighterChoice(["seele"], [40]),
                     FighterChoice(["seele"], [80]), seed=3)
        assert w.winner == "B"

    def test_normal_enemy_vs_character_fair_fight(self):
        w = run_duel(GreedyAgent("A"), GreedyAgent("B"),
                     FighterChoice(["blaze_out_of_space"], [80]),
                     FighterChoice(["dan_heng"], [80]), seed=3)
        assert w.turn_count < 40  # no stall
        assert w.winner in ("A", "B", "draw")

    def test_mixed_team_with_per_unit_levels(self):
        pick = FighterChoice(["seele", "dan_heng"], [40, 80])
        b = make_battle(pick, FighterChoice(["blaze_out_of_space"], [80]))
        assert [u.level for u in b.side_a.units] == [40, 80]

    def test_imported_character_can_fight(self):
        w = run_duel(GreedyAgent("A"), GreedyAgent("B"),
                     FighterChoice(["aglaea"], [80]),
                     FighterChoice(["blaze_out_of_space"], [80]), seed=3)
        assert w.winner in ("A", "B", "draw")

    def test_full_battle_from_make_battle(self):
        b = make_battle(FighterChoice(["seele"], [80]),
                        FighterChoice(["blaze_out_of_space"], [80]))
        winner = b.run(
            lambda bt, actor, legal: legal[0],
            lambda bt, actor, legal: legal[0],
        )
        assert winner in ("A", "B", "draw")


class TestDraftEnv:
    def test_pool_and_action_space(self):
        env = DraftEnv(seed=1)
        assert env.n_pool >= 100
        assert env.draft_action_space_size == env.n_pool * len(env.draft_config.level_buckets)

    def test_full_episode_flow(self):
        env = DraftEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert info["phase"] == "draft"

        pool_ids = [f["id"] for f in env.pool]
        si = pool_ids.index("seele")
        action = si * env.n_level_buckets + 3  # seele @ Lv80 bucket
        assert env.action_mask[action]
        obs, rew, term, trunc, info = env.step(action)

        assert info["phase"] == "battle"
        assert info["my_picks"] == ["seele"]
        assert info["my_levels"] == [80]
        assert info["opp_picks"] and info["opp_levels"]

        # play out the battle with first-legal-action policy
        steps = 0
        while not (term or trunc) and steps < 500:
            legal = np.flatnonzero(np.array(env.action_masks()))
            if len(legal) == 0:
                break
            obs, r, term, trunc, info = env.step(int(legal[0]))
            steps += 1
        assert term or trunc or steps >= 500
        assert info.get("winner") in ("A", "B", "draw", None)

    def test_draft_masks_exclude_picked_units(self):
        env = DraftEnv(seed=1, draft_config=DraftConfig(team_size=2))
        env.reset(seed=1)
        pool_ids = [f["id"] for f in env.pool]
        si = pool_ids.index("seele")
        base = si * env.n_level_buckets
        env.step(base + 3)
        # every remaining legal action must not be seele
        for a in np.flatnonzero(np.array(env.draft_action_masks())):
            idx = int(a) // env.n_level_buckets
            assert pool_ids[idx] != "seele"

    def test_illegal_draft_action_penalized(self):
        env = DraftEnv(seed=1)
        env.reset(seed=1)
        env.action_mask[:] = False  # force illegal
        obs, r, term, trunc, info = env.step(0)
        assert r == -1.0
        assert info.get("illegal") is True

    def test_observation_shape_matches_space(self):
        env = DraftEnv(seed=1)
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        pool_ids = [f["id"] for f in env.pool]
        si = pool_ids.index("seele")
        obs, _, _, _, _ = env.step(si * env.n_level_buckets + 3)
        assert obs.shape == env.observation_space.shape
