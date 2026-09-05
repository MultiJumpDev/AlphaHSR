import random

import pytest

from cli_hsr import db
from cli_hsr.agents import RandomAgent, run_battle_between
from cli_hsr.engine import Battle
from cli_hsr.gear import (Loadout, apply_loadout_to_row, cone_base_stats,
                          cone_passives, get_light_cone, get_relic_set,
                          loadout_from_dict, relic_set_bonuses,
                          resolve_loadout_stats)
from cli_hsr.tournament import Contestant, Tournament, play_match


@pytest.fixture(scope="module", autouse=True)
def seeded_db():
    db.seed()
    yield


# ------------------------------------------------------------------ #
# roster expansion                                                    #
# ------------------------------------------------------------------ #
def test_expanded_roster_elements_and_paths():
    chars = db.list_units("character")
    ids = {c["id"] for c in chars}
    expected = {"serval", "pela", "herta", "hanya", "astamatek", "sushang",
                "natasha", "bailu", "gepard", "welt", "argenti", "yanqing"}
    assert expected <= ids
    assert len(chars) >= 26
    elements = {c["element"] for c in chars}
    assert {"Physical", "Fire", "Ice", "Lightning", "Wind", "Quantum",
            "Imaginary"} <= elements


def test_new_enemies_seeded():
    ids = db.enemy_ids()
    assert "shadow_of_feixiao" in ids and "topaz_boss" in ids


# ------------------------------------------------------------------ #
# gear resolution                                                     #
# ------------------------------------------------------------------ #
def test_cone_base_stats_and_passives():
    cone = get_light_cone("swordplay")
    assert cone is not None
    stats = cone_base_stats(cone)
    assert stats["atk_flat"] == 476
    passive_s1 = cone_passives(cone, 1)
    passive_s5 = cone_passives(cone, 5)
    assert passive_s1[0]["value"] == pytest.approx(0.08)
    assert passive_s5[0]["value"] == pytest.approx(0.16)


def test_energy_cone_interpolation():
    cone = get_light_cone("today_is_another_peaceful_day")
    p1 = cone_passives(cone, 1)[0]
    p5 = cone_passives(cone, 5)[0]
    assert p1["value"] == 16 and p5["value"] == 32


def test_relic_set_bonuses_by_pieces():
    musk = get_relic_set("musketeer_of_wild_wheat")
    assert relic_set_bonuses(musk, 2) == [
        {"stat": "atk_percent", "value": 0.12, "set": "musketeer_of_wild_wheat",
         "pieces": 2, "note": ""}]
    b4 = relic_set_bonuses(musk, 4)
    assert len(b4) == 2


def test_resolve_loadout_stats_combines_cone_and_relics():
    loadout = Loadout(unit_id="seele", light_cone="swordplay", superimposition=5,
                      relic_sets={"musketeer_of_wild_wheat": 4, "inert_salsotto": 2})
    stats, passives = resolve_loadout_stats(loadout)
    assert stats["atk_flat"] == pytest.approx(476)
    assert stats["atk_percent"] == pytest.approx(0.12)
    assert stats["crit_rate"] == pytest.approx(0.08)
    assert any(p["kind"] == "battle_start_buff" for p in passives)


def test_apply_loadout_to_row_bakes_gear():
    row = db.get_unit("seele")
    geared = apply_loadout_to_row(row, {
        "light_cone": "sleep_like_the_dead", "superimposition": 1,
        "relics": {"inert_salsotto": 2}})
    gear = geared["kit_json"]["gear"]
    # relic bonus baked into stats
    assert gear["stats"]["crit_rate"] == pytest.approx(0.08)
    # cone passive arrives as a battle-start buff (S1 = +0.18 crit rate)
    passive = next(p for p in gear["passives"] if p["kind"] == "battle_start_buff")
    assert passive["stat"] == "crit_rate" and passive["value"] == pytest.approx(0.18)
    # original row untouched
    assert "gear" not in row["kit_json"]


def test_geared_unit_stronger_in_battle():
    plain = db.get_unit("seele")
    geared = apply_loadout_to_row(db.get_unit("seele"), {
        "light_cone": "sleep_like_the_dead", "superimposition": 5})
    b1 = Battle([plain], [db.get_unit("blaze_out_of_space")], rng=random.Random(0))
    b2 = Battle([geared], [db.get_unit("blaze_out_of_space")], rng=random.Random(0))
    assert b2.side_a.units[0].effective_crit_rate() > b1.side_a.units[0].effective_crit_rate()


def test_energy_flat_start_cone():
    geared = apply_loadout_to_row(db.get_unit("seele"), {
        "light_cone": "today_is_another_peaceful_day", "superimposition": 5})
    battle = Battle([geared], [db.get_unit("blaze_out_of_space")], rng=random.Random(0))
    # base start = 50% of 120 = 60; +32 from the cone
    assert battle.side_a.units[0].energy == pytest.approx(92)


# ------------------------------------------------------------------ #
# tournament with gear                                                #
# ------------------------------------------------------------------ #
def test_play_match_with_gear():
    ca = Contestant("Geared", ["seele"], lambda: RandomAgent(0),
                    gear={"light_cone": "swordplay", "superimposition": 3,
                          "relics": {"inert_salsotto": 2}})
    cb = Contestant("Plain", ["blaze_out_of_space"], lambda: Greedy(1))
    outcome = play_match(ca, cb, seed=4, max_av=2000)
    assert outcome.winner in ("Geared", "Plain", "draw")


class Greedy:
    def __init__(self, seed):
        from cli_hsr.agents import GreedyAgent
        self._a = GreedyAgent(seed)

    @property
    def name(self):
        return "Greedy"

    def choose(self, b, a, l):
        return self._a.choose(b, a, l)

    def end_battle(self, won, draw):
        pass


def test_tournament_four_with_gear():
    gear = {"light_cone": "meshing_cogs", "superimposition": 5}
    contestants = [
        Contestant(f"G{i}", ["seele", "march_7th"],
                   lambda i=i: RandomAgent(seed=i), gear=gear)
        for i in range(4)
    ]
    t = Tournament(contestants, group_size=4, advance_per_group=2, seed=3, max_av=1200)
    champ = t.run()
    assert champ in [c.name for c in contestants]
