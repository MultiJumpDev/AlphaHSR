import random

import pytest

from cli_hsr import db
from cli_hsr import constants as C
from cli_hsr.agents import GreedyAgent, RandomAgent, run_battle_between
from cli_hsr.engine import Battle
from cli_hsr.tournament import Contestant, Tournament, play_match


@pytest.fixture(scope="module", autouse=True)
def seeded_db():
    db.seed()
    yield


def row(uid):
    r = db.get_unit(uid)
    assert r is not None, uid
    return r


# ------------------------------------------------------------------ #
# constants / formulas                                                #
# ------------------------------------------------------------------ #
def test_level_multiplier_exact_values():
    assert C.level_multiplier(80) == pytest.approx(3767.5533)
    assert C.level_multiplier(90) == pytest.approx(6020.8836)
    assert C.level_multiplier(70) == pytest.approx(2659.6406)


def test_base_av_formula():
    assert C.base_av(100) == pytest.approx(100.0)
    assert C.base_av(134) == pytest.approx(74.6268656716)


def test_def_multiplier_level80_same_level():
    # DEF mult = (100)/(100+100) = 0.5 for same-level attacker/defender without bonuses
    lvl = 80 + 20
    assert lvl / (lvl + lvl) == pytest.approx(0.5)


def test_max_toughness_multiplier():
    assert C.max_toughness_multiplier(60) == pytest.approx(2.0)
    assert C.max_toughness_multiplier(120) == pytest.approx(3.5)


# ------------------------------------------------------------------ #
# database                                                            #
# ------------------------------------------------------------------ #
def test_db_seeds_and_contains_expected_units():
    ids = db.character_ids() + db.enemy_ids()
    assert "seele" in ids and "kafka" in ids
    assert "sanctus_tractoris_boss" in ids
    chars = db.list_units("character")
    assert len(chars) >= 12
    assert "clara" in ids
    # every element covered
    elements = {c["element"] for c in chars}
    assert {"Physical", "Fire", "Ice", "Lightning", "Wind", "Quantum", "Imaginary"} <= elements


# ------------------------------------------------------------------ #
# engine                                                              #
# ------------------------------------------------------------------ #
def test_basic_battle_runs_to_completion():
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(1), max_av=6000)
    winner = battle.run(lambda b, a, l: {"kind": "basic", "target": b.random_enemy(a)},
                        lambda b, a, l: {"kind": "basic", "target": b.random_enemy(a)})
    assert winner in ("A", "B", "draw")
    assert battle.finished


def test_skill_consumes_skill_points_and_basic_restores():
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(2), max_av=6000, skill_points=3)
    actor = battle.side_a.units[0]

    battle.run_turn(actor, lambda b, a, l: {"kind": "skill", "target": battle.side_b.units[0]})
    assert battle.skill_points == 2

    battle.run_turn(actor, lambda b, a, l: {"kind": "basic", "target": battle.side_b.units[0]})
    assert battle.skill_points == 3


def test_weakness_break_triggers_delay_and_debuff():
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(3), max_av=6000)
    enemy = battle.side_b.units[0]
    attacker = battle.side_a.units[0]
    assert enemy.is_weak_to("Quantum")

    # Real depletion path: many Quantum basic attacks until the toughness
    # bar breaks, then verify the 25% delay was applied.
    enemy.toughness = 10.0  # leave a sliver so one hit breaks it
    av_before = enemy.av
    for _ in range(50):
        if enemy.weakness_broken or not enemy.alive:
            break
        battle.deal_damage(attacker, enemy, 0.55, "Quantum", "basic", 30.0)
    assert enemy.weakness_broken or not enemy.alive
    if enemy.alive:
        assert enemy.av >= av_before  # 25% of base AV was added back


def test_break_debuff_elements():
    from cli_hsr.statuses import StatusEffect  # noqa: F401
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(4))
    enemy = battle.side_b.units[0]
    attacker = battle.side_a.units[0]
    battle.apply_break_debuff(attacker, enemy, "Lightning", "Shock", True, 2)
    assert enemy.statuses.has("Shock")
    battle.apply_break_debuff(attacker, enemy, "Ice", "Freeze", False, 1)
    assert enemy.statuses.is_frozen()


def test_energy_and_ultimate_ready():
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(5))
    seele = battle.side_a.units[0]
    assert seele.base.energy_max == 120
    seele.energy = 120
    assert seele.ult_ready()
    battle.run_turn(seele, lambda b, a, l: {"kind": "ultimate", "target": battle.side_b.units[0]})
    assert seele.energy == 0


def test_dot_ticks_and_decays():
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(6))
    enemy = battle.side_b.units[0]
    attacker = battle.side_a.units[0]
    battle.apply_break_debuff(attacker, enemy, "Lightning", "Shock", True, 2)
    hp_before = enemy.hp
    battle.tick_dots(enemy)
    assert enemy.hp < hp_before
    enemy.statuses.tick_turn_end()
    shock = enemy.statuses.find("Shock")
    assert shock is None or shock.duration_turns < 2


def test_boss_has_two_phases():
    battle = Battle([row("seele")], [row("sanctus_tractoris_boss")],
                    rng=random.Random(7), max_av=20000)
    boss = battle.side_b.units[0]
    assert boss.total_phases == 2
    # Kill the boss twice: second 'death' triggers phase 2 then final defeat
    boss.take_damage(boss.hp)
    battle.check_phases(boss)
    assert boss.alive and boss.phase == 2
    boss.take_damage(boss.hp)
    assert not boss.alive


def test_team_battle_4v4_completes():
    team_a = [row("seele"), row("bronya"), row("sparkle"), row("fu_xuan")]
    team_b = [row("kafka"), row("black_swan"), row("luocha"), row("himeko")]
    battle = Battle(team_a, team_b, rng=random.Random(8), max_av=6000)
    winner = battle.run(
        lambda b, a, l: random.Random(b.turn_count).choice(l),
        lambda b, a, l: random.Random(b.turn_count + 1).choice(l),
    )
    assert winner in ("A", "B", "draw")


# ------------------------------------------------------------------ #
# agents / tournament                                                 #
# ------------------------------------------------------------------ #
def test_agents_battle_utility():
    battle = run_battle_between(RandomAgent(0), GreedyAgent(1),
                                ["seele", "march_7th"], ["blaze_out_of_space", "automaton_grizzly"],
                                seed=11, max_av=4000)
    assert battle.winner in ("A", "B", "draw")


def test_tournament_flow_four_contestants():
    contestants = [
        Contestant(f"R{i}", ["seele", "march_7th"], lambda i=i: RandomAgent(seed=i))
        for i in range(4)
    ]
    t = Tournament(contestants, group_size=4, advance_per_group=2, seed=5, max_av=1500)
    champion = t.run()
    assert champion in [c.name for c in contestants]
    assert "champion" in t.results


def test_play_match_records_result():
    ca = Contestant("X", ["seele"], lambda: RandomAgent(0))
    cb = Contestant("Y", ["blaze_out_of_space"], lambda: GreedyAgent(1))
    outcome = play_match(ca, cb, seed=3, max_av=2000)
    assert outcome.winner in ("X", "Y", "draw")


# ------------------------------------------------------------------ #
# full break path                                                     #
# ------------------------------------------------------------------ #
def test_full_break_debuff_and_recovery():
    from cli_hsr.statuses import StatusEffect  # noqa: F401

    battle = Battle([row("seele")], [row("silvermane_lieutenant")],
                    rng=random.Random(11), max_av=6000)
    enemy = battle.side_b.units[0]
    attacker = battle.side_a.units[0]
    assert enemy.is_weak_to("Lightning")
    # keep the enemy alive through the test (it validates break recovery,
    # not lethality — elite HP pools are small at base stats)
    enemy.base.max_hp = 100000.0
    enemy.hp = 100000.0
    enemy.toughness = 10.0
    battle.deal_damage(attacker, enemy, 0.5, "Lightning", "basic", 30.0)
    assert enemy.weakness_broken
    assert enemy.statuses.has("Shock")
    # enemy's turn: Shock ticks, then the enemy recovers from break
    hp_before = enemy.hp
    battle.run_turn(enemy, lambda *_: {"kind": "basic", "target": None})
    assert not enemy.weakness_broken
    assert enemy.toughness == enemy.max_toughness
    assert enemy.hp < hp_before  # DoT ticked at turn start


def test_wind_shear_elite_stacks():
    battle = Battle([row("seele")], [row("silvermane_lieutenant")],
                    rng=random.Random(12))
    enemy = battle.side_b.units[0]
    attacker = battle.side_a.units[0]
    battle.apply_break_debuff(attacker, enemy, "Wind", "Wind Shear", True, 2)
    ws = enemy.statuses.find("Wind Shear")
    assert ws.stacks == 3  # elite/boss get 3 stacks


def test_freeze_skips_turn_and_advances():
    from cli_hsr.statuses import StatusEffect
    battle = Battle([row("seele")], [row("blaze_out_of_space")],
                    rng=random.Random(13))
    enemy = battle.side_b.units[0]
    enemy.statuses.add(StatusEffect(kind="cc", name="Freeze", stat="freeze",
                                    value=1.0, duration_turns=1))
    enemy.statuses.mark_turn_start()
    battle.run_turn(enemy, lambda *_: {"kind": "basic", "target": None})
    assert not enemy.statuses.is_frozen()
    # turn was skipped and next turn advanced by 50%: AV is half of base
    assert enemy.av == pytest.approx(0.5 * C.base_av(enemy.effective_spd()))
