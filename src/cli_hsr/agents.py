"""Simple policies/agents: compatible with both the engine chooser API and the env."""

from __future__ import annotations

import random
from typing import Any, Iterable

from .engine import Battle
from .units import Unit


# ---------------------------------------------------------------------- #
# Chooser functions (battle, actor, legal) -> action dict                 #
# ---------------------------------------------------------------------- #
def random_action(battle: Battle, actor: Unit, legal: list[dict[str, Any]]) -> dict[str, Any]:
    return random.choice(legal)


class RandomAgent:
    """Stateful agent wrapper (for tournaments)."""
    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, battle: Battle, actor: Unit, legal: list[dict[str, Any]]) -> dict[str, Any]:
        return self.rng.choice(legal)

    def observe(self, battle: Battle, actor: Unit, action: dict[str, Any], result: dict[str, Any]) -> None:
        pass

    def end_battle(self, won: bool, draw: bool) -> None:  # pragma: no cover
        pass


class GreedyAgent:
    """Heuristic baseline: prefer Ultimate > Skill on weak element > Skill > Basic.
    Targets the enemy with the lowest HP; supports allies when required."""

    name = "Greedy"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, battle: Battle, actor: Unit, legal: list[dict[str, Any]]) -> dict[str, Any]:
        def score(act: dict[str, Any]) -> float:
            kind = act["kind"]
            s = 0.0
            if kind == "ultimate":
                s += 100
            elif kind == "skill":
                s += 50
            if act.get("target") is not None:
                t = act["target"]
                if t.is_enemy if hasattr(t, "is_enemy") else False:
                    base = 10 + 90 * (1 - t.hp_fraction())
                    if t.is_weak_to(actor.element):
                        base += 40
                    if t.weakness_broken:
                        base += 30
                    s += base
            return s

        return max(legal, key=score)

    def observe(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass

    def end_battle(self, won: bool, draw: bool) -> None:  # pragma: no cover
        pass


class HumanAgent:
    """Interactive console play: prints the battle and asks for a move."""

    name = "Human"

    def choose(self, battle: Battle, actor: Unit, legal: list[dict[str, Any]]) -> dict[str, Any]:
        print()
        print(f">>> {actor.name}'s turn (SP={battle.skill_points}, "
              f"AV={battle.time:.0f})")
        print("    Enemies:")
        for i, f in enumerate(battle.enemies_of(actor)):
            weak = "/".join(f.weaknesses)
            print(f"      [{i}] {f.name}  HP {f.hp:.0f}/{f.max_hp:.0f}"
                  f"  Tough {f.toughness:.0f}/{f.max_toughness:.0f}  weak: {weak}"
                  f"{'  [BROKEN]' if f.weakness_broken else ''}")
        print("    Your team:")
        for i, a in enumerate(battle.allies_of(actor)):
            e = f"  ULT READY ({a.energy:.0f}/{a.base.energy_max})" if a.ult_ready() else ""
            print(f"      {a.name}  HP {a.hp:.0f}/{a.max_hp:.0f}{e}")

        print("    Actions:")
        for i, act in enumerate(legal):
            target = act.get("target")
            tname = target.name if target is not None else "-"
            extra = ""
            if act["kind"] == "skill":
                extra = f" (SP cost 1, {battle.skill_points} left)"
            if act["kind"] == "ultimate":
                extra = " (ready!)"
            print(f"      [{i}] {act['kind'].upper():<9} -> {tname}{extra}")

        while True:
            raw = input("    choose action index: ").strip()
            if raw.isdigit() and int(raw) < len(legal):
                return legal[int(raw)]
            print("    invalid input")

    def observe(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        pass

    def end_battle(self, won: bool, draw: bool) -> None:  # pragma: no cover
        print("VICTORY!" if won else ("DRAW" if draw else "DEFEAT"))


def run_battle_between(agent_a, agent_b, team_a: list[str], team_b: list[str],
                       seed: int = 0, verbose: bool = False, **kw: Any) -> Battle:
    """Utility: run a full battle between two chooser-style agents."""
    from . import db as _db

    rows_a = [_db.get_unit(u) for u in team_a]
    rows_b = [_db.get_unit(u) for u in team_b]
    battle = Battle(rows_a, rows_b, name_a=agent_a.name, name_b=agent_b.name,
                    rng=random.Random(seed), verbose=verbose, **kw)

    def chooser(agent):
        return lambda b, actor, legal: agent.choose(b, actor, legal)

    winner = battle.run(chooser(agent_a), chooser(agent_b))
    agent_a.end_battle(winner == "A", winner == "draw")
    agent_b.end_battle(winner == "B", winner == "draw")
    return battle
