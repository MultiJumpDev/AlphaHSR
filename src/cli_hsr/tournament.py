"""Tournament system for model competition.

Goal 2 of the project: pit trained models against each other in 1v1 team
battles organized in groups of 4 (round robin) then a knockout bracket
(quarters -> semis -> final).
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Any

from . import db
from .agents import RandomAgent, run_battle_between
from .engine import Battle


@dataclass
class Contestant:
    """A trained model (or baseline agent) that owns a full team.

    `gear` is a per-slot list of loadout dicts, e.g.
        [{"light_cone": "swordplay", "superimposition": 5,
          "relics": {"musketeer_of_wild_wheat": 4}}, None, ...]
    aligned with `team`. `gear` may also be a single dict applied to slot 0.
    """
    name: str
    team: list[str]                 # unit ids
    agent_factory: Any              # zero-arg callable returning an agent instance
    seed: int = 0
    gear: list[dict] | dict | None = None

    def gear_for_slots(self) -> list[dict]:
        if self.gear is None:
            return [{} for _ in self.team]
        if isinstance(self.gear, dict):
            return [self.gear] + [{} for _ in range(len(self.team) - 1)]
        out = [{} for _ in self.team]
        for i, g in enumerate(self.gear[: len(self.team)]):
            if g:
                out[i] = g
        return out

    def new_agent(self):
        return self.agent_factory()


@dataclass
class MatchOutcome:
    contestant_a: str
    contestant_b: str
    winner: str                     # name, or "draw"
    rounds: int
    seed: int
    log: list[dict[str, Any]] = field(default_factory=list)


def play_match(ca: Contestant, cb: Contestant, seed: int = 0,
               verbose: bool = False, best_of: int = 1, **battle_kw: Any) -> MatchOutcome:
    """Play best-of-N battles between two contestants (mirrored seeds)."""
    from .gear import apply_loadout_to_row

    def rows_for(contestant: Contestant) -> list[dict[str, Any]]:
        gear_slots = contestant.gear_for_slots()
        rows = []
        for uid, g in zip(contestant.team, gear_slots):
            row = db.get_unit(uid)
            if row is None:
                raise ValueError(f"unknown unit id: {uid}")
            if g:
                row = apply_loadout_to_row(row, g)
            rows.append(row)
        return rows

    wins_a = wins_b = draws = 0
    last: Battle | None = None
    total_rounds = 0
    for game in range(best_of):
        battle_seed = seed * 1000 + game
        # Fair-play protocol: alternate who takes side A across games.
        # Side A always wins AV ties (acts first), which is a real in-game
        # advantage; without swapping, mirror matches would be systematically
        # decided by list order.
        first, second = (ca, cb) if game % 2 == 0 else (cb, ca)
        battle = Battle(
            rows_for(first), rows_for(second),
            name_a=first.name, name_b=second.name,
            rng=random.Random(battle_seed), verbose=verbose, **battle_kw,
        )

        # keep a stable agent per battle for stateful agents
        agent_a = first.new_agent()
        agent_b = second.new_agent()
        winner = battle.run(lambda b, a, l: agent_a.choose(b, a, l),
                            lambda b, a, l: agent_b.choose(b, a, l))
        agent_a.end_battle(winner == "A", winner == "draw")
        agent_b.end_battle(winner == "B", winner == "draw")
        total_rounds += battle.turn_count
        game_winner_name = first.name if battle.winner == "A" else (
            second.name if battle.winner == "B" else "draw")
        if game_winner_name == ca.name:
            wins_a += 1
        elif game_winner_name == cb.name:
            wins_b += 1
        else:
            draws += 1
        last = battle
    if wins_a > wins_b:
        winner = ca.name
    elif wins_b > wins_a:
        winner = cb.name
    elif draws > 0:
        winner = "draw"
    else:  # pragma: no cover
        winner = "draw"

    outcome = MatchOutcome(ca.name, cb.name, winner, total_rounds, seed,
                           last.log if last else [])
    db.save_match_result(
        team_a=json.dumps(ca.team), team_b=json.dumps(cb.team),
        agent_a=ca.name, agent_b=cb.name, winner=("draw" if winner == "draw" else
                                                  ("A" if winner == ca.name else "B")),
        rounds=total_rounds, seed_val=seed,
    )
    return outcome


def round_robin(contestants: list[Contestant], seed: int = 0,
                best_of: int = 1, **battle_kw: Any) -> dict[str, dict[str, int]]:
    stats = {c.name: {"wins": 0, "losses": 0, "draws": 0, "points": 0} for c in contestants}
    for ca, cb in itertools.combinations(contestants, 2):
        outcome = play_match(ca, cb, seed=seed, best_of=best_of, **battle_kw)
        if outcome.winner == ca.name:
            stats[ca.name]["wins"] += 1
            stats[cb.name]["losses"] += 1
            stats[ca.name]["points"] += 3
        elif outcome.winner == cb.name:
            stats[cb.name]["wins"] += 1
            stats[ca.name]["losses"] += 1
            stats[cb.name]["points"] += 3
        else:
            stats[ca.name]["draws"] += 1
            stats[cb.name]["draws"] += 1
            stats[ca.name]["points"] += 1
            stats[cb.name]["points"] += 1
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["points"]))


class Tournament:
    """Groups of 4 (round robin) then knockout: groups x (qf/sf/final).

    With 8 contestants: 2 groups of 4 -> top 2 advance -> semis -> final.
    With 4 contestants: 1 group of 4 -> top 4 -> semis -> final.
    """

    def __init__(self, contestants: list[Contestant], group_size: int = 4,
                 advance_per_group: int = 2, seed: int = 0, best_of: int = 1,
                 name: str = "hsr-tournament", **battle_kw: Any) -> None:
        if len(contestants) % group_size != 0:
            raise ValueError("number of contestants must be a multiple of group_size")
        self.contestants = {c.name: c for c in contestants}
        self.group_size = group_size
        self.advance_per_group = advance_per_group
        self.seed = seed
        self.best_of = best_of
        self.name = name
        self.battle_kw = battle_kw
        self.results: dict[str, Any] = {"groups": {}, "knockout": []}

    def _shuffled(self) -> list[Contestant]:
        c = list(self.contestants.values())
        random.Random(self.seed).shuffle(c)
        return c

    def run_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[Contestant]] = {}
        shuffled = self._shuffled()
        for gi in range(0, len(shuffled), self.group_size):
            group = shuffled[gi:gi + self.group_size]
            gname = f"Group {chr(ord('A') + gi // self.group_size)}"
            groups[gname] = group
            stats = round_robin(group, seed=self.seed + gi,
                                best_of=self.best_of, **self.battle_kw)
            self.results["groups"][gname] = stats
            db.save_tournament_result(self.name, gname,
                                      [c.name for c in group],
                                      winner=stats and next(iter(stats)),
                                      scoreboard=json.dumps(stats))
        return {g: [c.name for c in group] for g, group in groups.items()}

    def run_knockout(self, qualified: dict[str, list[str]]) -> str:
        bracket = [name for names in qualified.values() for name in names[:self.advance_per_group]]
        round_no, round_name = 0, "Quarterfinals"
        while len(bracket) > 1:
            if len(bracket) == 8:
                round_name = "Quarterfinals"
            elif len(bracket) == 4:
                round_name = "Semifinals"
            elif len(bracket) == 2:
                round_name = "Final"
            else:
                round_name = f"Round of {len(bracket)}"
            next_round: list[str] = []
            round_log: list[dict[str, Any]] = []
            for i in range(0, len(bracket), 2):
                a_name, b_name = bracket[i], bracket[i + 1]
                outcome = play_match(self.contestants[a_name], self.contestants[b_name],
                                     seed=self.seed + 100 + round_no * 10 + i,
                                     best_of=self.best_of, **self.battle_kw)
                winner = a_name if outcome.winner == a_name else (
                    b_name if outcome.winner == b_name else None)
                if winner is None:
                    # tie-break by total team HP remaining proxy: replay once
                    outcome = play_match(self.contestants[a_name], self.contestants[b_name],
                                         seed=self.seed + 777, best_of=3, **self.battle_to3())
                    winner = (a_name if outcome.winner == a_name else
                              (b_name if outcome.winner == b_name else a_name))
                round_log.append({"a": a_name, "b": b_name, "winner": winner})
                next_round.append(winner)
                round_no += 1
            self.results["knockout"].append({round_name: round_log})
            db.save_tournament_result(self.name, round_name, bracket, winner=next_round[0] if len(next_round) == 1 else "",
                                      scoreboard=json.dumps(round_log))
            bracket = next_round
            round_no += 1
        champion = bracket[0]
        self.results["champion"] = champion
        return champion

    def battle_to3(self):  # pragma: no cover - helper
        return dict(self.battle_kw)

    def run(self) -> str:
        qualified = self.run_groups()
        champion = self.run_knockout(qualified)
        return champion


def format_tournament_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    for gname, stats in results.get("groups", {}).items():
        lines.append(f"== {gname} ==")
        for name, s in stats.items():
            lines.append(f"  {name:<18} W{s['wins']} L{s['losses']} D{s['draws']}  {s['points']} pts")
    for round_entry in results.get("knockout", []):
        (round_name, matches), = round_entry.items()
        lines.append(f"== {round_name} ==")
        for m in matches:
            lines.append(f"  {m['a']} vs {m['b']} -> {m['winner']}")
    lines.append(f"CHAMPION: {results.get('champion', '?')}")
    return "\n".join(lines)
