"""Interactive command line interface for cli-hsr."""

from __future__ import annotations

import argparse
import random
import sys
from typing import Any

from . import db
from .agents import GreedyAgent, HumanAgent, RandomAgent, run_battle_between
from .rl.train import resolve_device
from .engine import Battle
from .tournament import Contestant, Tournament, format_tournament_report


def _print_units(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        stats = r["stats_json"]
        line = (f"  {r['id']:<22} {r['name']:<28} {r['unit_type']:<10} "
                f"{r['element']:<10} HP {stats.get('max_hp', '?')}")
        if r["unit_type"] == "character":
            print(line)
        else:
            kit = r["kit_json"]
            print(line + f"  weak: {'/'.join(kit.get('weaknesses', []))}")


def cmd_list(args: argparse.Namespace) -> None:
    if args.type in ("character", "all"):
        print("Characters:")
        _print_units(db.list_units("character"))
    if args.type in ("enemy", "all"):
        print("\nEnemies:")
        for cat in ("normal", "elite", "boss"):
            _print_units(db.list_units(cat))
    if args.type in ("gear", "all"):
        print("\nLight Cones:")
        for row in db.list_light_cones():
            print(f"  {row['id']:<36} {row['name']:<36} {row.get('path') or ''}")
        print("\nRelic Sets:")
        for row in db.list_relic_sets():
            print(f"  {row['id']:<36} {row['name']}")


def _parse_team(spec: str) -> list[str]:
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    if not (1 <= len(ids) <= 5):
        raise SystemExit("team must have between 1 and 5 units (comma separated ids)")
    for uid in ids:
        if db.get_unit(uid) is None:
            raise SystemExit(f"unknown unit id: {uid} (see `python main.py list`)")
    return ids


def _parse_gear(spec: str | None) -> list[dict] | None:
    """Gear spec: `unit-slot:cone=S,sid=pieces|...` per slot separated by ';'.

    Example for 2 slots:
        swordplay=5,musketeer_of_wild_wheat=4;sleep_like_the_dead=5,inert_salsotto=2
    """
    if not spec:
        return None
    slots: list[dict] = []
    for part in spec.split(";"):
        loadout: dict = {}
        part = part.strip()
        if part:
            for tok in part.split(","):
                k, _, v = tok.partition("=")
                if k in ("swordplay", "sleep_like_the_dead") or _is_cone(k):
                    loadout["light_cone"] = k
                    loadout["superimposition"] = int(v or 1)
                else:
                    loadout.setdefault("relics", {})[k] = int(v or 2)
        slots.append(loadout)
    return slots


def _is_cone(token: str) -> bool:
    from .gear import get_light_cone
    return get_light_cone(token) is not None


def _agent_from_spec(spec: str, seed: int):
    if spec.startswith("model:"):
        from .registry import CheckpointAgent
        path = spec.split(":", 1)[1]
        from .registry import load_model_bundle
        model, meta = load_model_bundle(path)
        return CheckpointAgent(model, meta.name)
    classes = {"random": RandomAgent, "greedy": GreedyAgent, "human": HumanAgent}
    cls = classes[spec]
    return HumanAgent() if cls is HumanAgent else cls(seed)


def cmd_fight(args: argparse.Namespace) -> None:
    team_a = _parse_team(args.team_a)
    team_b = _parse_team(args.team_b)
    gear_a = _parse_gear(args.gear_a)
    gear_b = _parse_gear(args.gear_b)

    agent_a = _agent_from_spec(args.agent_a, args.seed or 0)
    agent_b = _agent_from_spec(args.agent_b, (args.seed or 0) + 1)

    from .gear import apply_loadout_to_row

    def rows_for(team: list[str], gear: list[dict] | None) -> list[dict]:
        out = []
        for i, uid in enumerate(team):
            row = db.get_unit(uid)
            if gear and i < len(gear) and gear[i]:
                row = apply_loadout_to_row(row, gear[i])
            out.append(row)
        return out

    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    battle = Battle(rows_for(team_a, gear_a), rows_for(team_b, gear_b),
                    name_a=args.agent_a, name_b=args.agent_b,
                    rng=random.Random(seed), verbose=not args.quiet,
                    max_av=args.max_av, max_rounds=args.max_rounds)
    winner = battle.run(lambda b, a, l: agent_a.choose(b, a, l),
                        lambda b, a, l: agent_b.choose(b, a, l))
    agent_a.end_battle(winner == "A", winner == "draw")
    agent_b.end_battle(winner == "B", winner == "draw")
    print("=" * 60)
    print(f"Winner: {battle.winner}  ({battle.time:.0f} AV, {battle.turn_count} turns)")
    db.save_match_result(
        team_a=str(team_a), team_b=str(team_b),
        agent_a=args.agent_a, agent_b=args.agent_b,
        winner=battle.winner or "draw", rounds=battle.turn_count, seed_val=seed,
    )


def _parse_duel_side(spec: str) -> tuple[list[str], list[int]]:
    """'seele:80,dan_heng:60' -> ([seele, dan_heng], [80, 60])"""
    units: list[str] = []
    levels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        uid, _, lvl = part.partition(":")
        units.append(uid.strip())
        levels.append(int(lvl) if lvl else 80)
    if not units:
        raise SystemExit(f"empty duel side spec: {spec!r}")
    return units, levels


def cmd_duel(args: argparse.Namespace) -> None:
    from .draft import FighterChoice, run_duel

    units_a, levels_a = _parse_duel_side(args.side_a)
    units_b, levels_b = _parse_duel_side(args.side_b)
    pick_a = FighterChoice(units_a, levels_a)
    pick_b = FighterChoice(units_b, levels_b)

    agent_a = _agent_from_spec(args.agent_a, args.seed or 0)
    agent_b = _agent_from_spec(args.agent_b, (args.seed or 0) + 1)

    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    battle = run_duel(agent_a, agent_b, pick_a, pick_b,
                      seed=seed, max_av=args.max_av)
    if not args.quiet:
        for e in battle.log:
            print(battle.format_event(e))
    print("=" * 60)
    print(f"Winner: {battle.winner}  ({battle.time:.0f} AV, {battle.turn_count} turns)")
    print(f"  A: {', '.join(f'{u} (Lv{lv})' for u, lv in zip(units_a, levels_a))}")
    print(f"  B: {', '.join(f'{u} (Lv{lv})' for u, lv in zip(units_b, levels_b))}")
    db.save_match_result(
        team_a=f"{units_a}@{levels_a}", team_b=f"{units_b}@{levels_b}",
        agent_a=args.agent_a, agent_b=args.agent_b,
        winner=battle.winner or "draw", rounds=battle.turn_count, seed_val=seed,
    )


def _team_prompt(side: str, defaults: list[str]) -> list[str]:
    raw = input(f"{side} team (ids comma separated, empty = {'/'.join(defaults)}): ").strip()
    if not raw:
        return defaults
    return _parse_team(raw)


def _qualified_from_results(t: Tournament) -> dict[str, list[str]]:
    qualified: dict[str, list[str]] = {}
    for gname, stats in t.results["groups"].items():
        ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["points"], kv[0]))
        qualified[gname] = [name for name, _ in ranked]
    return qualified


def _contestants_from_args(args: argparse.Namespace) -> list[Contestant]:
    """Build contestant list. --model may be given multiple times.
    Remaining slots are filled with baseline agents on random teams."""
    contestants: list[Contestant] = []
    rng = random.Random(args.seed)
    for i, spec in enumerate(args.model or []):
        if spec.startswith("checkpoint:"):
            from .registry import contestant_from_checkpoint
            contestants.append(contestant_from_checkpoint(spec.split(":", 1)[1]))
            continue
        # spec format: Name=teamA,teamB,teamC,teamD[:gear-spec]
        name, _, rest = spec.partition("=")
        team_spec, _, gear_spec = rest.partition(":")
        team = _parse_team(team_spec)
        gear = _parse_gear(gear_spec or None)
        agent_name = args.agent_for_named or "greedy"
        agent_cls = {"random": RandomAgent, "greedy": GreedyAgent}.get(
            agent_name, GreedyAgent)
        contestants.append(Contestant(
            name=name, team=team, gear=gear,
            agent_factory=(lambda cls=agent_cls, s=i: cls(seed=args.seed + s)),
        ))
    # fill with baselines to reach the requested contestant count
    roster = db.character_ids() + db.enemy_ids()
    while len(contestants) < args.contestants:
        i = len(contestants)
        name = f"{'Random' if i % 2 == 0 else 'Greedy'}-{i // 2 + 1}"
        team = rng.sample(roster, min(args.team_size, len(roster)))
        agent_cls = RandomAgent if i % 2 == 0 else GreedyAgent
        contestants.append(Contestant(
            name=name, team=team,
            agent_factory=(lambda cls=agent_cls, s=i: cls(seed=args.seed + s)),
            seed=args.seed + i,
        ))
    return contestants


def cmd_tournament_cli(args: argparse.Namespace) -> None:
    contestants = _contestants_from_args(args)
    n = len(contestants)
    if n % args.group_size != 0:
        raise SystemExit(f"contestant count ({n}) must be a multiple of --group-size")
    t = Tournament(contestants, group_size=args.group_size,
                   advance_per_group=args.advance, seed=args.seed,
                   best_of=args.best_of, name="cli-tournament",
                   max_av=args.max_av)
    t.run_groups()
    t.run_knockout(_qualified_from_results(t))
    print(format_tournament_report(t.results))


def cmd_train(args: argparse.Namespace) -> None:
    """Train MaskablePPO (default) or MaskableDQN and save a loadable bundle."""
    from .rl.rewards import RewardConfig
    from .rl.train import TrainConfig, train
    from .registry import ModelMetadata, save_model_bundle

    team_a = _parse_team(args.team_a)
    team_b = _parse_team(args.team_b)
    # checkpoint defaults: cpu-style short dry-runs unless overridden; on a
    # CUDA box (device=auto -> cuda) scale up so a full run needs no flags.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/{args.algo}_run"
    if args.checkpoint_freq is None:
        args.checkpoint_freq = 200_000 if resolve_device(args.device) == "cuda" else 25_000
    config = TrainConfig(
        algo=args.algo,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        self_play_prob=args.self_play,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        verbose=1 if not args.quiet else 0,
        seed=args.seed,
        device=args.device,
    )
    model, _env = train(team_a, team_b, config)
    out_dir = args.out or f"checkpoints/{args.name}"
    meta = ModelMetadata(
        name=args.name,
        algo=config.algo,
        team=team_a,
        level=80,
        max_av=3000.0,
        max_rounds=80,
        trained_timesteps=args.timesteps,
    )
    path = save_model_bundle(model, meta, out_dir)
    print(f"Model saved to: {path}")
    print(f"Fight it with:  python main.py fight --team-a {','.join(team_a)} "
          f"--team-b <ids> --agent-a model:{path}")


def cmd_watch(args: argparse.Namespace) -> None:
    roster = db.character_ids()
    rng = random.Random(args.seed)
    team_a = rng.sample(roster, 4)
    team_b = rng.sample(roster, 4)
    print("Team A:", ", ".join(team_a))
    print("Team B:", ", ".join(team_b))
    battle = run_battle_between(GreedyAgent(seed=args.seed), GreedyAgent(seed=args.seed + 1),
                                team_a, team_b, seed=args.seed, verbose=True,
                                max_av=args.max_av)
    print("=" * 60)
    print(f"Winner: {battle.winner}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli-hsr",
                                description="Honkai: Star Rail CLI combat simulator & RL environment")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list database units and gear")
    sp.add_argument("--type", choices=["character", "enemy", "gear", "all"], default="all")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("fight", help="run a battle: agent/model vs agent/model")
    sp.add_argument("--team-a", required=True, help="comma separated unit ids")
    sp.add_argument("--team-b", required=True)
    sp.add_argument("--agent-a", default="greedy",
                    help="random | greedy | human | model:<checkpoint-dir>")
    sp.add_argument("--agent-b", default="greedy")
    sp.add_argument("--gear-a", default=None, help="gear spec per slot, ';'-separated")
    sp.add_argument("--gear-b", default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--max-av", type=float, default=4000)
    sp.add_argument("--max-rounds", type=int, default=100)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_fight)

    sp = sub.add_parser("tournament", help="run a tournament between models/agents")
    sp.add_argument("--model", action="append", default=[],
                    help="'checkpoint:<dir>' or 'Name=team1,team2:gear-spec' (repeatable)")
    sp.add_argument("--contestants", type=int, default=4)
    sp.add_argument("--group-size", type=int, default=4)
    sp.add_argument("--advance", type=int, default=2)
    sp.add_argument("--team-size", type=int, default=4)
    sp.add_argument("--best-of", type=int, default=1)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-av", type=float, default=4000)
    sp.set_defaults(func=cmd_tournament_cli)

    sp = sub.add_parser("train", help="train MaskablePPO/DQN on a matchup")
    sp.add_argument("--team-a", required=True)
    sp.add_argument("--team-b", required=True)
    sp.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    sp.add_argument("--timesteps", type=int, default=100_000)
    sp.add_argument("--n-envs", type=int, default=1)
    sp.add_argument("--self-play", type=float, default=0.2,
                    help="probability of sampling the current policy as opponent")
    sp.add_argument("--name", default="model")
    sp.add_argument("--out", default=None, help="bundle output dir (default checkpoints/<name>)")
    sp.add_argument("--device", default="auto",
                    help="auto (cuda when available, else cpu) | cuda | cpu | ...")
    sp.add_argument("--log-dir", default="runs/")
    sp.add_argument("--checkpoint-dir", default=None,
                    help="intermediate checkpoint dir (default checkpoints/<algo>_run)")
    sp.add_argument("--checkpoint-freq", type=int, default=None,
                    help="env steps between checkpoints (default 25000; 0 disables)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("watch", help="watch a random verbose demo battle")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--max-av", type=float, default=4000)
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("duel", help="anyone-vs-anyone duel with per-unit levels")
    sp.add_argument("--side-a", required=True,
                    help="spec: unit[:level],... e.g. seele:80 or cocolia_boss:60")
    sp.add_argument("--side-b", required=True)
    sp.add_argument("--agent-a", default="greedy", help="random | greedy | human | model:<dir>")
    sp.add_argument("--agent-b", default="greedy")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--max-av", type=float, default=4000)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_duel)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db.ensure_seeded()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
