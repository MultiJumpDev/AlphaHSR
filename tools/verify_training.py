"""Verify RL training: measure win-rate progression at checkpoints vs baselines.

Usage:
    uv run python tools/verify_training.py --algo dqn --timesteps 20000 --checkpoints 5
    uv run python tools/verify_training.py --algo ppo --timesteps 50000 --checkpoints 4

The script trains on a fixed matchup, evaluating the current policy against
Random and Greedy baselines at regular intervals. Progression = win rate
should climb over time; a policy that saturates at 100% within the first
checkpoint suggests the matchup is too easy (see PART 2 balance work).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cli_hsr.rl.train import TrainConfig, train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="training progression verifier")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--checkpoints", type=int, default=5)
    parser.add_argument("--n-games", type=int, default=12)
    parser.add_argument("--team-a", default="seele,bronya")
    parser.add_argument("--team-b", default="kafka,himeko")
    parser.add_argument("--max-av", type=float, default=2500.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    team_a = [s.strip() for s in args.team_a.split(",")]
    team_b = [s.strip() for s in args.team_b.split(",")]
    total = args.timesteps
    per = total // args.checkpoints

    from cli_hsr.rl.train import evaluate_policy

    config = TrainConfig(
        algo=args.algo, total_timesteps=per, verbose=0, seed=args.seed,
        log_dir="runs/", n_steps=min(512, max(64, per // 4)),
        batch_size=64, learning_starts=200 if args.algo == "dqn" else 0,
    )

    print(f"Training {args.algo.upper()} on {team_a} vs {team_b} "
          f"({args.checkpoints} x {per} steps)")
    print("-" * 68)
    header = f"{'steps':>8} | {'vs Random W/D/L':>17} | {'vs Greedy W/D/L':>17} | {'secs':>6}"
    print(header)
    print("-" * 68)

    rows: list[dict] = []
    model = None
    t0 = time.time()
    for ck in range(1, args.checkpoints + 1):
        model, _ = train(team_a, team_b, config,
                         opponent_model=model,
                         env_kwargs={"max_av": args.max_av, "max_rounds": 60})
        steps_done = ck * per
        r = evaluate_policy(model, team_a, team_b, n_games=args.n_games,
                            opponent="random", seed=1000 + ck,
                            env_kwargs={"max_av": args.max_av, "max_rounds": 60})
        g = evaluate_policy(model, team_a, team_b, n_games=args.n_games,
                            opponent="greedy", seed=2000 + ck,
                            env_kwargs={"max_av": args.max_av, "max_rounds": 60})
        secs = time.time() - t0
        print(f"{steps_done:>8} | {r['wins']:>3}/{r['draws']}/{r['losses']:>2} "
              f"({r['win_rate']:>5.0%}) | {g['wins']:>3}/{g['draws']}/{g['losses']:>2} "
              f"({g['win_rate']:>5.0%}) | {secs:>6.0f}")
        rows.append({"steps": steps_done, "vs_random": r, "vs_greedy": g})

    print("-" * 68)
    rw_first = rows[0]["vs_random"]["win_rate"]
    rw_last = rows[-1]["vs_random"]["win_rate"]
    rg_last = rows[-1]["vs_greedy"]["win_rate"]
    print(f"vs Random: {rw_first:.0%} -> {rw_last:.0%} | final vs Greedy: {rg_last:.0%}")
    if rw_last >= 0.99 and rows[0]["vs_random"]["win_rate"] >= 0.9:
        print("WARNING: policy dominates from the first checkpoint -> matchup "
              "too easy; increase ENEMY_HP_SCALE or use harder opponents.")
    elif rw_last > rw_first or rg_last > 0.5:
        print("OK: policy improves and/or holds its own against Greedy.")
    else:
        print("INCONCLUSIVE: no clear progression - try more timesteps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
