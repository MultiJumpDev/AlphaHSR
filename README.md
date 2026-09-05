# cli-hsr — Honkai: Star Rail combat simulator & RL environment

A faithful CLI reproduction of Honkai: Star Rail's turn-based combat, wrapped
as a [Gymnasium](https://gymnasium.farama.org/) environment so that RL models
can **train (self-play, PPO/DQN)**, **compete in tournaments**, and **duel
humans** — all from the command line or a marimo notebook.

## Game systems reproduced

| System | Rules implemented |
|---|---|
| Turn order | Action Value = 10000/SPD, lowest-AV unit acts; action advance/delay; enemy SPD per wiki |
| Damage | Exact wiki chain: Base DMG → CRIT → DMG Boost → Weaken → DEF → RES → Vulnerability → Mitigation → Broken ×0.9; DEF Mult = 1 − DEF/(DEF + 200 + 10·LvAtk) |
| Toughness | Weakness-gated depletion, per-element Break DMG (2/2/1/1/1.5/0.5/0.5 × Level Multiplier × Max-Toughness Mult), 25% delay, 150% base chance debuff |
| Break debuffs | Bleed (HP-scaled, capped), Burn, Freeze (skip + next turn at 50% AV), Shock, Wind Shear (1/3 stacks, max 5), Entanglement (stacking + delay), Imprisonment (delay + SPD −10%) |
| Energy | 20/30/5 gains, +10 on kill, on-hit energy, Energy Regeneration Rate |
| Skill Points | +1 on Basic, −1 on Skill, shared per team |
| Bosses | Multi-phase: fresh HP/toughness + immediate action on transition |
| Levels | Exact per-level stats: datamine promotion curves for characters (verified against wiki tables), Level-Multiplier scaling for enemies (1–90) |
| Roster | **98 characters** (every playable character, all 7 elements, all paths — 15 hand-crafted faithful kits + 83 auto-imported) + 23 enemies (8 normal / 5 elite / 10 boss, incl. Cocolia, Argenti, Phantylia, Swarm, Yanqing, Aventurine bosses) |
| Gear | Light Cones (S1→S5 interpolation) and Relic Sets (2pc/4pc bonuses) |

## Install

```bash
uv sync --extra all     # or: pip install -e ".[all]"
```

Extras: `rl` (gymnasium + stable-baselines3 + sb3-contrib), `test`, `dev`.

## CLI usage

```bash
python main.py list                      # units + gear catalog
python main.py fight --team-a seele,bronya,sparkle,fu_xuan \
                     --team-b kafka,black_swan,luocha,himeko --agent-a greedy
python main.py fight --team-a seele --team-b blaze_out_of_space --agent-a human
python main.py watch                     # verbose demo battle
```

### Anyone-vs-anyone duels (new)

Any unit can fight any unit — characters, mobs, elites, bosses — at any
level, mixed teams allowed:

```bash
python main.py duel --side-a seele:80 --side-b cocolia_boss:80
python main.py duel --side-a cocolia_boss:60 --side-b seele:50 --agent-b random
python main.py duel --side-a seele:80 --side-b dan_heng:80 --agent-a human
```

Character stats resolve from the exact datamine promotion curves (see
`cli_hsr/levels.py`), enemy stats scale with the Level Multiplier, so a
Lv.40 unit against a Lv.80 one is a real David-vs-Goliath fight.

### Draft environment (new)

`DraftEnv` (`cli_hsr/draft.py`) prepends a **draft phase** to the battle:
the learning agent first picks its fighters from the whole roster (unit ×
level-bucket actions, with counter-pick by the opponent), then the fight
runs with the usual masked action space. This trains the counter-pick meta
jointly with in-combat play:

```python
from cli_hsr.draft import DraftEnv

env = DraftEnv(seed=0)
obs, info = env.reset()            # phase: "draft"
obs, r, term, trunc, info = env.step(action)   # pick a fighter + level
# info["phase"] == "battle" once the draft completes
```

### Re-importing the datamine roster

```bash
# clone the datamine, then:
#   git clone --filter=blob:none --sparse https://github.com/Mar-7th/StarRailRes
#   cd StarRailRes && git sparse-checkout set index_new/en
uv run python tools/import_starrailres.py StarRailRes/index_new/en --skill-level 10
```

### Gear (Phase 4)

```bash
# per-slot gear spec, ';'-separated: cone=superimposition,set=pieces,set=pieces
python main.py fight --team-a seele,bronya --team-b kafka,himeko \
    --gear-a "sleep_like_the_dead=5,inert_salsotto=2;swordplay=3,musketeer_of_wild_wheat=4"
```

## RL training (Phase 1)

```bash
# MaskablePPO (default) or MaskableDQN (custom masked DQN)
python main.py train --team-a seele,bronya --team-b kafka,himeko \
    --algo ppo --timesteps 200000 --name my_ppo
```

Programmatic API:

```python
from cli_hsr.rl.train import TrainConfig, train, evaluate_policy

config = TrainConfig(algo="ppo", total_timesteps=1_000_000, self_play_prob=0.2)
model, env = train(["seele", "bronya"], ["kafka", "himeko"], config)
stats = evaluate_policy(model, ["seele", "bronya"], ["kafka", "himeko"], n_games=100)
model.save("my_model")
```

- **Reward function** (`cli_hsr/rl/rewards.py`): configurable event-driven
  shaping (damage dealt/taken, heals, shields, breaks, kills, ults, SP) +
  terminal win/loss/draw bonuses, with per-turn clipping.
- **Action masking**: `env.action_masks()` (SB3 convention) + `env.action_mask`;
  illegal actions are impossible for masked policies.
- **Self-play**: `SelfPlayOpponent` samples each episode from
  {random, greedy, current policy}; raise `self_play_prob` as the model improves.
- **Perspective invariant**: observations always place the learning side in
  the first slots, so one policy fights on either side.

## Checkpoints & tournaments (Phase 2)

`python main.py train` saves a *bundle* under `checkpoints/<name>/`:
`model.zip` + `metadata.json` (team, gear, algo, timesteps, eval stats).

```bash
# duel a checkpoint against an agent
python main.py fight --team-a seele,bronya --team-b kafka,himeko \
    --agent-a "model:checkpoints/my_ppo" --agent-b greedy

# tournaments: mix checkpoints and named teams; remaining slots auto-fill
# with baselines
python main.py tournament --contestants 4 \
    --model "checkpoint:checkpoints/my_ppo" \
    --model "FireflyTeam=firefly,fu_xuan,luocha,himeko"
```

Programmatic: `contestant_from_checkpoint(path)` returns a tournament-ready
`Contestant` (model metadata drives team + gear).

Tournament format: groups of 4 (round robin, 3/1/0 points) → top 2 advance →
knockout bracket (QF → SF → Final, auto-named). All results persist to
SQLite (`match_results`, `tournament_results`).

## Training on molab (marimo)

[`notebooks/train_molab.py`](notebooks/train_molab.py) is a single-click
marimo notebook ready for [molab](https://molab.marimo.io) **and** local use:

- **Auto-setup** — the first cell headlessly installs `gymnasium`,
  `stable-baselines3`, `sb3-contrib`, `tensorboard` (+ `torch`) and the
  `cli-hsr` package itself (editable install from the workspace), preferring
  `uv pip` and falling back to pip. No manual input needed.
- **Hardware-aware defaults** — detects CUDA via `torch.cuda.is_available()`.
  GPU (molab): 3,000,000 steps, checkpoints every 200,000, 2 parallel envs.
  CPU (local): 50,000-step dry-run with checkpoints every 25,000.
- **MaskablePPO** — `MaskableActorCriticPolicy` on the masked `HSREnv`
  (`env.action_masks()`), with `CheckpointCallback` writing intermediate
  checkpoints to `checkpoints/ppo_molab/`.
- **Bundle export + download** — after training, the model is packaged as a
  registry-compatible bundle (`model.zip` + `metadata.json`) and offered
  through a non-blocking `mo.download` widget, alongside win-rate evaluation
  vs Greedy and a sample battle.

Run it with `marimo edit notebooks/train_molab.py`, or upload the repo to
molab and open the file — everything installs and runs from scratch.

## Architecture

```
src/cli_hsr/
  constants.py     # exact wiki formulas & tables (level multipliers, break, AV)
  db.py            # SQLite: units + cones + relic sets + match/tournament results
  statuses.py      # buffs/debuffs/DoTs/CC with game-accurate decay rules
  units.py         # Unit model: dynamic stats (incl. gear), shields, toughness, AV
  engine.py        # battle engine: AV loop, damage, break, energy, phases, FUAs
  env.py           # Gymnasium env: masked Discrete(32), perspective obs, rewards
  gear.py          # loadouts: cones + relic sets -> stats + battle-start passives
  agents.py        # Random / Greedy / Human + CheckpointAgent-compatible interface
  tournament.py    # groups of 4 + knockout, persisted, gear-aware
  registry.py      # checkpoint bundles (model.zip + metadata.json)
  rl/
    rewards.py     # RewardConfig / event-driven reward function
    train.py       # PPO/DQN training, self-play opponent, evaluation
    maskable_dqn.py# masked DQN (sb3-contrib has no MaskableDQN)
  cli.py           # argparse CLI (list/fight/train/tournament/watch)
data/              # characters.json, characters_new.json, enemies.json,
                   # light_cones.json, relics.json -> game.db
notebooks/         # marimo/molab training notebook
tests/             # 63 pytest tests
```

## Balance note

Simulator units field **true base stats** (no relics/light cones), so the
lore-accurate enemy values from `data/enemies.json` are scaled down by
category knobs in `cli_hsr/constants.py` — `ENEMY_HP_SCALE` (0.12 normal /
0.05 elite / 0.045 boss) and `ENEMY_ACTION_POWER_SCALE` (0.25, applied only
when an enemy attacks a *character*; enemy-vs-enemy duels fight at full lore
mults). Raise the HP scales toward 1.0 as gear/trace support matures. Enemy
SPD values in the JSON are final in-game speeds (no further level
multiplier).

## Extending

- **Units**: add entries to `data/characters_new.json` / `enemies.json`
  (schema identical to the base files) — the DB reseeds automatically.
- **Gear**: add cones/sets to `light_cones.json` / `relics.json`; cone
  passives interpolate linearly S1→S5 via `value`/`value_s5`.
- **Effects**: the engine's effect vocabulary covers `buff/debuff/dot/cc/
  action_advance/detonate_dots/implant_weakness/follow_up/burst_extra/
  execute_bonus/def_debuff/res_debuff/dot_boost/energy_to_target/delay/
  revive_heal/ally_shield_all/…`.

## Tests

```bash
uv run pytest -q        # 38 passed
```
