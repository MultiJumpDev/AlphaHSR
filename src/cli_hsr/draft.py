"""Draft & duel: the model picks ANY fighter (character, mob, elite, boss)
at ANY level, then fights.

Two ways to use it:

1. ``run_duel(agent_a, agent_b, pick_a, pick_b)`` — plain battles where each
   side is described by a ``FighterChoice`` (unit id + level, any unit type,
   mixed teams allowed).

2. ``DraftEnv`` — a Gymnasium environment that PREPENDS the draft to the
   battle: the learning agent first picks its own fighters (from the whole
   roster), then the opponent counter-picks, then the fight runs with the
   usual masked action space. This trains "who beats what" — the counter-pick
   meta — jointly with in-combat play.

Fighter encoding in the draft action space: every unit in the DB becomes
``N_LEVELS`` draft actions (unit x level bucket). Levels bucket into
{40, 55, 70, 80} by default to keep the action space small; exact levels can
always be used through ``FighterChoice`` directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import db
from . import levels as LV
from .agents import GreedyAgent, RandomAgent
from .engine import Battle
from .env import HSREnv, MAX_ACTIONS, OBS_DIM, battle_to_observation
from .rl.rewards import RewardConfig, RewardShaper

try:  # pragma: no cover - gymnasium may be absent
    import gymnasium as gym
    _GymEnvBase = gym.Env
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _GymEnvBase = object
    _HAS_GYM = False

DEFAULT_LEVEL_BUCKETS: tuple[int, ...] = (40, 55, 70, 80)


# --------------------------------------------------------------------------- #
# Fighter choices                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class FighterChoice:
    """One side of a duel: unit ids + their levels (any unit types)."""

    units: list[str]
    levels: list[int] | None = None
    gear: list[dict] | None = None
    level: int = 80  # fallback when per-unit levels are absent

    def resolved_levels(self) -> list[int]:
        n = len(self.units)
        if self.levels:
            return [int(self.levels[i % len(self.levels)]) for i in range(n)]
        return [self.level] * n


def list_fighters(db_path: str | None = None) -> list[dict[str, Any]]:
    """Every combatant in the game, with draft metadata.

    Characters get a self-element weakness injected so they can be attacked
    with toughness damage when fielded as the enemy side (characters in the
    real game have no toughness bar; this keeps anyone-vs-anyone fights
    mechanically complete).
    """
    out = []
    for row in db.list_units(db_path=db_path):
        kit = row["kit_json"]
        entry = {
            "id": row["id"],
            "name": row["name"],
            "unit_type": row["unit_type"],
            "category": row.get("category") or row["unit_type"],
            "element": row["element"],
            "path": row.get("path"),
            "rarity": row.get("rarity"),
            "has_level_curve": LV.has_promotion_curve(row["stats_json"]),
            # usable offensive kit? (draft-pool filter for include_supports)
            "has_damage": bool(
                row["unit_type"] != "character"
                or kit.get("basic") or kit.get("skill") or kit.get("ultimate")
            ),
        }
        if row["unit_type"] == "character":
            entry["weaknesses"] = [row["element"]]
        else:
            entry["weaknesses"] = list(kit.get("weaknesses", []))
        out.append(entry)
    return out


def _rows_for(choice: FighterChoice, db_path: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for i, uid in enumerate(choice.units):
        row = db.get_unit(uid, db_path)
        if row is None:
            raise ValueError(f"unknown unit id: {uid}")
        if row["unit_type"] == "character":
            # let any character be fought against (toughness/weakness system)
            kit = row["kit_json"]
            kit.setdefault("weaknesses", [row["element"]])
        if choice.gear and i < len(choice.gear) and choice.gear[i]:
            from .gear import apply_loadout_to_row
            row = apply_loadout_to_row(row, choice.gear[i])
        rows.append(row)
    return rows


def make_battle(
    pick_a: FighterChoice,
    pick_b: FighterChoice,
    name_a: str = "Side A",
    name_b: str = "Side B",
    max_av: float = 4000.0,
    rng: random.Random | None = None,
    db_path: str | None = None,
) -> Battle:
    """Build a battle from two FighterChoices (per-unit levels respected)."""
    return Battle(
        _rows_for(pick_a, db_path),
        _rows_for(pick_b, db_path),
        name_a=name_a, name_b=name_b,
        levels_a=pick_a.resolved_levels(),
        levels_b=pick_b.resolved_levels(),
        max_av=max_av,
        rng=rng or random.Random(),
    )


def run_duel(
    agent_a,
    agent_b,
    pick_a: FighterChoice,
    pick_b: FighterChoice,
    max_av: float = 4000.0,
    seed: int = 0,
    db_path: str | None = None,
) -> Battle:
    """Play one anyone-vs-anyone battle between two chooser-style agents."""

    def chooser(agent):
        return lambda b, actor, legal: agent.choose(b, actor, legal)

    battle = make_battle(pick_a, pick_b, rng=random.Random(seed), db_path=db_path)
    winner = battle.run(chooser(agent_a), chooser(agent_b))
    agent_a.end_battle(winner == "A", winner == "draw")
    agent_b.end_battle(winner == "B", winner == "draw")
    return battle


# --------------------------------------------------------------------------- #
# Draft environment                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class DraftConfig:
    """Draft phase configuration."""

    team_size: int = 1
    level_buckets: tuple[int, ...] = DEFAULT_LEVEL_BUCKETS
    # opponent draft behaviour: "counter" (greedy vs your pick) or "random"
    opponent_draft: str = "counter"
    # exclude units with no usable damage kit from the draft pool
    include_supports: bool = True


class DraftEnv(_GymEnvBase):
    """Draft + battle environment (RL: counter-pick meta).

    Action space (Discrete):
        - phase "draft":  len(pool) * len(level_buckets) actions
                          = pick pool[i] at level_buckets[j]
        - phase "battle": the usual MAX_ACTIONS masked combat actions

    The full draft action space is exposed for indexing; use
    ``draft_action_masks()`` during the draft phase (masks already-picked
    units) and ``action_masks()`` during battle.

    Episode flow:
        agent picks ``team_size`` fighters -> opponent counter-picks ->
        battle runs with the standard HSREnv semantics (reward shaping,
        perspective-correct observations).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_policy: Callable[[Battle, Any, list[dict[str, Any]]], dict[str, Any]] | None = None,
        agent_side: str = "A",
        max_av: float = 4000.0,
        max_rounds: int = 100,
        reward_shaper: RewardShaper | None = None,
        reward_config: RewardConfig | None = None,
        seed: int | None = None,
        db_path: str | None = None,
        draft_config: DraftConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        if not _HAS_GYM:
            raise ImportError("gymnasium is required for DraftEnv: pip install gymnasium")
        super().__init__()
        self.agent_side = agent_side
        self.opponent_policy = opponent_policy
        self.max_av = max_av
        self.max_rounds = max_rounds
        self.reward_shaper = reward_shaper or RewardShaper(reward_config or RewardConfig())
        self.db_path = db_path
        self.draft_config = draft_config or DraftConfig()
        self.render_mode = render_mode
        self.rng = random.Random(seed)

        self.pool = list_fighters(db_path)
        if not self.draft_config.include_supports:
            self.pool = [f for f in self.pool if f["has_damage"]]
        self.n_pool = len(self.pool)
        self.n_level_buckets = len(self.draft_config.level_buckets)

        # draft actions = pool x level buckets; battle actions = MAX_ACTIONS
        self.draft_action_space_size = self.n_pool * self.n_level_buckets
        max_action_space = max(self.draft_action_space_size, MAX_ACTIONS)
        import gymnasium.spaces as spaces
        self.action_space = spaces.Discrete(max_action_space)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.phase = "draft"
        self.my_picks: list[FighterChoice] = []
        self.opp_picks: list[FighterChoice] = []
        self._battle_env: HSREnv | None = None
        self.action_mask = np.zeros(max_action_space, dtype=bool)
        self._refresh_draft_mask()

    # ------------------------------------------------------------------ #
    # masks                                                              #
    # ------------------------------------------------------------------ #
    def _refresh_draft_mask(self) -> None:
        mask = np.zeros(self.action_space.n, dtype=bool)
        taken = {p.units[0] for p in self.my_picks} | {p.units[0] for p in self.opp_picks}
        for i, f in enumerate(self.pool):
            if f["id"] in taken:
                continue
            for j in range(self.n_level_buckets):
                mask[i * self.n_level_buckets + j] = True
        self.action_mask = mask

    def draft_action_masks(self) -> list[bool]:
        return self.action_mask.astype(bool).tolist()

    def action_masks(self) -> list[bool]:
        if self.phase == "draft":
            return self.draft_action_masks()
        assert self._battle_env is not None
        masks = self._battle_env.action_masks()
        out = np.zeros(self.action_space.n, dtype=bool)
        out[:len(masks)] = np.array(masks, dtype=bool)
        return out.tolist()

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _decode_draft_action(self, action: int) -> tuple[dict[str, Any], int]:
        idx = action // self.n_level_buckets
        li = action % self.n_level_buckets
        return self.pool[idx], self.draft_config.level_buckets[li]

    def _opponent_draft_pick(self) -> tuple[str, int]:
        """Simple counter-pick: prefer same-element advantage vs our pick."""
        cfg = self.draft_config
        taken = {p.units[0] for p in self.my_picks} | {p.units[0] for p in self.opp_picks}
        candidates = [f for f in self.pool if f["id"] not in taken]
        level = cfg.level_buckets[self.rng.randrange(len(cfg.level_buckets))]
        if cfg.opponent_draft == "random" or not candidates or not self.my_picks:
            f = self.rng.choice(candidates or self.pool)
            return f["id"], level
        # counter: pick a fighter that is strong INTO the opponent's last pick
        # (their element hits our weakness) — approximated by element advantage
        my_last = self.my_picks[-1].units[0]
        my_row = db.get_unit(my_last, self.db_path)
        my_elem = my_row["element"] if my_row else "Physical"
        elem_adv = {"Physical": "Quantum", "Fire": "Ice", "Ice": "Fire",
                    "Thunder": "Wind", "Wind": "Thunder", "Quantum": "Imaginary",
                    "Imaginary": "Physical"}
        weak_to_us = elem_adv.get(my_elem)
        counters = [f for f in candidates if f.get("element") == weak_to_us]
        f = self.rng.choice(counters) if counters else self.rng.choice(candidates)
        return f["id"], level

    def _finalize_teams(self) -> tuple[list[str], list[int], list[str], list[int]]:
        my = FighterChoice(units=[p.units[0] for p in self.my_picks],
                           levels=[p.resolved_levels()[0] for p in self.my_picks])
        opp = FighterChoice(units=[p.units[0] for p in self.opp_picks],
                            levels=[p.resolved_levels()[0] for p in self.opp_picks])
        if self.agent_side == "A":
            return my.units, my.levels, opp.units, opp.levels
        return opp.units, opp.levels, my.units, my.levels

    def _make_battle_env(self) -> HSREnv:
        team_a, levels_a, team_b, levels_b = self._finalize_teams()
        env = HSREnv(
            team_a, team_b,
            opponent_policy=self.opponent_policy,
            agent_side=self.agent_side,
            max_av=self.max_av,
            max_rounds=self.max_rounds,
            reward_shaper=self.reward_shaper,
            seed=self.rng.randint(0, 2**31 - 1),
            db_path=self.db_path,
        )
        # per-unit levels from the draft
        env.levels_a = levels_a
        env.levels_b = levels_b
        return env

    # ------------------------------------------------------------------ #
    # Gym API                                                            #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.phase = "draft"
        self.my_picks = []
        self.opp_picks = []
        self._battle_env = None
        self._refresh_draft_mask()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[-1] = 1.0
        return obs, {"phase": "draft", "pool_size": self.n_pool}

    def step(self, action: int):
        if self.phase == "draft":
            return self._step_draft(action)
        return self._step_battle(action)

    def _step_draft(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not (0 <= action < self.draft_action_space_size) or not self.action_mask[action]:
            return self._observation(), -1.0, False, False, {"phase": "draft", "illegal": True}
        fighter, level = self._decode_draft_action(action)
        self.my_picks.append(FighterChoice(units=[fighter["id"]], levels=[level]))

        done_drafting = len(self.my_picks) >= self.draft_config.team_size
        if done_drafting:
            for _ in range(self.draft_config.team_size):
                oid, olvl = self._opponent_draft_pick()
                self.opp_picks.append(FighterChoice(units=[oid], levels=[olvl]))
            self.phase = "battle"
            self._battle_env = self._make_battle_env()
            obs, info = self._battle_env.reset()
            self.reward_shaper.reset(self._battle_env.battle)
            info = {**info, "phase": "battle",
                    "my_picks": [p.units[0] for p in self.my_picks],
                    "opp_picks": [p.units[0] for p in self.opp_picks],
                    "my_levels": [p.resolved_levels()[0] for p in self.my_picks],
                    "opp_levels": [p.resolved_levels()[0] for p in self.opp_picks]}
            return obs, 0.0, False, False, info

        self._refresh_draft_mask()
        return self._observation(), 0.0, False, False, {"phase": "draft"}

    def _step_battle(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._battle_env is not None
        n = self._battle_env.action_space.n
        if action >= n:
            action = action % n
        obs, reward, term, trunc, info = self._battle_env.step(action)
        info = {**info, "phase": "battle"}
        return obs, float(reward), bool(term), bool(trunc), info

    def _observation(self) -> np.ndarray:
        if self._battle_env is not None and self.phase == "battle":
            return self._battle_env._observation()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[-1] = 1.0
        return obs

    def render(self):  # pragma: no cover
        if self.phase == "draft":
            picks = ", ".join(p.units[0] for p in self.my_picks) or "(none)"
            return f"[draft] my picks: {picks}"
        assert self._battle_env is not None
        return self._battle_env.render()

    def close(self) -> None:  # pragma: no cover
        if self._battle_env is not None:
            self._battle_env.close()
