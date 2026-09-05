"""Gymnasium environment wrapping the HSR battle engine for RL training.

Key properties
--------------
- **Discrete action space with masks**: the current actor's legal
  (action, target) pairs are exposed through `env.action_mask` and
  `env.action_masks()` (SB3 MaskablePPO compatible).
- **Perspective invariant**: observations always put the learning side in
  slots first ("my team" then "enemy team"), so a policy trained on side A
  plays side B without retraining.
- **Injectable rewards**: pass a `RewardShaper`/callable `(battle) -> float`
  or let the default `RewardConfig` shaping handle it.
- **Configurable format**: any team size 1..5, level, AV limits.

Gymnasium is an optional dependency.
"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np

from . import db
from .engine import Battle

try:  # pragma: no cover - gymnasium may be absent
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _HAS_GYM = False

from .rl.rewards import RewardConfig, RewardShaper  # noqa: E402

_GymEnvBase = gym.Env if _HAS_GYM else object

MAX_TEAM_SLOTS = 5
MAX_ACTIONS = 32

FEATURES_PER_UNIT = 16  # hp, energy, shield, tough, broken, alive, spd, av + 7 elements + pad
OBS_DIM = MAX_TEAM_SLOTS * FEATURES_PER_UNIT * 2 + 3

ELEMENTS = ["Physical", "Fire", "Ice", "Lightning", "Wind", "Quantum", "Imaginary"]


def unit_features(u) -> np.ndarray:
    feat = np.zeros(FEATURES_PER_UNIT, dtype=np.float32)
    feat[0] = u.hp_fraction()
    feat[1] = u.energy / u.base.energy_max if u.base.energy_max else 0.0
    feat[2] = min(1.0, u.shield / max(1.0, u.max_hp))
    feat[3] = u.toughness / u.max_toughness if u.max_toughness else 0.0
    feat[4] = 1.0 if u.weakness_broken else 0.0
    feat[5] = 1.0 if u.alive else 0.0
    feat[6] = u.effective_spd() / 200.0
    feat[7] = min(1.0, u.av / 250.0)
    if u.element in ELEMENTS:
        feat[8 + ELEMENTS.index(u.element)] = 1.0
    return feat


def battle_to_observation(battle: Battle, agent_side: str) -> np.ndarray:
    """Flat observation from `agent_side`'s perspective (allies first)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    own = battle.side_a if agent_side == "A" else battle.side_b
    foe = battle.side_b if agent_side == "A" else battle.side_a
    for i, u in enumerate(own.units[:MAX_TEAM_SLOTS]):
        obs[i * FEATURES_PER_UNIT:(i + 1) * FEATURES_PER_UNIT] = unit_features(u)
    off = MAX_TEAM_SLOTS * FEATURES_PER_UNIT
    for i, u in enumerate(foe.units[:MAX_TEAM_SLOTS]):
        obs[off + i * FEATURES_PER_UNIT: off + (i + 1) * FEATURES_PER_UNIT] = unit_features(u)
    obs[-3] = battle.skill_points / 5.0
    obs[-2] = min(1.0, battle.time / battle.max_av)
    obs[-1] = 1.0
    return obs


class HSREnv(_GymEnvBase):
    """Self-play-ready Gym environment. One learning side (`agent_side`)
    is controlled by the RL policy; the other side by `opponent_policy`."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        team_a: list[str],
        team_b: list[str],
        opponent_policy: Callable[[Battle, Any, list[dict[str, Any]]], dict[str, Any]] | None = None,
        agent_side: str = "A",
        level: int = 80,
        max_av: float = 4000.0,
        max_rounds: int = 100,
        reward_shaper: RewardShaper | None = None,
        reward_config: RewardConfig | None = None,
        seed: int | None = None,
        db_path: str | None = None,
        render_mode: str | None = None,
        gear_a: list[dict] | None = None,
        gear_b: list[dict] | None = None,
        levels_a: list[int] | None = None,
        levels_b: list[int] | None = None,
    ) -> None:
        if not _HAS_GYM:
            raise ImportError("gymnasium is required for HSREnv: pip install gymnasium")
        super().__init__()
        assert agent_side in ("A", "B")
        self.agent_side = agent_side
        self.team_a_ids = team_a
        self.team_b_ids = team_b
        self.opponent_policy = opponent_policy
        self.level = level
        self.max_av = max_av
        self.max_rounds = max_rounds
        self.reward_shaper = reward_shaper or RewardShaper(reward_config or RewardConfig())
        self.db_path = db_path
        self.render_mode = render_mode
        self.gear_a = gear_a
        self.gear_b = gear_b
        self.levels_a = levels_a
        self.levels_b = levels_b
        self.rng = random.Random(seed)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.action_mask = np.zeros(MAX_ACTIONS, dtype=bool)

        self.battle: Battle | None = None
        self._legal: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _rows(self, ids: list[str], gear: list[dict] | None) -> list[dict[str, Any]]:
        rows = []
        for i, uid in enumerate(ids):
            row = db.get_unit(uid, self.db_path)
            if row is None:
                raise ValueError(f"unknown unit id: {uid}")
            if gear and i < len(gear) and gear[i]:
                from .gear import apply_loadout_to_row
                row = apply_loadout_to_row(row, gear[i])
            rows.append(row)
        return rows

    def _build_battle(self) -> Battle:
        seed = self.rng.randint(0, 2**31 - 1)
        return Battle(
            self._rows(self.team_a_ids, self.gear_a),
            self._rows(self.team_b_ids, self.gear_b),
            name_a="Side A", name_b="Side B",
            level=self.level, max_av=self.max_av, max_rounds=self.max_rounds,
            levels_a=self.levels_a, levels_b=self.levels_b,
            rng=random.Random(seed), verbose=False,
        )

    def _observation(self) -> np.ndarray:
        assert self.battle is not None
        return battle_to_observation(self.battle, self.agent_side)

    # ------------------------------------------------------------------ #
    # current actor & legal actions from the agent's perspective          #
    # ------------------------------------------------------------------ #
    def _next_actor(self):
        b = self.battle
        assert b is not None
        alive = b.all_alive()
        if not alive:
            return None
        return min(alive, key=lambda u: u.av)

    def _refresh_legal(self) -> list[dict[str, Any]]:
        """Recompute the legal action list for the agent's next decision."""
        b = self.battle
        assert b is not None
        self._legal = []
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        actor = self._next_actor()
        if actor is not None and not b.finished and actor.side == self.agent_side:
            self._legal = b.legal_actions(actor)[:MAX_ACTIONS]
            for i in range(len(self._legal)):
                mask[i] = True
        self.action_mask = mask
        return self._legal

    # Gymnasium MaskablePPO convention ---------------------------------- #
    def action_masks(self) -> list[bool]:
        return self.action_mask.astype(bool).tolist()

    # ------------------------------------------------------------------ #
    # Gym API                                                            #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.battle = self._build_battle()
        self.reward_shaper.reset(self.battle)

        # fast-forward until it is the agent's side to act
        guard = 0
        while not self.battle.finished and guard < 128:
            actor = self._next_actor()
            if actor is None or actor.side == self.agent_side:
                break
            self.battle.advance_time_to(actor)
            policy = self.opponent_policy
            if policy is None:
                from .agents import random_action
                policy = random_action
            self.battle.run_turn(actor, policy)
            self.battle.after_turn_hooks(actor)
            guard += 1

        self._refresh_legal()
        info: dict[str, Any] = {"snapshot": self.battle.snapshot(),
                                "mask": self.action_mask}
        return self._observation(), info

    def step(self, action: int):
        b = self.battle
        assert b is not None
        terminated = truncated = False
        reward = 0.0

        if self._legal and 0 <= action < len(self._legal):
            chosen = self._legal[action]
            actor = self._next_actor()
            if actor is not None:
                self.reward_shaper.reset(b)
                b.run_turn(actor, lambda *_: chosen)
                b.after_turn_hooks(actor)
                reward += self.reward_shaper.turn_reward(b, self.agent_side)
        else:
            reward += self.reward_shaper.config.illegal_action
            if self._legal:
                actor = self._next_actor()
                if actor is not None:
                    self.reward_shaper.reset(b)
                    fallback = self._legal[0]
                    b.run_turn(actor, lambda *_: fallback)
                    b.after_turn_hooks(actor)
                    reward += self.reward_shaper.turn_reward(b, self.agent_side)

        # opponent acts until it is the agent's side again or the battle ends
        guard = 0
        while not b.finished and guard < 64:
            nxt = self._next_actor()
            if nxt is None or nxt.side == self.agent_side:
                break
            b.advance_time_to(nxt)
            policy = self.opponent_policy
            if policy is None:
                from .agents import random_action
                policy = random_action
            self.reward_shaper.reset(b)
            b.run_turn(nxt, policy)
            b.after_turn_hooks(nxt)
            reward -= self.reward_shaper.turn_reward(b, nxt.side)
            guard += 1

        if b.finished:
            terminated = True
            if b.winner == self.agent_side:
                reward += self.reward_shaper.config.win
            elif b.winner in ("A", "B"):
                reward += self.reward_shaper.config.loss
            else:
                reward += self.reward_shaper.config.draw
        elif b.time >= b.max_av or b.turn_count >= b.max_rounds * 2:
            truncated = True

        self._refresh_legal()
        info = {"snapshot": b.snapshot(), "winner": b.winner, "mask": self.action_mask}
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):  # pragma: no cover
        b = self.battle
        if b is None:
            return ""
        return "\n".join(Battle.format_event(e) for e in b.log[-30:])

    def close(self) -> None:  # pragma: no cover
        pass


# Register with Gymnasium when available
if _HAS_GYM:  # pragma: no cover
    try:
        from gymnasium.envs.registration import register
        register(id="HSRDuel-v0", entry_point="cli_hsr.env:HSREnv")
    except Exception:
        pass
