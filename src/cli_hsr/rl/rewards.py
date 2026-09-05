"""Event-driven reward shaping for the RL environment.

`EventRewardFn` converts the engine's event log (populated during a turn)
into a scalar reward, then adds terminal win/loss bonuses. Everything is
configurable through `RewardConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..engine import Battle


@dataclass
class RewardConfig:
    # terminal
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    # per-event shaping
    damage_dealt_scale: float = 0.0002
    damage_taken_scale: float = 0.0002
    heal_scale: float = 0.0002
    shield_scale: float = 0.0002
    break_reward: float = 0.05
    dot_tick_scale: float = 0.0002
    kill_reward: float = 0.15
    ult_reward: float = 0.02
    fua_reward: float = 0.01
    sp_gained_scale: float = 0.005
    sp_spent_scale: float = 0.005
    illegal_action: float = -0.05
    # pitfall guards
    clip: float | None = 10.0
    max_time_bonus: float = 0.10     # reward finishing fast (fraction of win)
    max_av: float = 4000.0


class RewardFn(Protocol):
    def __call__(self, events: list[dict[str, Any]], battle: Battle,
                 agent_side: str, previous_len: int) -> float:
        ...


def event_reward_fn(config: RewardConfig) -> RewardFn:
    """Build a reward function scoring events produced by the agent's turn."""

    def reward(events: list[dict[str, Any]], battle: Battle,
               agent_side: str, previous_len: int) -> float:
        total = 0.0
        for e in events[previous_len:]:
            kind = e.get("event")
            if kind == "damage":
                dmg = float(e.get("damage", 0.0))
                if e.get("source_side", e.get("side")) == agent_side:
                    total += dmg * config.damage_dealt_scale
                else:
                    total -= dmg * config.damage_taken_scale
            elif kind == "heal":
                total += float(e.get("amount", 0.0)) * config.heal_scale
            elif kind == "shield":
                total += float(e.get("amount", 0.0)) * config.shield_scale
            elif kind == "weakness_break":
                total += config.break_reward
            elif kind == "dot_tick":
                total += float(e.get("damage", 0.0)) * config.dot_tick_scale
            elif kind == "defeat":
                total += config.kill_reward
            elif kind == "ultimate":
                total += config.ult_reward
            elif kind == "fua":
                total += config.fua_reward
            elif kind == "sp" and e.get("delta", 0) > 0:
                total += config.sp_gained_scale
        return total

    return reward


class RewardShaper:
    """Stateful helper used by the env each step."""

    def __init__(self, config: RewardConfig | None = None,
                 fn: RewardFn | None = None) -> None:
        self.config = config or RewardConfig()
        self.fn = fn or event_reward_fn(self.config)
        self._prev_log_len = 0

    def reset(self, battle: Battle) -> None:
        self._prev_log_len = len(battle.log)

    def turn_reward(self, battle: Battle, agent_side: str) -> float:
        r = self.fn(battle.log, battle, agent_side, self._prev_log_len)
        self._prev_log_len = len(battle.log)
        cfg = self.config
        if cfg.clip is not None:
            r = max(-cfg.clip, min(cfg.clip, r))
        return r
