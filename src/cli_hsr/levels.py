"""Per-level stat resolution.

Characters imported from the StarRailRes datamine carry their promotion
curves verbatim (7 ascension tiers, {base, step} per stat). The tier bases
are pre-loaded with the ascension bonuses, and the step counts from level 1
within every tier (verified against the official wiki stat tables — Seele
Lv.80: HP 931.4~931, ATK 640.3~640, DEF 363.8~363):

    stat(L) = base_tier(L) + step_tier(L) * (L - 1)

with tier index 0 for Lv.1-19, 1 for Lv.20-29, ..., 6 for Lv.70-80.
SPD/taunt/crit are unscaled by level.

Enemies are not present in the datamine; their stats scale procedurally with
the Level Multiplier used everywhere else in the engine:

    stat(L) = stat(80) * level_multiplier(L) / level_multiplier(80)
"""

from __future__ import annotations

from typing import Any

from . import constants as C

# First level of each ascension tier (index i of the promotions `values` list).
TIER_STARTS: tuple[int, ...] = (1, 20, 30, 40, 50, 60, 70)

# Promotion-curve keys -> UnitStats / kit stats keys.
_KEYMAP = {"hp": "max_hp", "atk": "atk", "def": "def", "spd": "spd"}


def promotion_stat(values: list[dict[str, Any]], key: str, level: int) -> float:
    """Exact datamine value of one promotion stat at a character level."""
    level = max(1, min(80, int(level)))
    # tier index: level 1-19 -> 0, 20-29 -> 1, ..., 70-80 -> 6
    if level >= 70:
        tier = 6
    elif level >= 60:
        tier = 5
    elif level >= 50:
        tier = 4
    elif level >= 40:
        tier = 3
    elif level >= 30:
        tier = 2
    elif level >= 20:
        tier = 1
    else:
        tier = 0
    entry = values[tier][key]
    return entry["base"] + entry["step"] * (level - 1)


def has_promotion_curve(kit: dict[str, Any]) -> bool:
    return bool(kit.get("promotion_curve"))


def character_stats_at_level(stats: dict[str, Any], level: int) -> dict[str, float]:
    """Resolve a character's base stats at ``level`` from its promotion curve.

    ``stats`` is the unit's stats dict (may carry ``promotion_curve`` +
    ``max_sp``). Returns a UnitStats-ready dict {max_hp, atk, def, spd,
    crit_rate, crit_dmg, energy_max}. Falls back to the dict itself when no
    curve is present.
    """
    curve = stats.get("promotion_curve")
    if not curve:
        static = dict(stats)
        static.setdefault("energy_max", 0.0)
        return static

    values = curve["values"]
    out: dict[str, float] = {}
    for pk, sk in _KEYMAP.items():
        out[sk] = promotion_stat(values, pk, level)
    out["crit_rate"] = float(promotion_stat(values, "crit_rate", level))
    out["crit_dmg"] = float(promotion_stat(values, "crit_dmg", level))
    out["energy_max"] = float(stats.get("max_sp") or 0)
    return out


def enemy_scale(level: int) -> float:
    """Multiplicative scale from the curated Lv.80 reference to ``level``."""
    return C.level_multiplier(level) / C.level_multiplier(80)


def enemy_stats_at_level(stats80: dict[str, Any], level: int) -> dict[str, Any]:
    """Scale curated enemy stats (defined at Lv.80) to any level 1-90."""
    k = enemy_scale(level)
    out = dict(stats80)
    out["max_hp"] = round(float(stats80["max_hp"]) * k, 1)
    out["atk"] = round(float(stats80["atk"]) * k, 1)
    d = stats80.get("def")
    if d is not None:
        out["def"] = round(float(d) * k, 1)
    return out
