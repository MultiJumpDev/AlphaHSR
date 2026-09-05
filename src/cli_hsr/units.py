"""Combat units: characters, enemies, and summons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import constants as C
from . import levels as LV
from .statuses import StatusEffect, StatusManager


@dataclass
class UnitStats:
    max_hp: float
    atk: float
    spd: float
    crit_rate: float = C.DEFAULT_CRIT_RATE
    crit_dmg: float = C.DEFAULT_CRIT_DMG
    energy_max: float = 0.0
    taunt: float = 100.0
    def_stat: float | None = None  # resolved character DEF (None = derive)


class Unit:
    """A combatant on either side. Wraps a DB row (character or enemy)."""

    def __init__(self, db_row: dict[str, Any], side: str, index: int, level: int = C.DEFAULT_LEVEL):
        self.id: str = db_row["id"]
        self.name: str = db_row["name"]
        self.db_row = db_row
        self.side = side              # "A" or "B"
        self.index = index
        self.level = level
        self.unit_type: str = db_row["unit_type"]
        self.element: str = db_row["element"]
        self.path: str | None = db_row.get("path")
        self.is_character = self.unit_type == "character"
        self.is_enemy = not self.is_character
        self.category = db_row.get("category") or ("character" if self.is_character else "normal")
        self.is_elite_or_boss = self.category in ("elite", "boss")

        stats = self._resolve_stats(db_row)
        self.base = UnitStats(
            max_hp=float(stats.get("max_hp", 1000)),
            atk=float(stats.get("atk", 100)),
            spd=float(stats.get("spd", 100)),
            crit_rate=float(stats.get("crit_rate", C.DEFAULT_CRIT_RATE)),
            crit_dmg=float(stats.get("crit_dmg", C.DEFAULT_CRIT_DMG)),
            energy_max=float(stats.get("energy_max", 0)),
            taunt=float(stats.get("taunt", 100)),
            def_stat=(float(stats["def"]) if stats.get("def") is not None else None),
        )
        if self.is_enemy:
            # JSON SPD values are the final in-game speeds at the enemy's level
            # (no further level multiplier). HP is scaled for balance against
            # gearless base-stat rosters (see constants for the knobs).
            hp_scale = C.ENEMY_HP_SCALE
            if self.category == "elite":
                hp_scale = C.ENEMY_HP_SCALE_ELITE
            elif self.category == "boss":
                hp_scale = C.ENEMY_HP_SCALE_BOSS
            self.base.max_hp = round(self.base.max_hp * hp_scale, 1)

        self.kit: dict[str, Any] = db_row["kit_json"]

        # dynamic state
        self.statuses = StatusManager()
        self.shield = 0.0
        self.alive = True
        self.hp = self.max_hp
        self.energy = 0.5 * self.base.energy_max if self.base.energy_max else 0.0

        # toughness (enemies only)
        self.max_toughness = float(self.kit.get("toughness", 0)) if self.is_enemy else 0.0
        self.toughness = self.max_toughness
        self.weakness_broken = False

        # weaknesses (enemies), possibly implanted
        self.weaknesses: list[str] = list(self.kit.get("weaknesses", []))

        # boss phases
        self.phase = 1
        self.total_phases = int(self.kit.get("phases", 1)) if self.is_enemy else 1

        # action value
        self.av = C.base_av(self.effective_spd())

        # per-turn flags
        self.took_turn_this_cycle = False

        # gear: flat cone base stats are additive to the base stat block
        gear = self.kit.get("gear", {}) if isinstance(self.kit, dict) else {}
        gear_stats = gear.get("stats", {})
        self.base.max_hp += float(gear_stats.get("hp_flat", 0.0))
        self.base.atk += float(gear_stats.get("atk_flat", 0.0))

        # character-specific runtime state
        self.runtime: dict[str, Any] = {}

    # ------------------------------------------------------------ stats
    def _resolve_stats(self, db_row: dict[str, Any]) -> dict[str, Any]:
        """Base stats for this unit at its level.

        Imported characters: exact datamine promotion curve. Curated enemies:
        procedural Level-Multiplier scaling from their Lv.80 reference. Curated
        characters (hand-written kits): static stats as-is.
        """
        kit = db_row["kit_json"]
        stats_json = db_row["stats_json"]
        if self.is_character:
            if LV.has_promotion_curve(stats_json):
                return LV.character_stats_at_level(stats_json, self.level)
            return dict(stats_json)
        return LV.enemy_stats_at_level(stats_json, self.level)

    def _stat_total(self, stat: str) -> float:
        gear = self.kit.get("gear", {}) if isinstance(self.kit, dict) else {}
        gear_stats = gear.get("stats", {})
        gear_value = float(gear_stats.get(stat, 0.0))
        return self.statuses.stat_total(stat) + gear_value

    def effective_max_hp(self) -> float:
        return self.base.max_hp * (1 + self._stat_total("max_hp_percent"))

    @property
    def max_hp(self) -> float:
        return self.effective_max_hp()

    def effective_atk(self) -> float:
        return self.base.atk * (1 + self._stat_total("atk_percent")) + self._stat_total("atk_flat")

    def effective_def(self) -> float:
        base = C.enemy_def(self.level) if self.is_enemy else self._base_def_from_kit()
        return base * (1 + self._stat_total("def_percent")) + self._stat_total("def_flat")

    def _base_def_from_kit(self) -> float:
        # Characters: promotion curve resolves into base.def; legacy kits may
        # carry kit/stats.def; else derive a stable default from level.
        d = self.base.def_stat
        if d is None:
            d = self.db_row.get("stats", {}).get("def", 460)
        return float(d)

    def effective_spd(self) -> float:
        spd = self.base.spd * (1 + self._stat_total("spd_percent")) + self._stat_total("spd_flat")
        # gear additive base stats also include flat hp/atk/def via gear_stats
        if self.statuses.is_imprisoned():
            spd *= 1 - C.IMPRISONMENT_SPD_REDUCTION
        return max(1.0, spd)

    def effective_crit_rate(self) -> float:
        return self.base.crit_rate + self._stat_total("crit_rate")

    def effective_crit_dmg(self) -> float:
        return self.base.crit_dmg + self._stat_total("crit_dmg")

    def effective_break_effect(self) -> float:
        return self._stat_total("break_effect")

    def effective_dmg_boost(self, element: str | None = None) -> float:
        total = self._stat_total("dmg_boost")
        if element:
            key = f"{element.lower()}_dmg_boost"
            total += self._stat_total(key)
        return total

    def effective_res(self, element: str | None = None) -> float:
        # Enemies default 0.2 all-type RES unless specified; characters 0.
        base_res = 0.0
        if self.is_enemy:
            base_res = float(self.kit.get("res", 0.2))
        total = base_res + self._stat_total("all_type_res")
        if element:
            total += self._stat_total(f"res_{element.lower()}")
        return total

    def energy_regen_rate(self) -> float:
        return 1.0 + self._stat_total("energy_regen")

    def healing_bonus(self) -> float:
        return self._stat_total("healing_boost")

    def is_immune_to_cc(self) -> bool:
        return bool(self.runtime.get("cc_immune", False))

    # ------------------------------------------------------------ damage intake
    def take_damage(self, amount: float) -> float:
        """Apply damage respecting shields; returns actual HP damage dealt."""
        if not self.alive or amount <= 0:
            return 0.0
        remaining = amount
        if self.shield > 0:
            absorbed = min(self.shield, remaining)
            self.shield -= absorbed
            remaining -= absorbed
        hp_dmg = min(self.hp, remaining)
        self.hp -= hp_dmg
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
        return hp_dmg

    def heal(self, amount: float) -> float:
        if not self.alive:
            return 0.0
        before = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - before

    def add_shield(self, amount: float, name: str = "Shield", duration_turns: float | None = 2) -> None:
        if amount <= 0:
            return
        existing = self.statuses.find(name)
        if existing is not None:
            self.shield -= existing.value
            self.statuses.effects.remove(existing)
        self.shield += amount
        self.statuses.add(StatusEffect(kind="shield", name=name, value=amount,
                                       duration_turns=duration_turns))

    # ------------------------------------------------------------ toughness
    def is_weak_to(self, element: str) -> bool:
        return element in self.weaknesses

    def has_toughness(self) -> bool:
        return self.toughness > 0

    def reduce_toughness(self, amount: float) -> None:
        if self.weakness_broken:
            return
        self.toughness = max(0.0, self.toughness - amount)
        if self.toughness == 0:
            self.weakness_broken = True

    def recover_from_break(self) -> None:
        self.weakness_broken = False
        self.toughness = self.max_toughness

    # ------------------------------------------------------------ action value
    def reset_av(self) -> None:
        self.av = C.base_av(self.effective_spd())

    def advance_av(self, ratio: float) -> None:
        """Advance action by ratio (0..1+). Action Value New = max(0, Old - BaseAV*ratio)."""
        self.av = max(0.0, self.av - C.base_av(self.effective_spd()) * ratio)

    def set_av_fraction(self, fraction: float) -> None:
        """Set the current AV to a fraction of the unit's full turn time.
        Used for 'skip turn but advance the next one by 50%' (Freeze)."""
        self.av = C.base_av(self.effective_spd()) * max(0.0, min(1.0, fraction))

    def delay_av(self, ratio: float) -> None:
        self.av = self.av + C.base_av(self.effective_spd()) * ratio

    # ------------------------------------------------------------ misc
    def gain_energy(self, base_amount: float, affected_by_regen: bool = True) -> None:
        if self.base.energy_max <= 0:
            return
        amt = base_amount * (self.energy_regen_rate() if affected_by_regen else 1.0)
        self.energy = min(self.base.energy_max, self.energy + amt)

    def ult_ready(self) -> bool:
        return self.base.energy_max > 0 and self.energy >= self.base.energy_max

    def hp_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Unit {self.name} [{self.side}{self.index}] HP {self.hp:.0f}/{self.max_hp:.0f}>"


def build_unit(db_row: dict[str, Any], side: str, index: int, level: int = C.DEFAULT_LEVEL) -> Unit:
    return Unit(db_row, side, index, level)
