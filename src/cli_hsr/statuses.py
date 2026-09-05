"""Status effects: buffs, debuffs, DoTs, crowd control, shields.

Durations follow the game rules:
- DoTs tick at the START of the afflicted unit's turn, and their duration
  decrements at the END of that same turn.
- Stat buffs/debuffs tick down at the end of the holder's turn; SPD-modifying
  statuses only tick down if they were present at the start of the turn.
- Crowd control (Freeze/Entanglement) is resolved when the holder's turn starts.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

_id_counter = itertools.count(1)

DOT_KINDS = {"burn", "shock", "bleed", "wind_shear"}
CC_KINDS = {"freeze", "entanglement", "imprisonment"}


@dataclass
class StatusEffect:
    kind: str                      # dot | buff | debuff | cc | shield | special
    name: str
    source_id: str = ""
    stat: str | None = None        # which stat this modifies (buffs/debuffs)
    value: float = 0.0             # damage coeff (dots) or stat delta (buffs/debuffs)
    duration_turns: float | None = 1   # None = lasts whole battle
    stacks: int = 1
    max_stacks: int = 1
    requires_turn_start: bool = False  # only decay if present at start of holder's turn
    source_level: int = 80         # attacker level, for Level-Multiplier dots
    source_break_effect: float = 0.0
    element: str | None = None     # for dots / implanted weakness
    extra: dict[str, Any] = field(default_factory=dict)
    uid: int = field(default_factory=lambda: next(_id_counter))
    was_present_at_turn_start: bool = False
    just_applied: bool = True

    # ---------------- helpers ----------------
    @property
    def is_dot(self) -> bool:
        return self.kind == "dot"

    @property
    def is_cc(self) -> bool:
        return self.kind == "cc"

    @property
    def is_buff(self) -> bool:
        return self.kind == "buff"

    @property
    def is_debuff(self) -> bool:
        return self.kind == "debuff"

    def add_stacks(self, n: int) -> None:
        self.stacks = min(self.max_stacks, self.stacks + n)

    def refresh(self, other: "StatusEffect") -> None:
        """Merge another application of the (same-name) effect into this one."""
        self.add_stacks(other.stacks)
        if other.duration_turns is not None:
            self.duration_turns = max(self.duration_turns or 0, other.duration_turns)
        self.value = max(self.value, other.value)
        self.source_break_effect = max(self.source_break_effect, other.source_break_effect)


class StatusManager:
    """Holds all active effects on one unit."""

    def __init__(self) -> None:
        self.effects: list[StatusEffect] = []

    # ------------------------------------------------ adding
    def add(self, effect: StatusEffect) -> StatusEffect:
        if effect.stat == "spd_percent" or effect.stat == "spd_flat":
            effect.requires_turn_start = True
        existing = self.find(effect.name)
        if existing is not None and existing.kind == effect.kind and existing.stat == effect.stat:
            existing.refresh(effect)
            return existing
        self.effects.append(effect)
        return effect

    def find(self, name: str) -> StatusEffect | None:
        for e in self.effects:
            if e.name == name:
                return e
        return None

    def has(self, name: str) -> bool:
        return self.find(name) is not None

    # ------------------------------------------------ querying
    def all_of(self, kind: str | None = None) -> list[StatusEffect]:
        if kind is None:
            return list(self.effects)
        return [e for e in self.effects if e.kind == kind]

    def dots(self) -> list[StatusEffect]:
        return self.all_of("dot")

    def stat_total(self, stat: str) -> float:
        """Sum of stat deltas from buffs (positive) and debuffs (negative)."""
        total = 0.0
        for e in self.effects:
            if e.stat == stat:
                if e.kind == "buff":
                    total += e.value * e.stacks if e.name == "wind_shear_spd" else e.value
                elif e.kind == "debuff":
                    total -= e.value * e.stacks if e.name.startswith("wind_shear") else e.value
        return total

    def vulnerability(self, element: str | None = None, damage_class: str = "normal") -> float:
        """Total damage-taken increase (elemental / all-type / dot / break)."""
        total = 0.0
        for e in self.effects:
            if e.kind != "debuff":
                continue
            if e.stat == "vuln":
                total += e.value
            elif e.stat == "element_vuln" and e.element == element:
                total += e.value
            elif e.stat == "dot_vuln" and damage_class == "dot":
                total += e.value
            elif e.stat == "break_vuln" and damage_class == "break":
                total += e.value
        return total

    def shield_total(self) -> float:
        return sum(e.value for e in self.effects if e.kind == "shield")

    def cc_names(self) -> list[str]:
        return [e.name for e in self.effects if e.kind == "cc"]

    def is_frozen(self) -> bool:
        return self.has("Freeze")

    def is_entangled(self) -> bool:
        return self.has("Entanglement")

    def is_imprisoned(self) -> bool:
        return self.has("Imprisonment")

    # ------------------------------------------------ decay
    def mark_turn_start(self) -> None:
        for e in self.effects:
            e.was_present_at_turn_start = True
            e.just_applied = False

    def tick_turn_end(self) -> None:
        """Called at the end of the holder's turn."""
        survivors: list[StatusEffect] = []
        for e in self.effects:
            if e.duration_turns is None:
                survivors.append(e)
                continue
            if e.requires_turn_start and not e.was_present_at_turn_start:
                survivors.append(e)  # SPD-type buffs don't tick the turn they were applied
                continue
            e.duration_turns -= 1
            if e.duration_turns > 0:
                survivors.append(e)
        self.effects = survivors

    # ------------------------------------------------ removal
    def remove(self, name: str) -> StatusEffect | None:
        for e in self.effects:
            if e.name == name:
                self.effects.remove(e)
                return e
        return None

    def remove_dots(self) -> list[StatusEffect]:
        removed = [e for e in self.effects if e.is_dot]
        for e in removed:
            self.effects.remove(e)
        return removed

    def remove_buffs(self) -> list[StatusEffect]:
        removed = [e for e in self.effects if e.is_buff]
        for e in removed:
            self.effects.remove(e)
        return removed

    def remove_debuffs(self) -> list[StatusEffect]:
        removed = [e for e in self.effects if e.kind == "debuff"]
        for e in removed:
            self.effects.remove(e)
        return removed

    def clear(self) -> None:
        self.effects.clear()
