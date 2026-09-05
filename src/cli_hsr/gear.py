"""Gear system: Light Cones and Relic Sets.

A `Loadout` attaches to one unit slot:
    {"light_cone": "swordplay", "superimposition": 5,
     "relics": {"musketeer": 4, "salsotto": 2}}

Effects are split in two channels:
- **stats** — flat cone base stats + set bonuses, merged into the unit's
  stat computation (`Unit.gear_stats`).
- **passives** — engine hooks applied at battle start (`battle_start_buff`,
  `energy_flat_start`). Cone passives scale linearly S1->S5.

Passive descriptions always include the real in-game text reference in
`data/light_cones.json` so simplifications are reviewable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import db

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@dataclass
class Loadout:
    unit_id: str = ""
    light_cone: str | None = None
    superimposition: int = 1
    relic_sets: dict[str, int] = field(default_factory=dict)  # set_id -> pieces

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "light_cone": self.light_cone,
            "superimposition": self.superimposition,
            "relics": dict(self.relic_sets),
        }


def loadout_from_dict(d: dict[str, Any] | None) -> Loadout:
    d = d or {}
    return Loadout(
        unit_id=d.get("unit_id", ""),
        light_cone=d.get("light_cone"),
        superimposition=int(d.get("superimposition", 1)),
        relic_sets={k: int(v) for k, v in (d.get("relics") or {}).items()},
    )


# ---------------------------------------------------------------------- #
# gear database access                                                   #
# ---------------------------------------------------------------------- #
def _fallback_load(filename: str) -> list[dict[str, Any]]:
    f = DATA_DIR / filename
    if not f.exists():
        return []
    key = "light_cones" if "cone" in filename else "relic_sets"
    return json.loads(f.read_text(encoding="utf-8"))[key]


def get_light_cone(cone_id: str) -> dict[str, Any] | None:
    row = db.get_light_cone(cone_id)
    if row is not None:
        return row
    for c in _fallback_load("light_cones.json"):
        if c["id"] == cone_id:
            return c
    return None


def get_relic_set(set_id: str) -> dict[str, Any] | None:
    row = db.get_relic_set(set_id)
    if row is not None:
        return row
    for r in _fallback_load("relics.json"):
        if r["id"] == set_id:
            return r
    return None


# ---------------------------------------------------------------------- #
# resolution                                                             #
# ---------------------------------------------------------------------- #
def _interp_s1_s5(s1: float, s5: float, s: int) -> float:
    return s1 + (s5 - s1) * (min(5, max(1, s)) - 1) / 4.0


def cone_base_stats(cone: dict[str, Any]) -> dict[str, float]:
    """Flat stats from the cone itself (Lv.80 reference values)."""
    bs = cone.get("base_stats", {})
    return {f"{k}_flat": float(v) for k, v in bs.items()
            if k in ("hp", "atk", "def")}


def cone_passives(cone: dict[str, Any], superimposition: int) -> list[dict[str, Any]]:
    """Passive hooks scaled to the superimposition level."""
    passive = cone.get("passive")
    if not passive:
        return []
    kind = passive.get("kind")
    if kind == "battle_start_buff":
        value = _interp_s1_s5(passive["value"], passive.get("value_s5", passive["value"]),
                              superimposition)
        return [{"kind": "battle_start_buff", "stat": passive["stat"], "value": value,
                 "name": cone["name"], "note": passive.get("note", "")}]
    if kind == "energy_flat_start":
        value = _interp_s1_s5(passive["value"], passive.get("value_s5", passive["value"]),
                              superimposition)
        return [{"kind": "energy_flat_start", "value": value,
                 "name": cone["name"], "note": passive.get("note", "")}]
    # unknown kinds are surfaced so reviewers see them
    return [{"kind": kind, "raw": passive, "note": "unimplemented passive kind"}]


def relic_set_bonuses(set_def: dict[str, Any], pieces: int) -> list[dict[str, Any]]:
    """Set bonuses for the equipped piece count (each threshold <= pieces)."""
    out: list[dict[str, Any]] = []
    bonuses = set_def.get("bonuses", {})
    for threshold in sorted(bonuses, key=int):
        if pieces >= int(threshold):
            b = bonuses[threshold]
            out.append({"stat": b["stat"], "value": float(b["value"]),
                        "set": set_def["id"], "pieces": int(threshold),
                        "note": b.get("note", "")})
    return out


def resolve_loadout_stats(loadout: Loadout) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute (stat bonuses, passives) for a loadout."""
    stats: dict[str, float] = {}
    passives: list[dict[str, Any]] = []

    if loadout.light_cone:
        cone = get_light_cone(loadout.light_cone)
        if cone is None:
            raise ValueError(f"unknown light cone: {loadout.light_cone}")
        for k, v in cone_base_stats(cone).items():
            stats[k] = stats.get(k, 0.0) + v
        passives.extend(cone_passives(cone, loadout.superimposition))

    for set_id, pieces in loadout.relic_sets.items():
        if pieces <= 0:
            continue
        set_def = get_relic_set(set_id)
        if set_def is None:
            raise ValueError(f"unknown relic set: {set_id}")
        for b in relic_set_bonuses(set_def, pieces):
            stats[b["stat"]] = stats.get(b["stat"], 0.0) + b["value"]

    return stats, passives


# ---------------------------------------------------------------------- #
# row application (bridge to units/engine)                               #
# ---------------------------------------------------------------------- #
def apply_loadout_to_row(row: dict[str, Any], loadout_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the DB row with the loadout baked into kit + stats.

    `kit_json["gear"] = {"stats": {...}, "passives": [...], "loadout": {...}}`
    """
    import copy

    loadout = loadout_from_dict(loadout_dict)
    stats, passives = resolve_loadout_stats(loadout)
    row = copy.deepcopy(row)
    row["kit_json"] = dict(row["kit_json"])
    row["kit_json"]["gear"] = {
        "stats": stats,
        "passives": [p for p in passives if p.get("kind") in
                     ("battle_start_buff", "energy_flat_start")],
        "loadout": loadout.to_dict(),
    }
    return row
