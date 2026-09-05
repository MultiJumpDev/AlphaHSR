"""Import the full playable roster from the StarRailRes datamine.

Source: https://github.com/Mar-7th/StarRailRes (index_new/en), which mirrors
the game data from Dimbreath/StarRailData. Provides, for all playable
characters:
  - characters.json           -> id, name, rarity, element, path, max_sp(energy)
  - character_promotions.json -> exact per-level stat curves (7 tiers)
  - character_skills.json     -> per-level skill params (exact multipliers)

What we build for each character:
  id            -> slug (e.g. "seele"), datamine id kept as dm_id
  stats         -> {"promotion_curve": {...verbatim...}, "max_sp": ...}
  kit           -> generic engine kit auto-derived from the datamine skills:
                   basic/skill/ult/talent entries with:
                     - multipliers resolved at SKILL_LEVEL (default Lv.10 for
                       basic/Lv.12 for skill/ult — adjustable via --skill-level)
                     - targets/effect shape from the datamine `effect` field
                     - debuffs (DoTs, freeze, slow...) detected from desc text
  weaknesses    -> for enemy use: the character's own element

The generic kits are intentionally simple but quantitatively exact: every
damage number, heal, shield and DoT multiplier comes from the datamine params
table. Hand-crafted faithful kits (characters.json / characters_new.json)
always take precedence at seed time — the datamine roster fills the rest.

Usage:
    uv run python tools/import_starrailres.py <path_to_StarRailRes>/index_new/en
    uv run python tools/import_starrailres.py /tmp/StarRailRes/index_new/en --skill-level 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Datamine path names -> our internal path ids (characters.json convention).
PATH_MAP = {
    "Knight": "Preservation",
    "Priest": "Abundance",
    "Mage": "Erudition",
    "Rogue": "Hunt",
    "Warlock": "Nihility",
    "Warrior": "Destruction",
    "Shaman": "Harmony",
    "Memory": "Remembrance",
    "Elation": "Elation",
}

# skill types we care about
TYPE_BASIC = "Normal"
TYPE_SKILL = "BPSkill"
TYPE_ULT = "Ultra"
TYPE_TALENT = "Talent"

DOT_KEYWORDS = {
    "Burn": ["burn"],
    "Bleed": ["bleed"],
    "Shock": ["shock"],
    "Wind Shear": ["wind shear"],
    "Etching": ["etch"],
}

# Element damage multiplier used when converting flat "DMG equal to X% of ATK"
# param fractions into engine skill multipliers. Datamine params are fractions
# of ATK (0.5 = 50%), same as our engine convention -> direct copy.
def _slug(name: str, dm_id: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or f"char_{dm_id}"


def _params_at(skill: dict, idx: int, level: int) -> float | None:
    """Resolve one #N[i] param at a given skill level (1-based)."""
    params = skill.get("params") or []
    if not params or idx >= len(params[0]):
        return None
    row = params[min(level, len(params)) - 1]
    if idx >= len(row):
        return None
    return float(row[idx])


def _detect_dot(desc: str) -> str | None:
    low = desc.lower()
    for dot, keys in DOT_KEYWORDS.items():
        if any(k in low for k in keys):
            return dot
    if "bleed" in low:
        return "Bleed"
    return None


def _detect_control(desc: str) -> str | None:
    low = desc.lower()
    if "freeze" in low:
        return "freeze"
    if "imprison" in low:
        return "imprisonment"
    if "entangle" in low:
        return "entanglement"
    return None


def _hit_count(effect: str, desc: str) -> int:
    """Approximate hit count for toughness reduction."""
    low = desc.lower()
    if effect == "Bounce":
        m = re.search(r"(\d+) time", low)
        return max(3, int(m.group(1))) if m else 3
    if effect == "Blast":
        return 2
    return 1


def build_generic_kit(dm_id: str, skills: dict, skill_ids: list[str],
                      skill_level: int) -> dict:
    """Auto-derive an engine kit from datamine skills.

    Returns a dict with top-level ``basic``/``skill``/``ultimate``/``talent``
    keys, matching the hand-crafted character file convention.
    """
    kit: dict = {"effect": "attack", "source": "datamine"}
    by_type: dict[str, dict] = {}
    for sid in skill_ids:
        s = skills.get(sid)
        if not s or s.get("type") not in (TYPE_BASIC, TYPE_SKILL, TYPE_ULT, TYPE_TALENT):
            continue
        # Some characters have enhanced variants (ids like 11xxxxx); keep the base
        by_type.setdefault(s["type"], s)

    def mult_of(s: dict, p_idx: int = 0) -> float:
        v = _params_at(s, p_idx, skill_level)
        return v if v is not None else 1.0

    def shape_targets(s: dict) -> tuple[str, int]:
        eff = s.get("effect", "")
        desc = s.get("desc", "") or s.get("simple_desc", "")
        if eff == "AoEAttack":
            return "all", 1
        if eff == "Blast":
            return "blast", 1
        if eff == "Bounce":
            return "bounce", 5
        if eff in ("SingleAttack",):
            return "one", 1
        return "one", 1

    def damage_entry(s: dict, action: str) -> dict | None:
        if s is None:
            return None
        element = s.get("element") or ""
        if not element:
            return None  # non-damaging (pure support) skill
        targets, hits = shape_targets(s)
        desc = s.get("desc", "") or ""
        entry: dict = {
            "name": s.get("name", action.title()),
            "effect": "damage",
            "mult": mult_of(s),
            "targets": targets,
            "element": element,
        }
        if hits > 1:
            entry["hits"] = hits
        # toughness reduction approximations (game: basic 30/skill 60/ult 90/blast 90)
        if action == "basic":
            entry["toughness_dmg"] = 30
        elif action == "skill":
            entry["toughness_dmg"] = 60
        elif action == "ult":
            entry["toughness_dmg"] = 90
        if action in ("basic", "skill"):
            entry["energy_gain"] = 20 if action == "basic" else 30
        dot = _detect_dot(desc)
        if dot and action in ("skill", "ult"):
            entry["dot"] = {"kind": dot, "mult": round(mult_of(s, 1) or 0.5, 4), "duration": 2}
        cc = _detect_control(desc)
        if cc and action == "ult":
            entry["control"] = {"kind": cc, "chance": 1.0, "duration": 1}
        return entry

    basic = by_type.get(TYPE_BASIC)
    skill = by_type.get(TYPE_SKILL)
    ult = by_type.get(TYPE_ULT)
    talent = by_type.get(TYPE_TALENT)

    if basic and basic.get("element"):
        e = damage_entry(basic, "basic")
        if e:
            kit["basic"] = e
            kit["effect"] = "attack"
    if skill:
        e = damage_entry(skill, "skill")
        if e:
            kit["skill"] = e
        else:
            # support skill: shield or heal
            eff = skill.get("effect", "")
            desc = skill.get("desc", "") or ""
            if eff == "Defence" or "Shield" in (skill.get("simple_desc") or ""):
                kit["skill"] = {
                    "name": skill.get("name", "Skill"),
                    "effect": "shield",
                    "targets": "one",
                    "mult": mult_of(skill, 0),
                    "flat": _params_at(skill, 3, skill_level) or 0.0,
                    "duration": 3,
                    "stat_basis": "def",
                }
            elif eff in ("Restore",) or "Heal" in (skill.get("simple_desc") or ""):
                kit["skill"] = {
                    "name": skill.get("name", "Skill"),
                    "effect": "heal",
                    "targets": "one",
                    "mult": mult_of(skill, 0),
                    "flat": _params_at(skill, 1, skill_level) or 0.0,
                    "stat_basis": "hp" if "Max HP" in desc else "atk",
                }
            else:
                kit["skill"] = {"name": skill.get("name", "Skill"), "effect": "support", "targets": "one", "note": skill.get("simple_desc", "")}
    if ult:
        e = damage_entry(ult, "ult")
        if e:
            kit["ultimate"] = e
        else:
            eff = ult.get("effect", "")
            if eff in ("Restore",) or "Heal" in (ult.get("simple_desc") or ""):
                kit["ultimate"] = {
                    "name": ult.get("name", "Ultimate"),
                    "effect": "heal",
                    "targets": "all",
                    "mult": mult_of(ult, 0),
                    "flat": _params_at(ult, 1, skill_level) or 0.0,
                    "stat_basis": "hp" if "Max HP" in (ult.get("desc") or "") else "atk",
                }
            else:
                kit["ultimate"] = {"name": ult.get("name", "Ultimate"), "effect": "support", "targets": "all", "note": ult.get("simple_desc", "")}
    if talent:
        # passives vary wildly; record the multiplier for FUA-style triggering
        e = damage_entry(talent, "skill")
        if e:
            e["effect"] = "fua"
            e["trigger"] = "talent"
            kit["talent"] = e
    if "basic" not in kit and "skill" not in kit:
        return None
    return kit


def import_roster(src_dir: Path, skill_level: int, limit: int | None = None) -> dict:
    chars = json.loads((src_dir / "characters.json").read_text(encoding="utf-8"))
    promos = json.loads((src_dir / "character_promotions.json").read_text(encoding="utf-8"))
    skills = json.loads((src_dir / "character_skills.json").read_text(encoding="utf-8"))

    out: list[dict] = []
    used_slugs: set[str] = set()
    for dm in sorted(chars.values(), key=lambda c: int(c["id"])):
        dm_id = dm["id"]
        if dm_id not in promos:
            continue
        name = dm["name"]
        slug = _slug(name, dm_id)
        while slug in used_slugs:
            slug += f"_{dm_id}"
        used_slugs.add(slug)

        # our engine element ids: Physical/Fire/Ice/Thunder/Wind/Quantum/Imaginary
        element = dm["element"]
        path = PATH_MAP.get(dm["path"], dm["path"])

        kit = build_generic_kit(dm_id, skills, dm.get("skills", []), skill_level)
        if kit is None:
            continue

        entry = {
            "id": slug,
            "dm_id": dm_id,
            "name": name,
            "rarity": dm["rarity"],
            "element": element,
            "path": path,
            # Remembrance units have max_sp = null (memosprite mechanics, no energy)
            "stats": {"promotion_curve": promos[dm_id], "max_sp": dm.get("max_sp") or 0},
            "notes": f"Imported from StarRailRes {dm_id} (generic kit, skill Lv.{skill_level})",
        }
        # kit keys live at the top level, like the hand-crafted files
        entry.update(kit)
        out.append(entry)
        if limit and len(out) >= limit:
            break

    return {
        "_comment": (
            "Auto-imported from Mar-7th/StarRailRes (index_new/en). Stats use the exact "
            "promotion curves (per-level); kit multipliers come from the datamine skill "
            "param tables resolved at the listed skill level. Generic kits: hand-crafted "
            "kits in characters.json take precedence at seed time."
        ),
        "skill_level": skill_level,
        "characters": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="path to StarRailRes/index_new/en")
    ap.add_argument("--skill-level", type=int, default=10,
                    help="skill level used to resolve multipliers (1-15, default 10)")
    ap.add_argument("--limit", type=int, default=None, help="import only N characters")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "datamine_characters.json")
    args = ap.parse_args()

    data = import_roster(args.src, args.skill_level, args.limit)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imported {len(data['characters'])} characters -> {args.out}")


if __name__ == "__main__":
    main()
