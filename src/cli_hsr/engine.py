"""Turn-based battle engine reproducing Honkai: Star Rail combat rules.

Damage formula (Damage wiki):
  DMG = Base DMG x CRIT x DMG Boost x Weaken x DEF Mult x RES Mult
        x Vulnerability x DMG Mitigation x Broken Mult

Break DMG = Break Base DMG(elem) x Level Multiplier x Max Toughness Mult
            x (1+Break Effect) x DEF x RES x Vulnerability x Mitigation
            x Broken Mult

AV loop (Speed wiki): AV = 10000/SPD; the lowest-AV unit acts; all units'
AV decrease by that amount. Weakness break delays 25%. DoTs tick at the
start of the afflicted unit's turn; durations decay at its end.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import constants as C
from .statuses import StatusEffect, StatusManager
from .units import Unit

ActionName = str  # "B" basic, "S" skill, "U" ultimate, "F" flee; target = unit


@dataclass
class Side:
    name: str
    units: list[Unit] = field(default_factory=list)

    def alive_units(self) -> list[Unit]:
        return [u for u in self.units if u.alive]

    def is_defeated(self) -> bool:
        return not self.alive_units()


@dataclass
class StepResult:
    """What an RL agent sees to pick its action."""
    actor: Unit | None
    legal_actions: list[dict[str, Any]] = field(default_factory=list)


class Battle:
    """One battle between two sides (1..N units each)."""

    def __init__(
        self,
        team_a_rows: list[dict[str, Any]],
        team_b_rows: list[dict[str, Any]],
        name_a: str = "Team A",
        name_b: str = "Team B",
        level: int = C.DEFAULT_LEVEL,
        max_av: float = 4000.0,
        rng: random.Random | None = None,
        verbose: bool = False,
        skill_points: int = 3,
        max_skill_points: int = 5,
        max_rounds: int = 200,
        level_a: int | None = None,
        level_b: int | None = None,
        levels_a: list[int] | None = None,
        levels_b: list[int] | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.verbose = verbose
        self.level = level
        self.max_av = max_av                # battle time limit in AV
        self.max_rounds = max_rounds
        self.skill_points = skill_points
        self.max_skill_points = max_skill_points

        # per-side levels override the shared `level` (anyone-vs-anyone duels:
        # e.g. a Lv.50 character against a Lv.80 boss); per-unit lists override
        # the per-side scalar (drafted teams may mix levels).
        lvl_a = level_a if level_a is not None else level
        lvl_b = level_b if level_b is not None else level

        def _lvl(levels: list[int] | None, fallback: int, i: int) -> int:
            return int(levels[i]) if levels and i < len(levels) else fallback

        self.side_a = Side(name_a, [Unit(r, "A", i, _lvl(levels_a, lvl_a, i)) for i, r in enumerate(team_a_rows)])
        self.side_b = Side(name_b, [Unit(r, "B", i, _lvl(levels_b, lvl_b, i)) for i, r in enumerate(team_b_rows)])

        self.units: list[Unit] = self.side_a.units + self.side_b.units
        self.time = 0.0                     # elapsed AV
        self.turn_count = 0
        self.log: list[dict[str, Any]] = []
        self.finished = False
        self.winner: str | None = None      # "A", "B" or "draw"

        # per-battle runtime: cooldowns/limits
        self.fua_used_this_turn: dict[str, int] = {}
        self.auto_field_used: dict[str, int] = {}
        self.start_of_battle()

    # ------------------------------------------------------------------ #
    # logging                                                            #
    # ------------------------------------------------------------------ #
    def emit(self, event: str, **kw: Any) -> None:
        entry = {"t": round(self.time, 1), "event": event, **kw}
        self.log.append(entry)
        if self.verbose:
            print(self.format_event(entry))

    @staticmethod
    def format_event(entry: dict[str, Any]) -> str:
        ev = entry["event"]
        t = entry["t"]
        if ev == "turn_start":
            return f"[{t:>7.1f} AV] === {entry['actor']} ({entry['side']}) turn {entry.get('turn', '')} ==="
        if ev == "damage":
            crit = " CRIT!" if entry.get("crit") else ""
            brk = " [BREAK]" if entry.get("break_event") else ""
            return (f"           {entry['source']} -> {entry['target']}: "
                    f"{entry['damage']:.0f} {entry.get('element', '')} DMG{crit}{brk} "
                    f"(HP {entry.get('hp_after', 0):.0f})")
        if ev == "dot_tick":
            return (f"           DoT {entry['name']} on {entry['target']}: "
                    f"{entry['damage']:.0f} DMG")
        if ev == "weakness_break":
            return (f"           WEAKNESS BREAK on {entry['target']} "
                    f"({entry.get('debuff', '')})")
        if ev == "heal":
            return f"           {entry['source']} heals {entry['target']} +{entry['amount']:.0f}"
        if ev == "shield":
            return f"           {entry['target']} gains shield {entry['amount']:.0f}"
        if ev == "ultimate":
            return f"           {entry['actor']} uses ULTIMATE: {entry['name']}"
        if ev == "skill":
            return f"           {entry['actor']} uses Skill: {entry['name']}"
        if ev == "basic":
            return f"           {entry['actor']} uses Basic ATK"
        if ev == "fua":
            return f"           {entry['actor']} Follow-Up ATK: {entry.get('name', '')}"
        if ev == "defeat":
            return f"           {entry['target']} is DEFEATED"
        if ev == "phase":
            return f"           {entry['target']} enters PHASE {entry['phase']}!"
        if ev == "end":
            return f"[{t:>7.1f} AV] BATTLE END - winner: {entry.get('winner')}"
        if ev == "sp":
            return f"           SP={entry['sp']}"
        if ev == "event":
            return f"           {entry.get('msg', '')}"
        return f"[{t:>7.1f}] {ev} {entry.get('msg', '')}"

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def enemies_of(self, unit: Unit) -> list[Unit]:
        other = self.side_b if unit.side == "A" else self.side_a
        return other.alive_units()

    def allies_of(self, unit: Unit) -> list[Unit]:
        side = self.side_a if unit.side == "A" else self.side_b
        return side.alive_units()

    def all_alive(self) -> list[Unit]:
        return self.side_a.alive_units() + self.side_b.alive_units()

    def random_enemy(self, unit: Unit) -> Unit | None:
        foes = self.enemies_of(unit)
        return self.rng.choice(foes) if foes else None

    def adjacent_of(self, target: Unit) -> list[Unit]:
        """Units adjacent to `target` on the target's own side (blast pattern)."""
        side = self.side_a if target.side == "A" else self.side_b
        ordered = [u for u in side.units if u.alive]
        if target not in ordered:
            return [u for u in ordered if u is not target][:2]
        idx = ordered.index(target)
        adj = [ordered[(idx - 1) % len(ordered)], ordered[(idx + 1) % len(ordered)]]
        seen: list[Unit] = []
        for u in adj:
            if u is not target and u not in seen:
                seen.append(u)
        return seen

    def damage_class_multiplier(self, attacker: Unit, element: str | None,
                                dmg_boost_extra: float = 0.0) -> float:
        return 1.0 + attacker.effective_dmg_boost(element) + dmg_boost_extra

    # ------------------------------------------------------------------ #
    # damage core                                                        #
    # ------------------------------------------------------------------ #
    def deal_damage(
        self,
        attacker: Unit,
        target: Unit,
        base_mult: float,
        element: str | None,
        attack_class: str = "basic",     # basic|skill|ultimate|fua|dot|break|additional|true
        toughness_dmg: float = 0.0,
        can_crit: bool = True,
        is_dot: bool = False,
        extra_dmg_boost: float = 0.0,
        label: str = "",
    ) -> dict[str, Any]:
        """Apply the wiki damage formula end-to-end (incl. toughness + break)."""
        if not target.alive or not attacker.alive:
            return {"damage": 0.0}

        stat = attacker.effective_atk()
        base_dmg = stat * base_mult

        # CRIT
        crit = False
        if can_crit and not is_dot:
            crit = self.rng.random() < attacker.effective_crit_rate()
        crit_mult = (1 + attacker.effective_crit_dmg()) if crit else 1.0

        # DMG boost
        dmg_boost = self.damage_class_multiplier(attacker, element, extra_dmg_boost)

        # Weaken on attacker
        weaken = max(0.0, attacker.statuses.stat_total("weaken"))
        weaken_mult = 1.0 - min(1.0, weaken)

        # DEF multiplier (Damage wiki):
        #   DEF Mult = 1 - DEF / (DEF + 200 + 10 x LevelAttacker)
        # with effective DEF = base DEF x max(0, 1 + DEFbonus - DEFred - DEFignore)
        def_bonus = target.statuses.stat_total("def_percent")
        def_ignore = attacker.statuses.stat_total("def_pen")
        target_def = target.effective_def() * max(0.0, 1.0 + def_bonus - def_ignore)
        def_mult = 1.0 - target_def / (target_def + 200 + 10 * attacker.level)

        # RES multiplier
        res = target.effective_res(element)
        res_pen = attacker.statuses.stat_total("res_pen")
        res_mult = 1.0 - (res - res_pen)

        # Vulnerability on target
        vuln = target.statuses.vulnerability(element, "dot" if is_dot else "normal")
        vuln_mult = 1.0 + vuln

        # universal mitigation (e.g. boss innate 10% handled as dmg_reduction)
        mitigation = max(0.0, target.statuses.stat_total("dmg_reduction"))
        mitigation_mult = 1.0 - min(1.0, mitigation)

        # broken multiplier
        broken_mult = 0.9 if (target.is_enemy and not target.weakness_broken) else 1.0

        dmg = (base_dmg * crit_mult * dmg_boost * weaken_mult * def_mult
               * res_mult * vuln_mult * mitigation_mult * broken_mult)

        # toughness depletion
        break_info: dict[str, Any] = {}
        if toughness_dmg > 0 and target.is_enemy and target.has_toughness() and not target.weakness_broken:
            if element and target.is_weak_to(element):
                eff = (1 + attacker.statuses.stat_total("break_efficiency"))
                toughness_dmg *= eff
                before = target.toughness
                target.reduce_toughness(toughness_dmg)
                if target.weakness_broken and before > 0:
                    break_info = self.trigger_weakness_break(attacker, target, element)

        dealt = target.take_damage(dmg)

        entry = self.emit("damage", source=attacker.name, target=target.name,
                          source_side=attacker.side, target_side=target.side,
                          damage=dmg, element=element or "", crit=crit,
                          hp_after=target.hp, break_event=bool(break_info),
                          label=label or attack_class)

        # on-kill energy
        if not target.alive:
            self.emit("defeat", target=target.name)
            attacker.gain_energy(C.ENERGY_ON_KILL, affected_by_regen=False)
            self.check_battle_end()

        return {"damage": dmg, "crit": crit, "break": break_info, "killed": not target.alive}

    # ------------------------------------------------------------------ #
    # weakness break                                                     #
    # ------------------------------------------------------------------ #
    def trigger_weakness_break(self, attacker: Unit, target: Unit, element: str) -> dict[str, Any]:
        """Break DMG + 25% AV delay + 150% base chance element debuff."""
        level_mult = C.level_multiplier(attacker.level)
        mtm = C.max_toughness_multiplier(target.max_toughness)
        be = attacker.effective_break_effect()

        coeff = C.BREAK_BASE_DMG[element]
        base_dmg = coeff * level_mult * mtm

        # standard multipliers (same DEF formula as deal_damage)
        target_def = target.effective_def() * max(0.0, 1.0
                         + target.statuses.stat_total("def_percent")
                         - attacker.statuses.stat_total("def_pen"))
        def_mult = 1.0 - target_def / (target_def + 200 + 10 * attacker.level)
        res_mult = 1.0 - (target.effective_res(element) - attacker.statuses.stat_total("res_pen"))
        vuln_mult = 1.0 + target.statuses.vulnerability(element, "break")
        dmg = base_dmg * (1 + be) * def_mult * res_mult * vuln_mult
        # broken multiplier = 1.0 (target is broken at this instant)

        target.take_damage(dmg)

        # 25% action delay
        target.delay_av(C.WEAKNESS_BREAK_ACTION_DELAY)

        debuff_name = ""
        name, is_dot, turns = C.BREAK_DEBUFF[element]
        if self.rng.random() < C.BASE_CHANCE_BREAK_DEBUFF:
            debuff_name = self.apply_break_debuff(attacker, target, element, name, is_dot, turns)

        self.emit("weakness_break", target=target.name, element=element,
                  debuff=debuff_name or name, damage=dmg)

        # on-break talent hooks
        self.on_weakness_break_hooks(attacker, target, element)

        if not target.alive:
            self.emit("defeat", target=target.name)
            self.check_battle_end()
        return {"damage": dmg, "debuff": debuff_name or name}

    def apply_break_debuff(self, attacker: Unit, target: Unit, element: str,
                           name: str, is_dot: bool, turns: int) -> str:
        be = attacker.effective_break_effect()
        if element == C.Element.PHYSICAL:
            # Bleed: DoT based on target max HP, capped
            coeff = (C.BLEED_HP_COEFFICIENT_ELITE if target.is_elite_or_boss
                     else C.BLEED_HP_COEFFICIENT_NORMAL)
            cap = (C.BLEED_CAP_COEFFICIENT * C.level_multiplier(attacker.level)
                   * C.max_toughness_multiplier(target.max_toughness))
            value = min(coeff * target.max_hp, cap)
            target.statuses.add(StatusEffect(
                kind="dot", name=name, source_id=attacker.id, stat="bleed",
                value=value, duration_turns=turns, source_level=attacker.level,
                source_break_effect=be, element=element))
        elif element == C.Element.WIND:
            stacks = (C.WIND_SHEAR_STACKS_ELITE if target.is_elite_or_boss
                      else C.WIND_SHEAR_STACKS_NORMAL)
            eff = target.statuses.find("Wind Shear")
            if eff:
                eff.add_stacks(stacks)
                eff.duration_turns = turns
            else:
                target.statuses.add(StatusEffect(
                    kind="dot", name=name, source_id=attacker.id, stat="wind_shear",
                    value=1.0, duration_turns=turns, stacks=stacks,
                    max_stacks=C.WIND_SHEAR_MAX_STACKS,
                    source_level=attacker.level, source_break_effect=be,
                    element=element))
        elif element == C.Element.QUANTUM:
            target.statuses.add(StatusEffect(
                kind="cc", name=name, source_id=attacker.id, stat="entanglement",
                value=C.ENTANGLEMENT_DELAY, duration_turns=turns, stacks=1,
                max_stacks=C.ENTANGLEMENT_MAX_STACKS,
                source_level=attacker.level, source_break_effect=be))
            target.delay_av(C.ENTANGLEMENT_DELAY * (1 + be))
        elif element == C.Element.IMAGINARY:
            target.statuses.add(StatusEffect(
                kind="cc", name=name, source_id=attacker.id, stat="imprisonment",
                value=C.IMPRISONMENT_DELAY, duration_turns=turns,
                source_level=attacker.level, source_break_effect=be))
            target.delay_av(C.IMPRISONMENT_DELAY * (1 + be))
        elif is_dot:  # Burn / Shock
            coeff = {C.Element.FIRE: 1.0, C.Element.LIGHTNING: 2.0}[element]
            target.statuses.add(StatusEffect(
                kind="dot", name=name, source_id=attacker.id, stat=name.lower().replace(" ", "_"),
                value=coeff, duration_turns=turns, source_level=attacker.level,
                source_break_effect=be, element=element))
        else:  # Freeze
            target.statuses.add(StatusEffect(
                kind="cc", name=name, source_id=attacker.id, stat="freeze",
                value=1.0, duration_turns=turns,
                source_level=attacker.level, source_break_effect=be))
        return name

    # ------------------------------------------------------------------ #
    # DoTs                                                               #
    # ------------------------------------------------------------------ #
    def tick_dots(self, unit: Unit) -> None:
        """Called at the start of `unit`'s turn (Damage wiki: DoTs tick then)."""
        for dot in unit.statuses.dots():
            if dot.name == "Bleed":
                dmg = dot.value * (1 + dot.source_break_effect)
            elif dot.name == "Wind Shear":
                lm = C.level_multiplier(dot.source_level)
                dmg = dot.stacks * lm * (1 + dot.source_break_effect)
            else:  # Burn / Shock coeff x Level Multiplier
                lm = C.level_multiplier(dot.source_level)
                dmg = dot.value * lm * (1 + dot.source_break_effect)
            # DoT multipliers: DEF/RES/Vuln apply
            def_factor = max(0.0, 1.0 + unit.statuses.stat_total("def_percent"))
            def_mult = (dot.source_level + 20) / ((unit.level + 20) * def_factor + dot.source_level + 20)
            res_mult = 1.0 - unit.effective_res(dot.element)
            vuln_mult = 1.0 + unit.statuses.vulnerability(dot.element, "dot")
            final = dmg * def_mult * res_mult * vuln_mult
            unit.take_damage(final)
            self.emit("dot_tick", name=dot.name, target=unit.name, damage=final)
        if not unit.alive:
            self.emit("defeat", target=unit.name)
            self.check_battle_end()

    def detonate_dots(self, attacker: Unit, targets: Iterable[Unit], ratio: float = 1.0) -> None:
        """Kafka/Black Swan style: trigger DoTs immediately at ratio of tick dmg."""
        for t in targets:
            for dot in t.statuses.dots():
                if dot.element != C.Element.LIGHTNING and ratio >= 1.0 and attacker.id == "kafka":
                    continue  # Kafka detonates only Lightning DoTs
                lm = C.level_multiplier(dot.source_level)
                if dot.name == "Bleed":
                    dmg = dot.value * (1 + dot.source_break_effect)
                elif dot.name == "Wind Shear":
                    dmg = dot.stacks * lm * (1 + dot.source_break_effect)
                else:
                    dmg = dot.value * lm * (1 + dot.source_break_effect)
                dmg *= ratio
                def_factor = max(0.0, 1.0 + t.statuses.stat_total("def_percent"))
                def_mult = (dot.source_level + 20) / ((t.level + 20) * def_factor + dot.source_level + 20)
                res_mult = 1.0 - t.effective_res(dot.element)
                vuln_mult = 1.0 + t.statuses.vulnerability(dot.element, "dot")
                final = dmg * def_mult * res_mult * vuln_mult
                t.take_damage(final)
                self.emit("dot_tick", name=f"detonate:{dot.name}", target=t.name, damage=final)
                if not t.alive:
                    self.emit("defeat", target=t.name)
                    attacker.gain_energy(C.ENERGY_ON_KILL, affected_by_regen=False)
                    break
        self.check_battle_end()

    # ------------------------------------------------------------------ #
    # turn flow                                                          #
    # ------------------------------------------------------------------ #
    def start_of_battle(self) -> None:
        # initial AVs
        for u in self.units:
            u.av = C.base_av(u.effective_spd())
        # gear passives (light cones / relics)
        for u in self.units:
            gear = u.kit.get("gear", {}) if isinstance(u.kit, dict) else {}
            for p in gear.get("passives", []):
                if p.get("kind") == "battle_start_buff":
                    u.statuses.add(StatusEffect(
                        kind="buff", name=p.get("name", "Cone Passive"),
                        stat=p["stat"], value=float(p["value"]),
                        duration_turns=None))
                elif p.get("kind") == "energy_flat_start":
                    u.energy = min(u.base.energy_max, u.energy + float(p["value"]))
        # kit start-of-battle hooks (lightning-lord, firefly countdown etc.)
        for u in self.units:
            if u.is_character and u.kit.get("talent", {}).get("kind") == "summon":
                self.spawn_lightning_lord(u)
            if u.is_character and u.kit.get("talent", {}).get("kind") == "flying_aureus":
                u.runtime["flying_aureus"] = 0
            if u.is_character and u.kit.get("talent", {}).get("kind") == "crimson_knot":
                u.runtime["crimson_knot_ready"] = False
            if u.is_character and u.kit.get("ultimate", {}).get("kind") == "acheron_ult":
                u.runtime["crimson_knot_stacks"] = 0
            if u.is_character and u.kit.get("talent", {}).get("kind") == "auto_field":
                self.auto_field_used[u.id] = 0

    def next_actor(self) -> Unit | None:
        alive = self.all_alive()
        if not alive:
            return None
        return min(alive, key=lambda u: u.av)

    def advance_time_to(self, actor: Unit) -> None:
        """Advance global time to the actor's turn.

        Every other alive unit's AV decreases by the same delta, clamped at 0:
        units tied with the actor (AV 0 after the clamp) act immediately after,
        in list order. Without the clamp, a tie loser goes negative and the
        next 'delta' becomes negative - time would flow BACKWARD and that unit
        would receive free actions (this caused a systematic side-B advantage).
        """
        delta = max(0.0, actor.av)
        self.time += delta
        for u in self.all_alive():
            if u is actor:
                u.av = 0.0
            else:
                u.av = max(0.0, u.av - delta)

    def check_battle_end(self) -> bool:
        if self.finished:
            return True
        self.try_revives()
        a_dead = self.side_a.is_defeated()
        b_dead = self.side_b.is_defeated()
        if a_dead or b_dead:
            self.finished = True
            if a_dead and b_dead:
                self.winner = "draw"
            elif a_dead:
                self.winner = "B"
            else:
                self.winner = "A"
            self.emit("end", winner=self.winner)
            return True
        if self.time >= self.max_av or self.turn_count >= self.max_rounds * 2:
            self.finished = True
            # side with higher remaining HP fraction wins
            fa = sum(u.hp_fraction() for u in self.side_a.alive_units())
            fb = sum(u.hp_fraction() for u in self.side_b.alive_units())
            if abs(fa - fb) < 1e-9:
                self.winner = "draw"
            else:
                self.winner = "A" if fa > fb else "B"
            self.emit("end", winner=self.winner, timeout=True)
            return True
        return False

    # ------------------------------------------------------------------ #
    # legal actions (RL + UI)                                            #
    # ------------------------------------------------------------------ #
    def legal_actions(self, actor: Unit) -> list[dict[str, Any]]:
        """All legal (action, target) pairs for the actor."""
        acts: list[dict[str, Any]] = []
        kit = actor.kit
        foes = self.enemies_of(actor)
        allies = self.allies_of(actor)

        def push(kind: str, target: Unit | None, **extra: Any) -> None:
            acts.append({"kind": kind, "target": target, "actor": actor, **extra})

        # Basic ATK: always legal vs each enemy
        for f in foes:
            push("basic", f)

        # Skill
        skill = kit.get("skill", {})
        if skill and self.skill_points > 0:
            t = skill.get("targets", "single")
            if t in ("single", "blast", "aoe", "aoe_rainblade"):
                for f in foes:
                    push("skill", f)
            elif t in ("ally_single", "ally_heal", "ally_shield"):
                for a in allies:
                    push("skill", a)

        # Ultimate
        ult = kit.get("ultimate", {})
        if ult:
            if actor.ult_ready():
                t = ult.get("targets", "single")
                if t in ("single", "blast", "aoe", "aoe_rainblade"):
                    for f in foes:
                        push("ultimate", f)
                elif t in ("ally_single", "ally_heal", "ally_shield", "allies", "self"):
                    push("ultimate", actor if t == "self" else None)
        return acts

    # ------------------------------------------------------------------ #
    # action execution                                                   #
    # ------------------------------------------------------------------ #
    def perform_action(self, actor: Unit, action: dict[str, Any]) -> None:
        """Execute one selected action for the actor (called on their turn)."""
        kind = action["kind"]
        target = action.get("target")
        kit = actor.kit

        if kind == "basic":
            self.emit("basic", actor=actor.name)
            self.use_basic(actor, target)
        elif kind == "skill":
            self.emit("skill", actor=actor.name, name=kit["skill"].get("name", "Skill"))
            self.skill_points -= 1
            # energy is granted exactly once inside use_skill (kit value)
            self.use_skill(actor, target)
            self.emit("sp", sp=self.skill_points)
        elif kind == "ultimate":
            ult = kit.get("ultimate", {})
            self.emit("ultimate", actor=actor.name, name=ult.get("name", "?"))
            actor.energy = 0
            self.use_ultimate(actor, target)
        else:
            raise ValueError(f"unknown action kind: {kind}")

    def gain_energy_and_sp_hook(self, actor: Unit, action_kind: str) -> None:
        """Deprecated: energy is granted once inside use_basic/use_skill.
        Kept for API compatibility; only emits the SP event."""
        if action_kind == "skill":
            self.emit("sp", sp=self.skill_points)

    def use_basic(self, actor: Unit, target: Unit | None) -> None:
        kit = actor.kit
        basic = kit.get("basic", {})
        mult = basic.get("mult", 1.0)
        element = actor.element
        toughness = basic.get("toughness", C.BTO_BY_ACTION["basic"])
        # Firefly enhanced basics in combustion handled via use_skill-like hook
        if target is not None:
            self.deal_damage(actor, target, mult, element, "basic", toughness)
        actor.gain_energy(basic.get("energy_gain", C.ENERGY_ON_BASIC))
        self.skill_points = min(self.max_skill_points, self.skill_points + 1)
        self.emit("sp", sp=self.skill_points)
        # Bronya-style talent: advance self after basic
        talent = kit.get("talent", {})
        if talent.get("kind") == "on_basic" and self.rng.random() < talent.get("advance_chance", 0):
            actor.advance_av(talent.get("value", 0.25))

    def use_skill(self, actor: Unit, target: Unit | None) -> None:
        kit = actor.kit
        skill = kit.get("skill", {})
        mult = skill.get("mult", 0.0)
        element = actor.element
        toughness = skill.get("toughness", C.BTO_BY_ACTION["skill"])
        t = skill.get("targets", "single")

        if t in ("ally_single", "ally_heal", "ally_shield") and target is not None:
            if skill.get("heal_pct"):
                amount = actor.max_hp * skill["heal_pct"] * (1 + actor.healing_bonus())
                target.heal(amount)
                self.emit("heal", source=actor.name, target=target.name, amount=amount)
            if skill.get("shield_pct"):
                base_stat = actor.effective_def() if skill.get("shield_scale") == "def" else actor.max_hp
                shield = base_stat * skill["shield_pct"]
                target.add_shield(shield, name=f"{actor.name} Shield")
                self.emit("shield", target=target.name, amount=shield)
            for eff in skill.get("effects", []):
                self.apply_effect(actor, target if target else actor, eff)
            actor.gain_energy(skill.get("energy_gain", C.ENERGY_ON_SKILL))
            return

        if target is None:
            return
        # offensive skill
        if t == "single":
            self.deal_damage(actor, target, mult, element, "skill", toughness)
            self.apply_skill_effects(actor, [target], skill)
        elif t == "blast":
            self.deal_damage(actor, target, mult, element, "skill", toughness)
            for adj in self.adjacent_of(target):
                self.deal_damage(actor, adj, mult * skill.get("adjacent_ratio", 0.5),
                                 element, "skill", toughness * 0.5)
            self.apply_skill_effects(actor, [target] + self.adjacent_of(target), skill)
        elif t in ("aoe", "aoe_rainblade"):
            for f in self.enemies_of(actor):
                self.deal_damage(actor, f, mult, element, "skill", toughness)
            self.apply_skill_effects(actor, self.enemies_of(actor), skill)
        elif t == "bounce":
            for _ in range(skill.get("hits", 5)):
                ft = self.random_enemy(actor)
                if ft:
                    self.deal_damage(actor, ft, mult, element, "skill",
                                     toughness / skill.get("hits", 5))
            self.apply_skill_effects(actor, self.enemies_of(actor), skill)

        actor.gain_energy(skill.get("energy_gain", C.ENERGY_ON_SKILL))

    def apply_skill_effects(self, actor: Unit, targets: list[Unit], skill: dict[str, Any]) -> None:
        for eff in skill.get("effects", []):
            for t in targets:
                self.apply_effect(actor, t, eff)

    def use_ultimate(self, actor: Unit, target: Unit | None) -> None:
        kit = actor.kit
        ult = kit.get("ultimate", {})
        kind = ult.get("kind", "standard")
        element = actor.element
        toughness = ult.get("toughness", C.BTO_BY_ACTION["ultimate"])

        if kind == "complete_combustion":  # Firefly
            actor.advance_av(ult.get("advance", 1.0))
            actor.statuses.add(StatusEffect(
                kind="buff", name="Complete Combustion", stat="spd_flat",
                value=ult.get("spd_flat", 30), duration_turns=None,
                requires_turn_start=True))
            if ult.get("weakness_implant"):
                actor.runtime["combustion"] = True
            # countdown marker handled by fixed AV counter below
            actor.runtime["combustion_end_av"] = self.time + ult.get("countdown_av", 142.857)
            return

        if kind == "acheron_ult":  # Acheron
            stacks = actor.runtime.get("crimson_knot_stacks", 0)
            if stacks < 9:
                actor.energy = actor.base.energy_max  # can't actually fire; refund
                return
            actor.runtime["crimson_knot_stacks"] = 0
            targets = self.enemies_of(actor)
            main = target if target in targets else (targets[0] if targets else None)
            rb_mult = ult.get("rainblade_mult", 0.2592)
            knot_base = ult.get("knot_base", 0.09)
            for _ in range(ult.get("rainblade_hits", 3)):
                if main and main.alive:
                    self.deal_damage(actor, main, rb_mult, element, "ultimate",
                                     toughness / 3, label="Rainblade")
                    removed = min(3, stacks)
                    stacks -= removed
                    for f in targets:
                        if f.alive:
                            self.deal_damage(actor, f, knot_base * removed, element,
                                             "ultimate", 0, can_crit=True, label="Knot")
            if main and main.alive:
                self.deal_damage(actor, main, ult.get("resurge_mult", 1.296), element,
                                 "ultimate", toughness / 3, label="Stygian Resurge")
            for f in targets:
                if f.alive:
                    self.deal_damage(actor, f, ult.get("resurge_mult", 1.296) * 0.5,
                                     element, "ultimate", 0, label="Resurge splash")
            return

        t = ult.get("targets", "single")
        mult = ult.get("mult", 0.0)

        if t == "self":
            for eff in ult.get("effects", []):
                self.apply_effect(actor, actor, eff)
            return

        if target is None:
            return

        if t == "single":
            if ult.get("kind") == "boltsunder_blitz":  # Feixiao
                for _ in range(ult.get("hit_count", 6)):
                    if target.alive:
                        self.deal_damage(actor, target, ult.get("mult", 0.36), element,
                                         "ultimate", toughness / 7, label="Flying Aureus")
                if target.alive:
                    self.deal_damage(actor, target, ult.get("final_mult", 1.44), element,
                                     "ultimate", toughness / 7, label="Final Hit")
            else:
                self.deal_damage(actor, target, mult, element, "ultimate", toughness)
                self.apply_ult_effects(actor, [target], ult)
        elif t == "blast":
            self.deal_damage(actor, target, mult, element, "ultimate", toughness)
            for adj in self.adjacent_of(target):
                self.deal_damage(actor, adj, mult * ult.get("adjacent_ratio", 0.5),
                                 element, "ultimate", toughness * 0.5)
            self.apply_ult_effects(actor, [target] + self.adjacent_of(target), ult)
        elif t == "aoe":
            for f in self.enemies_of(actor):
                self.deal_damage(actor, f, mult, element, "ultimate", toughness)
            self.apply_ult_effects(actor, self.enemies_of(actor), ult)

    def apply_ult_effects(self, actor: Unit, targets: list[Unit], ult: dict[str, Any]) -> None:
        for eff in ult.get("effects", []):
            for t in targets:
                self.apply_effect(actor, t, eff)

    # ------------------------------------------------------------------ #
    # generic effect application                                         #
    # ------------------------------------------------------------------ #
    def apply_effect(self, source: Unit, target: Unit, eff: dict[str, Any]) -> None:
        kind = eff.get("kind")
        chance = eff.get("chance", 1.0)
        if kind not in ("detonate_dots",) and self.rng.random() > chance:
            return

        if kind == "buff":
            who = self.resolve_effect_targets(source, target, eff.get("on", "self"))
            for w in who:
                w.statuses.add(StatusEffect(
                    kind="buff", name=eff.get("name", f"{source.name} {eff.get('stat','buff')}"),
                    source_id=source.id, stat=eff.get("stat"), value=eff.get("value", 0),
                    duration_turns=eff.get("duration_turns", 1)))
        elif kind == "debuff":
            who = self.resolve_effect_targets(source, target, eff.get("on", "target"))
            for w in who:
                # effect hit rate vs effect res
                ehr = 1.0 + source.statuses.stat_total("effect_hit")
                er = w.effective_res(None) * 0  # all-type res used via effective_res
                er = w.statuses.stat_total("all_type_res")
                if self.rng.random() < min(1.0, chance * ehr - er):
                    w.statuses.add(StatusEffect(
                        kind="debuff", name=eff.get("name", eff.get("stat", "debuff")),
                        source_id=source.id, stat=eff.get("stat"), value=eff.get("value", 0),
                        duration_turns=eff.get("duration_turns", 2),
                        element=eff.get("element")))
        elif kind == "action_advance":
            who = self.resolve_effect_targets(source, target, eff.get("on", "ally_target"))
            for w in who:
                w.advance_av(eff.get("value", 1.0))
        elif kind == "dispel_buff":
            who = self.resolve_effect_targets(source, target, eff.get("on", "target"))
            for w in who:
                w.statuses.remove_buffs()
        elif kind == "detonate_dots":
            det_targets = [t for t in ([target] if target is not None else []) if t is not None]
            self.detonate_dots(source, det_targets, ratio=eff.get("ratio", 1.0))
        elif kind == "implant_weakness":
            who = self.resolve_effect_targets(source, target, "target")
            allies = self.allies_of(source)
            if allies:
                elem = self.rng.choice([a.element for a in allies])
                for w in who:
                    if elem not in w.weaknesses:
                        w.weaknesses.append(elem)
                        w.statuses.add(StatusEffect(
                            kind="debuff", name="Implanted Weakness", stat="element_vuln",
                            value=0.0, element=elem, duration_turns=2))
        elif kind == "apply_dot":
            # Generic DoT application (e.g. Luka's Bleed): coeff x Level Multiplier ticks
            target.statuses.add(StatusEffect(
                kind="dot", name=eff.get("name", "DoT"), source_id=source.id,
                stat=eff.get("stat", "generic_dot"), value=eff.get("value", 1.0),
                duration_turns=eff.get("duration_turns", 2),
                max_stacks=eff.get("max_stacks", 1),
                source_level=source.level, element=source.element))
        elif kind == "chance_freeze":
            who = self.resolve_effect_targets(source, target, "target")
            for w in who:
                if not w.is_immune_to_cc():
                    w.statuses.add(StatusEffect(
                        kind="cc", name="Freeze", stat="freeze", value=1.0,
                        duration_turns=eff.get("duration_turns", 1)))
        elif kind == "lightning_lord":
            pass  # handled in spawn hook
        elif kind == "cipher_points":
            source.runtime["cipher_points"] = min(
                source.kit.get("talent", {}).get("cipher_max", 7),
                source.runtime.get("cipher_points", 0) + eff.get("amount", 4))
        elif kind == "matrix_of_prescience":
            for a in self.allies_of(source):
                a.statuses.add(StatusEffect(
                    kind="buff", name="Matrix of Prescience", stat="crit_rate",
                    value=0.12, duration_turns=3))
            source.runtime["matrix"] = True
        elif kind == "epiphany":
            target.statuses.add(StatusEffect(
                kind="debuff", name="Epiphany", stat="dot_vuln",
                value=eff.get("value", 0.25), duration_turns=eff.get("duration_turns", 3)))
        elif kind == "apply_shock":
            if self.rng.random() < 1.0 and not target.statuses.has("Shock"):
                target.statuses.add(StatusEffect(
                    kind="dot", name="Shock", source_id=source.id, stat="shock",
                    value=2.0, duration_turns=2, source_level=source.level,
                    element=C.Element.LIGHTNING))
        elif kind == "allies_heal":
            for a in self.allies_of(source):
                amount = source.max_hp * eff.get("value", 0.1) * (1 + source.healing_bonus())
                a.heal(amount)
                self.emit("heal", source=source.name, target=a.name, amount=amount)
        elif kind == "energy_to_target":
            who = self.resolve_effect_targets(source, target, eff.get("on", "ally_target"))
            for w in who:
                w.gain_energy(eff.get("value", 50), affected_by_regen=eff.get("regen_affected", False))
        elif kind == "delay":
            who = self.resolve_effect_targets(source, target, eff.get("on", "target"))
            for w in who:
                w.delay_av(eff.get("value", 0.3) * (1 + source.effective_break_effect()
                                                    if eff.get("scale_be") else 1.0))
        elif kind == "follow_up":
            # immediate FUA strike defined inline by the effect
            t = eff.get("targets", "single")
            mult = eff.get("mult", 1.0)
            tough = eff.get("toughness", 30)
            if t == "single" and target is not None and target.alive:
                self.deal_damage(source, target, mult, source.element, "fua", tough)
            elif t == "aoe":
                for f in self.enemies_of(source):
                    self.deal_damage(source, f, mult, source.element, "fua", tough)
            elif t == "bounce":
                for _ in range(eff.get("hits", 3)):
                    ft = self.random_enemy(source)
                    if ft:
                        self.deal_damage(source, ft, mult, source.element, "fua", tough)
            source.gain_energy(eff.get("energy_gain", 0))
        elif kind == "burst_extra":
            # extra hit if N or fewer enemies remain (Argenti-style)
            if len(self.enemies_of(source)) <= eff.get("max_enemies", 2):
                mult = eff.get("mult", 1.0)
                if eff.get("targets", "aoe") == "single" and target is not None:
                    self.deal_damage(source, target, mult, source.element, "ultimate", eff.get("toughness", 0))
                else:
                    for f in self.enemies_of(source):
                        self.deal_damage(source, f, mult, source.element, "ultimate", eff.get("toughness", 0))
        elif kind == "execute_bonus":
            # bonus hit when the target is below an HP threshold (Luka/Hook)
            if target is not None and target.alive and target.hp_fraction() <= eff.get("threshold", 0.5):
                self.deal_damage(source, target, eff.get("mult", 0.5), source.element,
                                 "ultimate", eff.get("toughness", 0))
        elif kind == "dot_boost":
            who = self.resolve_effect_targets(source, target, eff.get("on", "target"))
            for w in who:
                w.statuses.add(StatusEffect(
                    kind="debuff", name=eff.get("name", "DoT Vulnerability"),
                    stat="dot_vuln", value=eff.get("value", 0.25),
                    duration_turns=eff.get("duration_turns", 2)))
        elif kind == "res_debuff":
            who = self.resolve_effect_targets(source, target, eff.get("on", "target"))
            for w in who:
                stat = f"res_{eff['element'].lower()}" if eff.get("element") else "all_type_res"
                w.statuses.add(StatusEffect(
                    kind="debuff", name=eff.get("name", f"RES Down {eff.get('element', 'All')}"),
                    stat=stat, value=eff.get("value", 0.1),
                    duration_turns=eff.get("duration_turns", 2)))
        elif kind == "def_debuff":
            who = self.resolve_effect_targets(source, target, eff.get("on", "target"))
            for w in who:
                w.statuses.add(StatusEffect(
                    kind="debuff", name=eff.get("name", "DEF Down"),
                    stat="def_percent", value=-abs(eff.get("value", 0.3)),
                    duration_turns=eff.get("duration_turns", 3)))
        elif kind == "spd_buff_flat":
            who = self.resolve_effect_targets(source, target, eff.get("on", "self"))
            for w in who:
                w.statuses.add(StatusEffect(
                    kind="buff", name=eff.get("name", "SPD Up"),
                    stat="spd_flat", value=eff.get("value", 25),
                    duration_turns=eff.get("duration_turns", 2)))
        elif kind == "heal_target":
            if target is not None:
                amount = (source.max_hp if eff.get("scale") == "max_hp" else source.effective_atk()) \
                    * eff.get("value", 0.3) * (1 + source.healing_bonus())
                target.heal(amount)
                self.emit("heal", source=source.name, target=target.name, amount=amount)
        elif kind == "revive_heal":
            # Bailu-style: prevent a knock-down once per battle
            if target is not None and not target.alive and target.runtime.get("revive_used") is None:
                target.runtime["revive_used"] = True
                target.alive = True
                target.hp = target.max_hp * eff.get("value", 0.5)
                self.emit("heal", source=source.name, target=target.name,
                          amount=target.hp, revive=True)
        elif kind == "ally_shield_all":
            base_stat = source.effective_def() if eff.get("scale", "def") == "def" else source.max_hp
            shield = base_stat * eff.get("value", 0.3)
            for a in self.allies_of(source):
                a.add_shield(shield, name=f"{source.name} Shield")
                self.emit("shield", target=a.name, amount=shield)

    def resolve_effect_targets(self, source: Unit, target: Unit | None, on: str) -> list[Unit]:
        if on == "self":
            return [source]
        if on == "allies":
            return self.allies_of(source)
        if on == "ally_target" and target is not None:
            return [target]
        if on == "enemies":
            return self.enemies_of(source)
        if on == "target" and target is not None:
            return [target]
        if target is not None:
            return [target]
        return [source]

    # ================================================================== #
    #  MAIN LOOP                                                         #
    # ================================================================== #
    def run_turn(self, actor: Unit, choose_action) -> None:
        """Run a full turn for `actor`; `choose_action(battle, actor, legal)` -> dict."""
        if self.finished or not actor.alive:
            return
        self.turn_count += 1
        self.fua_used_this_turn = {}
        for u in self.all_alive():
            u.runtime["fua_done"] = False
        self.auto_field_used[actor.id] = self.auto_field_used.get(actor.id, 0)

        self.emit("turn_start", actor=actor.name, side=actor.side, turn=self.turn_count)
        actor.statuses.mark_turn_start()

        # ---- DoTs tick at start of the actor's turn
        self.tick_dots(actor)
        if self.finished or not actor.alive:
            return

        # ---- CC resolution at turn start
        if actor.statuses.is_frozen():
            actor.statuses.remove("Freeze")
            self.emit("event", msg=f"{actor.name} is frozen and loses their turn")
            # ordered fix-up: set the next turn to half a turn away (game rule
            # on unfreezing); done INSTEAD of the normal full reset
            actor.set_av_fraction(0.5)
            self.end_of_turn(actor)
            return
        if actor.statuses.is_entangled():
            ent = actor.statuses.find("Entanglement")
            if ent:
                self.emit("event", msg=f"{actor.name} is Entangled ({ent.stacks} stacks)")
            actor.statuses.remove("Entanglement")
            actor.reset_av()
            self.end_of_turn(actor)
            return

        # ---- enemy AI action or player/agent decision
        # NOTE: AV reset happens exactly once at the start of the action phase;
        # every path through the turn ends via end_of_turn() exactly once.
        actor.reset_av()
        if actor.is_enemy:
            self.perform_enemy_action(actor)
        else:
            legal = self.legal_actions(actor)
            if not legal:
                self.end_of_turn(actor)
                return
            action = choose_action(self, actor, legal)
            self.perform_action(actor, action)

        self.end_of_turn(actor)

    def end_of_turn(self, actor: Unit) -> None:
        """Weakness-broken enemies recover here; status durations decay once."""
        if actor.is_enemy and actor.weakness_broken:
            actor.recover_from_break()
        actor.statuses.tick_turn_end()
        self.check_battle_end()

    def run(self, choose_action_a, choose_action_b, max_turns: int = 500) -> str:
        """Battle loop until finish. Returns winner ('A'|'B'|'draw')."""
        while not self.finished and self.turn_count < max_turns:
            actor = self.next_actor()
            if actor is None:
                break
            self.advance_time_to(actor)
            chooser = choose_action_a if actor.side == "A" else choose_action_b
            self.run_turn(actor, chooser)
            self.after_turn_hooks(actor)
            # Firefly combustion countdown
            for u in self.all_alive():
                if u.runtime.get("combustion") and self.time >= u.runtime.get("combustion_end_av", 0):
                    u.runtime.pop("combustion", None)
                    u.statuses.remove("Complete Combustion")
                    self.emit("event", msg=f"{u.name} exits Complete Combustion")
            self.check_battle_end()
        if not self.finished:
            self.finished = True
            self.winner = "draw"
            self.emit("end", winner="draw", timeout=True)
        return self.winner  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # enemy behaviour                                                    #
    # ------------------------------------------------------------------ #
    def perform_enemy_action(self, actor: Unit) -> None:
        actions = actor.kit.get("actions", [{"name": "Strike", "targets": "single", "mult": 1.0}])
        total = sum(a.get("weight", 1.0) for a in actions)
        roll = self.rng.random() * total
        upto = 0.0
        chosen = actions[0]
        for a in actions:
            upto += a.get("weight", 1.0)
            if roll <= upto:
                chosen = a
                break

        kind = chosen.get("kind", "attack")
        if kind in ("buff_def", "buff_atk"):
            stat = "def_percent" if kind == "buff_def" else "atk_percent"
            actor.statuses.add(StatusEffect(
                kind="buff", name=chosen["name"], stat=stat,
                value=chosen.get("value", 0.2),
                duration_turns=chosen.get("duration_turns", 2)))
            self.emit("event", msg=f"{actor.name} uses {chosen['name']}")
            return
        if kind == "freeze":
            t = self.pick_enemy_target(actor, "single")
            if t and self.rng.random() < chosen.get("chance", 0.5) and not t.is_immune_to_cc():
                t.statuses.add(StatusEffect(kind="cc", name="Freeze", stat="freeze",
                                            value=1.0, duration_turns=chosen.get("duration_turns", 1)))
                self.emit("event", msg=f"{actor.name} freezes {t.name}")
            return
        if kind == "imprison":
            t = self.pick_enemy_target(actor, "single")
            if t and not t.is_immune_to_cc():
                t.statuses.add(StatusEffect(kind="cc", name="Imprisonment", stat="imprisonment",
                                            value=chosen.get("value", 0.2),
                                            duration_turns=chosen.get("duration_turns", 1)))
                t.delay_av(chosen.get("value", 0.2))
                self.emit("event", msg=f"{actor.name} imprisons {t.name}")
            return
        if kind == "entangle":
            t = self.pick_enemy_target(actor, "single")
            if t and not t.is_immune_to_cc():
                t.statuses.add(StatusEffect(kind="cc", name="Entanglement", stat="entanglement",
                                            value=chosen.get("value", 0.25),
                                            duration_turns=chosen.get("duration_turns", 1),
                                            stacks=1, max_stacks=C.ENTANGLEMENT_MAX_STACKS))
                t.delay_av(chosen.get("value", 0.25))
                self.emit("event", msg=f"{actor.name} entangles {t.name}")
            return
        if kind == "dot":
            # direct DoT application (e.g. Wind Shear stacks, Burn) without break
            t = self.pick_enemy_target(actor, chosen.get("targets", "single"))
            targets = self.enemies_of(actor) if t and chosen.get("targets") == "aoe" else [t]
            for target in targets:
                if target is None:
                    continue
                name = chosen.get("dot", "Burn")
                stacks = int(chosen.get("stacks", 1))
                existing = target.statuses.find(name)
                if existing is not None:
                    existing.add_stacks(stacks)
                    existing.duration_turns = chosen.get("duration_turns", 2)
                else:
                    target.statuses.add(StatusEffect(
                        kind="dot", name=name, source_id=actor.id,
                        stat=name.lower().replace(" ", "_"),
                        value=float(chosen.get("value", 1.0)), stacks=stacks,
                        max_stacks=int(chosen.get("max_stacks", 5)),
                        duration_turns=chosen.get("duration_turns", 2),
                        source_level=actor.level, element=actor.element))
                self.emit("event", msg=f"{actor.name} applies {name} to {target.name}")
            return
        if kind == "summon":
            from . import db as _db
            from .units import build_unit
            sid = chosen.get("unit")
            row = _db.get_unit(sid) if sid else None
            if row is None:
                return
            side = actor.side
            side_obj = self.side_a if side == "A" else self.side_b
            if len(side_obj.units) >= 5:
                return  # field is full
            summon = build_unit(row, side, len(side_obj.units), level=actor.level)
            side_obj.units.append(summon)
            self.units.append(summon)
            summon.av = actor.av  # acts in step with the summoner
            self.emit("summon", name=summon.name, by=actor.name)
            return
        if kind == "heal":
            for t in self._allied_targets(actor, chosen.get("targets", "self")):
                amount = chosen.get("value", 0.0) * t.max_hp
                healed = t.heal(amount)
                self.emit("heal", source=actor.name, target=t.name, amount=healed)
            return
        if kind == "shield":
            for t in self._allied_targets(actor, chosen.get("targets", "self")):
                amount = chosen.get("value", 0.0) * t.max_hp
                t.add_shield(amount, name=chosen.get("name", "Shield"),
                             duration_turns=chosen.get("duration_turns", 2))
                self.emit("event", msg=f"{actor.name} shields {t.name}")
            return

        # attacking action (power scaling applied in `enemy_attack`, the
        # single choke point for all enemy damaging actions)
        t = chosen.get("targets", "single")
        mult = chosen.get("mult", 1.0) * float(chosen.get("damage_mult", 1.0))
        element = actor.element
        hit_energy = chosen.get("energy_to_hit", C.ENERGY_ON_HIT_MEDIUM)
        if t == "single":
            target = self.pick_enemy_target(actor, "single")
            if target:
                self.enemy_attack(actor, target, mult, element, hit_energy)
        elif t == "blast":
            target = self.pick_enemy_target(actor, "single")
            if target:
                self.enemy_attack(actor, target, mult, element, hit_energy)
                for adj in self.adjacent_of(target):
                    self.enemy_attack(actor, adj, mult * 0.5, element, hit_energy)
        elif t == "aoe":
            for a in self.enemies_of(actor):
                self.enemy_attack(actor, a, mult, element, hit_energy)

    def _allied_targets(self, actor: Unit, pattern: str) -> list[Unit]:
        """Ally-side targeting for enemy support actions (heal/shield/buff)."""
        allies = [u for u in (self.side_a.units if actor.side == "A" else self.side_b.units)
                  if u.alive]
        if pattern == "self" or not allies:
            return [actor] if actor.alive else []
        if pattern == "all":
            return allies
        if pattern == "lowest_hp":
            return [min(allies, key=lambda u: u.hp_fraction())]
        return [self.rng.choice(allies)]

    def pick_enemy_target(self, actor: Unit, pattern: str) -> Unit | None:
        targets = self.enemies_of(actor)
        if not targets:
            return None
        weights = [max(0.1, u.base.taunt + u.statuses.stat_total("aggro")) for u in targets]
        return self.rng.choices(targets, weights=weights, k=1)[0]

    def enemy_attack(self, attacker: Unit, target: Unit, mult: float,
                     element: str, hit_energy: float) -> None:
        target.gain_energy(hit_energy, affected_by_regen=False)
        # Character protection only: enemy-vs-enemy duels fight at lore mults
        # against their (scaled) HP pools.
        power = C.ENEMY_ACTION_POWER_SCALE if target.is_character else 1.0
        self.deal_damage(attacker, target, mult * power,
                         element, "basic", 0, can_crit=True)
        # March 7th counter hook
        self.on_ally_attacked_hooks(attacker, target)

    # ------------------------------------------------------------------ #
    # talent / trigger hooks                                             #
    # ------------------------------------------------------------------ #
    def on_ally_attacked_hooks(self, attacker: Unit, target: Unit) -> None:
        """Called when an enemy attack lands on character `target`."""
        if not target.is_character or not target.alive:
            return
        talent = target.kit.get("talent", {})
        # March 7th style counter
        if talent.get("kind") == "counter" and target.shield > 0:
            limit = talent.get("per_turn_limit", 2)
            used = target.runtime.get("counter_used", 0)
            if used < limit and attacker.alive:
                target.runtime["counter_used"] = used + 1
                self.emit("fua", actor=target.name, name="Counter")
                self.deal_damage(target, attacker, talent.get("mult", 0.9), target.element,
                                 "fua", talent.get("toughness", 30))
                target.gain_energy(talent.get("energy_gain", 10))

    def on_weakness_break_hooks(self, breaker: Unit, target: Unit, element: str) -> None:
        """Himeko charge gain on any weakness break."""
        for u in self.allies_of(breaker) + ([breaker] if breaker.is_character else []):
            if not u.is_character:
                continue
            talent = u.kit.get("talent", {})
            if talent.get("kind") == "fua" and talent.get("trigger") == "enemy_weakness_broken":
                charges = u.runtime.get("himeko_charges", 0) + 1
                u.runtime["himeko_charges"] = min(charges, talent.get("charge_max", 3))
                self.try_himeko_fua(u)

    def try_himeko_fua(self, u: Unit) -> None:
        talent = u.kit.get("talent", {})
        if u.runtime.get("himeko_charges", 0) >= talent.get("charge_max", 3):
            if u.runtime.get("fua_done"):
                return
            u.runtime["fua_done"] = True
            u.runtime["himeko_charges"] = 0
            self.emit("fua", actor=u.name, name="Victory Rush")
            for f in self.enemies_of(u):
                self.deal_damage(u, f, talent.get("mult", 1.32), u.element, "fua",
                                 talent.get("toughness", 30))
            u.gain_energy(talent.get("energy_gain", 10))

    def after_turn_hooks(self, actor: Unit) -> None:
        """Hooks after a turn fully resolves: Fu Xuan auto field, FUA triggers,
        lightning lord, Feixiao stacks, Acheron stacks, phases."""
        # conditional follow-ups (Herta / Yanqing style)
        self.trigger_conditional_fuas(actor)

        # reset per-turn counters at the end of every turn
        for u in self.all_alive():
            if u.is_character:
                u.runtime["counter_used"] = 0
                u.runtime["fua_done"] = False

        # Luocha auto heal
        for u in self.all_alive():
            if not (u.is_character and u.kit.get("talent", {}).get("kind") == "auto_field"):
                continue
            limit = u.kit["talent"].get("per_turn_limit", 2)
            used = self.auto_field_used.get(u.id, 0)
            if used >= limit:
                continue
            for a in self.allies_of(u):
                if a.hp_fraction() < 0.5 and a.alive:
                    self.auto_field_used[u.id] = used + 1
                    amount = u.max_hp * u.kit["talent"].get("heal_pct", 0.18)
                    a.heal(amount)
                    self.emit("heal", source=u.name, target=a.name, amount=amount)
                    break

        # black swan talent: FUA at enemy turn start handled in run_turn of enemies
        if actor.is_enemy and actor.alive:
            for ch in self.side_a.alive_units() + self.side_b.alive_units():
                if not ch.is_character:
                    continue
                talent = ch.kit.get("talent", {})
                if talent.get("kind") == "wind_shear_fua" and not ch.runtime.get("fua_done"):
                    ws = actor.statuses.find("Wind Shear")
                    if ws:
                        ch.runtime["fua_done"] = True
                        self.emit("fua", actor=ch.name, name="Loom of Fate's Caprice")
                        mult = talent.get("mult_per_stack", 0.192) * ws.stacks
                        self.deal_damage(ch, actor, mult, ch.element, "fua", 0)
                        ch.gain_energy(talent.get("energy_gain", 5))

        # follow-up triggers keyed on the actor having just acted
        if actor.is_character and actor.alive:
            talent = actor.kit.get("talent", {})
            # Feixiao / Jing Yuan style: gain stacks on attack
            if talent.get("kind") == "flying_aureus":
                actor.runtime["flying_aureus"] = min(
                    12, actor.runtime.get("flying_aureus", 0) + 2)
            if talent.get("kind") == "crimson_knot":
                for f in self.enemies_of(actor):
                    if f.statuses.all_of("debuff") or f.statuses.dots():
                        pass

        # Acheron stacks: any debuffed enemy attacked by anyone -> +1 knot
        for ch in self.all_alive():
            if ch.is_character and ch.kit.get("ultimate", {}).get("kind") == "acheron_ult":
                for f in self.enemies_of(ch):
                    if (f.statuses.all_of("debuff") or f.statuses.dots()) and f.alive:
                        ch.runtime["crimson_knot_stacks"] = min(
                            9, ch.runtime.get("crimson_knot_stacks", 0) + 1)
                        break

        # Feixiao: ally attacks give stacks; at 12, fire FUA
        for ch in self.all_alive():
            if not (ch.is_character and ch.kit.get("talent", {}).get("kind") == "flying_aureus"):
                continue
            if actor.is_character and actor is not ch:
                ch.runtime["flying_aureus"] = min(12, ch.runtime.get("flying_aureus", 0) + 1)
            if actor.is_character and actor is ch:
                ch.runtime["flying_aureus"] = min(12, ch.runtime.get("flying_aureus", 0) + 1)
            if ch.runtime.get("flying_aureus", 0) >= 12 and not ch.runtime.get("fua_done"):
                ch.runtime["fua_done"] = True
                ch.runtime["flying_aureus"] = 0
                talent = ch.kit["talent"]
                self.emit("fua", actor=ch.name, name="Flying Aureus")
                target = self.random_enemy(ch)
                if target:
                    for _ in range(6):
                        if target.alive:
                            self.deal_damage(ch, target, talent.get("mult", 0.585),
                                             ch.element, "fua", talent.get("toughness", 30))
                    final = self.random_enemy(ch)
                    if final:
                        self.deal_damage(ch, final, talent.get("mult", 0.585) * 2,
                                         ch.element, "fua", talent.get("toughness", 30))

        # Lightning-Lord acts when its AV comes due (handled via summon unit list)
        self.lightning_lord_turn(actor)
        self.check_phases(actor)
        self.check_battle_end()

    def spawn_lightning_lord(self, owner: Unit) -> None:
        summon = owner.kit.get("talent", {}).get("summon", {})
        owner.runtime["lightning_lord"] = {
            "hits": summon.get("hits", 3), "spd": summon.get("spd", 60),
            "av": C.base_av(summon.get("spd", 60)),
        }
        self.emit("event", msg=f"{owner.name} summons {summon.get('name', 'Lightning-Lord')}")

    def lightning_lord_turn(self, actor: Unit) -> None:
        """Lightning-Lord accumulates AV; when due, unleash hits (FUA type)."""
        for ch in self.all_alive():
            ll = ch.runtime.get("lightning_lord")
            if not ll:
                continue
            if ch.kit.get("talent", {}).get("kind") != "summon":
                continue
            ll["av"] -= 0  # advanced externally
            # accumulate AV with global time progress: approximate by tying to turns
            ll["acc"] = ll.get("acc", 0.0) + (C.base_av(ch.effective_spd()) * 0.5)
            if ll["acc"] >= ll["av"]:
                ll["acc"] = 0.0
                summon = ch.kit["talent"]["summon"]
                self.emit("fua", actor=f"{ch.name} ({summon.get('name', 'Lightning-Lord')})",
                          name="Lightning-Lord")
                for _ in range(ll["hits"]):
                    t = self.random_enemy(ch)
                    if t:
                        self.deal_damage(ch, t, summon.get("hit_mult", 0.66),
                                         ch.element, "fua", 30)
                        for adj in self.adjacent_of(t):
                            self.deal_damage(ch, adj, summon.get("hit_mult", 0.66)
                                             * summon.get("adjacent_ratio", 0.25),
                                             ch.element, "fua", 0)

    def check_phases(self, actor: Unit) -> None:
        """Boss phase transitions: next phase = full HP/toughness + immediate action."""
        for boss in self.units:
            if not boss.is_enemy or boss.total_phases <= 1:
                continue
            if not boss.alive and boss.phase < boss.total_phases:
                boss.phase += 1
                boss.alive = True
                boss.hp = boss.max_hp
                boss.recover_from_break()
                boss.av = 0.0   # immediately take action
                self.emit("phase", target=boss.name, phase=boss.phase)

    # energy convenience for agents
    def try_revives(self) -> None:
        """Bailu/Gepard-style one-time revive of freshly downed allies."""
        for u in self.units:
            if u.alive or not u.is_character:
                continue
            side = self.side_a if u.side == "A" else self.side_b
            for mate in side.units:
                if not mate.alive or not mate.is_character:
                    continue
                talent = mate.kit.get("talent", {})
                eff = talent.get("revive_heal")
                if eff and not mate.runtime.get("revive_used"):
                    self.apply_effect(mate, u, eff)
                    break
            # Bailu's talent declares revive directly on kind
            if not u.alive and u.kit.get("talent", {}).get("kind") == "revive" and \
                    not u.runtime.get("revive_used"):
                u.runtime["revive_used"] = True
                u.alive = True
                u.hp = u.max_hp * float(u.kit["talent"].get("value", 0.5))
                self.emit("heal", source=u.name, target=u.name, amount=u.hp, revive=True)

    def trigger_conditional_fuas(self, actor: Unit) -> None:
        """Herta (enemy_below_half) and Yanqing (after_ally_attack) style FUAs."""
        if not actor.is_character or not actor.alive:
            return
        for ch in self.side_a.alive_units() + self.side_b.alive_units():
            if not ch.is_character:
                continue
            talent = ch.kit.get("talent", {})
            if talent.get("kind") != "fua" or ch.runtime.get("fua_done"):
                continue
            trigger = talent.get("trigger")
            if trigger == "enemy_below_half":
                target = next((f for f in self.enemies_of(ch)
                               if f.hp_fraction() <= 0.5), None)
                if target:
                    ch.runtime["fua_done"] = True
                    self.emit("fua", actor=ch.name, name=talent.get("name", "FUA"))
                    self.deal_damage(ch, target, talent.get("mult", 0.88), ch.element,
                                     "fua", talent.get("toughness", 30))
                    ch.gain_energy(talent.get("energy_gain", 5))
            elif trigger == "after_ally_attack" and ch is not actor:
                if self.rng.random() < talent.get("chance", 0.5):
                    ch.runtime["fua_done"] = True
                    target = self.random_enemy(ch)
                    if target:
                        self.emit("fua", actor=ch.name, name=talent.get("name", "FUA"))
                        self.deal_damage(ch, target, talent.get("mult", 0.5), ch.element,
                                         "fua", talent.get("toughness", 30))
                        ch.gain_energy(talent.get("energy_gain", 10))

    def snapshot(self) -> dict[str, Any]:
        def unit_view(u: Unit) -> dict[str, Any]:
            return {
                "id": u.id, "name": u.name, "side": u.side, "index": u.index,
                "element": u.element, "hp": round(u.hp, 1),
                "max_hp": round(u.max_hp, 1), "energy": round(u.energy, 1),
                "energy_max": u.base.energy_max, "shield": round(u.shield, 1),
                "toughness": u.toughness, "max_toughness": u.max_toughness,
                "weakness_broken": u.weakness_broken, "weaknesses": list(u.weaknesses),
                "av": round(u.av, 1), "spd": round(u.effective_spd(), 1),
                "statuses": [
                    {"name": e.name, "kind": e.kind, "turns": e.duration_turns,
                     "stacks": e.stacks}
                    for e in u.statuses.all_of()
                ],
            }
        return {
            "time": round(self.time, 1), "skill_points": self.skill_points,
            "side_a": [unit_view(u) for u in self.side_a.units],
            "side_b": [unit_view(u) for u in self.side_b.units],
            "finished": self.finished, "winner": self.winner,
        }
