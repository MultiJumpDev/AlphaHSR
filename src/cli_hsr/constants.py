"""Exact Honkai: Star Rail combat constants.

All values are taken from the Honkai: Star Rail Wiki (Damage, Toughness,
Speed, Energy pages) and reproduced faithfully for the simulation engine.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
# Elements (Combat Types) and Paths                                   #
# ------------------------------------------------------------------ #

class Element:
    PHYSICAL = "Physical"
    FIRE = "Fire"
    ICE = "Ice"
    LIGHTNING = "Lightning"
    WIND = "Wind"
    QUANTUM = "Quantum"
    IMAGINARY = "Imaginary"


ALL_ELEMENTS = [
    Element.PHYSICAL, Element.FIRE, Element.ICE, Element.LIGHTNING,
    Element.WIND, Element.QUANTUM, Element.IMAGINARY,
]

ELEMENT_ALIASES = {
    "phys": Element.PHYSICAL,
    "physical": Element.PHYSICAL,
    "fire": Element.FIRE,
    "ice": Element.ICE,
    "lightning": Element.LIGHTNING,
    "wind": Element.WIND,
    "quantum": Element.QUANTUM,
    "imaginary": Element.IMAGINARY,
}


class Path:
    DESTRUCTION = "Destruction"
    THE_HUNT = "The Hunt"
    ERUDITION = "Erudition"
    HARMONY = "Harmony"
    NIHILITY = "Nihility"
    PRESERVATION = "Preservation"
    ABUNDANCE = "Abundance"
    REMEMBRANCE = "Remembrance"


ALL_PATHS = [
    Path.DESTRUCTION, Path.THE_HUNT, Path.ERUDITION, Path.HARMONY,
    Path.NIHILITY, Path.PRESERVATION, Path.ABUNDANCE, Path.REMEMBRANCE,
]

# ------------------------------------------------------------------ #
# Stat names                                                          #
# ------------------------------------------------------------------ #

class Stat:
    MAX_HP = "Max HP"
    ATK = "ATK"
    DEF = "DEF"
    SPD = "SPD"
    CRIT_RATE = "CRIT Rate"
    CRIT_DMG = "CRIT DMG"
    EFFECT_HIT = "Effect Hit Rate"
    EFFECT_RES = "Effect RES"
    BREAK_EFFECT = "Break Effect"
    ENERGY_RATE = "Energy Regeneration Rate"
    HEALING_BOOST = "Outgoing Healing Boost"
    AGGRO = "Aggro"
    DMG_BOOST = "DMG Boost"
    RES_PEN = "RES PEN"
    DEF_PEN = "DEF PEN"
    WEAKEN = "Weaken"
    DMG_REDUCTION = "DMG Reduction"
    VULNERABILITY = "Vulnerability"
    TOUGHNESS_VULN = "Toughness Vulnerability"
    BREAK_DMG_VULN = "Break DMG Vulnerability"
    DOT_VULN = "DoT Vulnerability"


# ------------------------------------------------------------------ #
# Level multiplier (Toughness wiki page) — exact table                #
# ------------------------------------------------------------------ #

LEVEL_MULTIPLIER_TABLE: dict[int, float] = {
    1: 54.0000, 2: 58.0000, 3: 62.0000, 4: 67.5264, 5: 70.5094,
    6: 73.5228, 7: 76.5660, 8: 79.6385, 9: 82.7395, 10: 85.8684,
    11: 91.4944, 12: 97.0680, 13: 102.5892, 14: 108.0579,
    15: 113.4743, 16: 118.8383, 17: 124.1499, 18: 129.4091, 19: 134.6159,
    20: 139.7703, 21: 149.3323, 22: 158.8011, 23: 168.1768, 24: 177.4594,
    25: 186.6489, 26: 195.7452, 27: 204.7484, 28: 213.6585, 29: 222.4754,
    30: 231.1992, 31: 246.4276, 32: 261.1810, 33: 275.4733,
    34: 289.3179, 35: 302.7275, 36: 315.7144, 37: 328.2905, 38: 340.4671,
    39: 352.2554, 40: 363.6658, 41: 408.1240, 42: 451.7883, 43: 494.6798,
    44: 536.8188, 45: 578.2249, 46: 618.9172, 47: 658.9138, 48: 698.2325,
    49: 736.8905, 50: 774.9041, 51: 871.0599, 52: 964.8705, 53: 1056.4206,
    54: 1145.7910, 55: 1233.0585, 56: 1318.2965, 57: 1401.5750,
    58: 1482.9608, 59: 1562.5178, 60: 1640.3068,
    61: 1752.3215, 62: 1861.9011, 63: 1969.1242, 64: 2074.0659,
    65: 2176.7983, 66: 2277.3904, 67: 2375.9085, 68: 2472.4160,
    69: 2566.9739, 70: 2659.6406, 71: 2780.3044, 72: 2898.6022,
    73: 3014.6029, 74: 3128.3729, 75: 3239.9758, 76: 3349.4730,
    77: 3456.9236, 78: 3562.3843, 79: 3665.9099, 80: 3767.5533,
    81: 3957.8618, 82: 4155.2118, 83: 4359.8638, 84: 4572.0878,
    85: 4792.1641, 86: 5020.3833, 87: 5257.0466, 88: 5502.4664,
    89: 5756.9667, 90: 6020.8836, 91: 6294.5654, 92: 6578.3734,
    93: 6872.6823, 94: 7177.8806, 95: 7494.3713,
}


def level_multiplier(level: float) -> float:
    """Wiki Level Multiplier, linearly interpolated for missing levels."""
    table = LEVEL_MULTIPLIER_TABLE
    if level in table:
        return table[level]
    keys = sorted(table)
    if level <= keys[0]:
        return table[keys[0]]
    if level >= keys[-1]:
        return table[keys[-1]]
    lo, hi = 0, len(keys) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if keys[mid] <= level:
            lo = mid
        else:
            hi = mid
    l0, l1 = keys[lo], keys[hi]
    v0, v1 = table[l0], table[l1]
    return v0 + (v1 - v0) * (level - l0) / (l1 - l0)


# ------------------------------------------------------------------ #
# Weakness break constants                                            #
# ------------------------------------------------------------------ #

# Break Base DMG coefficients (x Level Multiplier x Max Toughness Multiplier)
BREAK_BASE_DMG: dict[str, float] = {
    Element.PHYSICAL: 2.0,
    Element.FIRE: 2.0,
    Element.ICE: 1.0,
    Element.LIGHTNING: 1.0,
    Element.WIND: 1.5,
    Element.QUANTUM: 0.5,
    Element.IMAGINARY: 0.5,
}

# Debuff applied by a weakness break, per element: (name, is_dot, turns)
BREAK_DEBUFF: dict[str, tuple[str, bool, int]] = {
    Element.PHYSICAL: ("Bleed", True, 2),
    Element.FIRE: ("Burn", True, 2),
    Element.ICE: ("Freeze", False, 1),
    Element.LIGHTNING: ("Shock", True, 2),
    Element.WIND: ("Wind Shear", True, 2),
    Element.QUANTUM: ("Entanglement", False, 1),
    Element.IMAGINARY: ("Imprisonment", False, 1),
}

# Wind Shear stacks: 1 on normal enemies, 3 on elite/boss (max 5)
WIND_SHEAR_STACKS_NORMAL = 1
WIND_SHEAR_STACKS_ELITE = 3
WIND_SHEAR_MAX_STACKS = 5

# Entanglement: +20% AV delay x (1 + Break Effect) per break; 1 stack per hit
ENTANGLEMENT_DELAY = 0.20
ENTANGLEMENT_MAX_STACKS = 5

# Imprisonment: 30% AV delay x (1+BE), -10% SPD
IMPRISONMENT_DELAY = 0.30
IMPRISONMENT_SPD_REDUCTION = 0.10

# Bleed cap: 2 x Level Multiplier x Max Toughness Multiplier
BLEED_CAP_COEFFICIENT = 2.0
BLEED_HP_COEFFICIENT_NORMAL = 0.16
BLEED_HP_COEFFICIENT_ELITE = 0.07

# Weakness break universally delays the enemy's next action by 25%
WEAKNESS_BREAK_ACTION_DELAY = 0.25

# Universal incoming-damage multiplier while the target still has toughness
BROKEN_MULTIPLIER_TOUGH = 0.9

# Super Break: (toughness reduction / 10) x Level Multiplier x ...
SUPER_BREAK_TOUGHNESS_DIVISOR = 10.0


def max_toughness_multiplier(max_toughness: float) -> float:
    """Max Toughness Multiplier = 0.5 + Max Toughness / 40."""
    return 0.5 + max_toughness / 40.0


# ------------------------------------------------------------------ #
# Toughness damage (BTO) per attack archetype.                        #
# The wiki lists exact BTO per skill; archetypes below are the        #
# standard values used across the roster data files.                  #
# ------------------------------------------------------------------ #

BTO_BY_ACTION: dict[str, float] = {
    "basic": 30.0,
    "skill": 60.0,
    "ultimate": 90.0,
    "fua": 30.0,
    "counter": 30.0,
    "additional": 0.0,
    "dot": 0.0,
    "break": 0.0,
    "true": 0.0,
}


# ------------------------------------------------------------------ #
# Action value / speed                                                #
# ------------------------------------------------------------------ #

BASE_ACTION_VALUE = 10000.0


def base_av(speed: float) -> float:
    """Base Action Value = 10000 / SPD."""
    if speed <= 0:
        return float("inf")
    return BASE_ACTION_VALUE / speed


# Enemy SPD multiplier by level (Speed wiki page). Kept for reference:
# the enemy JSON already stores final in-game speeds, so the engine does
# NOT apply this multiplier.
def enemy_speed_multiplier(level: int) -> float:
    if level >= 86:
        return 1.32
    if level >= 78:
        return 1.2
    if level >= 65:
        return 1.1
    return 1.0


# Balance knob: enemy HP in the JSON is lore-accurate (assumes geared endgame
# characters). Simulator units field TRUE BASE stats (no relics/cones) dealing
# ~300-800 per action, so enemy HP is scaled down to keep battles in the
# 10-40 turn range. Category-aware: bosses carry ~12x normal HP in the JSON
# which drags gearless duels past 50 turns, so their scale is gentler.
ENEMY_HP_SCALE = 0.12          # fallback / normal
ENEMY_HP_SCALE_ELITE = 0.05
ENEMY_HP_SCALE_BOSS = 0.045

# Balance knob: mirror of the above for fights where ENEMIES are played BY an
# agent against characters (anyone-vs-anyone duels). Enemy attack actions are
# scaled by this factor so a boss piloted by a model doesn't one-shot a
# gearless character (boss ATK ~900 vs character HP ~1000).
ENEMY_ACTION_POWER_SCALE = 0.25


# Cycle length for timed modes: first cycle 150 AV, then 100 AV each
FIRST_CYCLE_AV = 150.0
CYCLE_AV = 100.0

# ------------------------------------------------------------------ #
# Energy                                                              #
# ------------------------------------------------------------------ #

ENERGY_ON_BASIC = 20.0
ENERGY_ON_SKILL = 30.0
ENERGY_ON_ULTIMATE = 5.0
ENERGY_ON_MEMOSPRITE_SKILL = 10.0
ENERGY_ON_KILL = 10.0

# Being hit: usually 5/10/15/20/25 depending on the enemy attack weight
ENERGY_ON_HIT_SMALL = 5.0
ENERGY_ON_HIT_MEDIUM = 10.0
ENERGY_ON_HIT_LARGE = 15.0
ENERGY_ON_HIT_HUGE = 25.0


# ------------------------------------------------------------------ #
# Damage formula defaults                                             #
# ------------------------------------------------------------------ #

# Enemies: DEF = 200 + 10 x level
def enemy_def(level: int) -> float:
    return 200.0 + 10.0 * level


# Standard base chance for break debuffs and many skill debuffs
BASE_CHANCE_BREAK_DEBUFF = 1.50  # 150% base chance for weakness break debuffs

# Crit
DEFAULT_CRIT_RATE = 0.05
DEFAULT_CRIT_DMG = 1.0

# Combat variant default level for both sides
DEFAULT_LEVEL = 80
