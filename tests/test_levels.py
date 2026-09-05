"""Tests for per-level stat resolution (datamine curves + enemy scaling)."""

import pytest

from cli_hsr import db, levels as LV


@pytest.fixture(scope="module", autouse=True)
def seeded_db():
    db.seed()


def _get_row(uid):
    row = db.get_unit(uid)
    assert row is not None, f"missing unit {uid}"
    return row


class TestPromotionCurves:
    def test_datamine_curve_matches_official_wiki_table(self):
        """stat(L) = base_tier + step*(L-1), verified against the official
        wiki ascension table for Seele (floored display values):
        Lv1: 126/87/49, Lv80: 931/640/363."""
        import json
        dm = json.load(open("data/datamine_characters.json", encoding="utf-8"))
        curve = next(c for c in dm["characters"] if c["id"] == "seele")
        curve = curve["stats"]["promotion_curve"]["values"]
        assert LV.promotion_stat(curve, "hp", 1) == pytest.approx(126, abs=1.0)
        assert LV.promotion_stat(curve, "atk", 1) == pytest.approx(87, abs=1.0)
        assert LV.promotion_stat(curve, "def", 1) == pytest.approx(49, abs=1.0)
        assert LV.promotion_stat(curve, "hp", 80) == pytest.approx(931, abs=1.0)
        assert LV.promotion_stat(curve, "atk", 80) == pytest.approx(640, abs=1.0)
        assert LV.promotion_stat(curve, "def", 80) == pytest.approx(363, abs=1.0)
        # mid-ascension boundary (Lv20 post-ascension): wiki 297
        assert LV.promotion_stat(curve, "hp", 20) == pytest.approx(297, abs=1.0)

    def test_imported_unit_stats_at_80(self):
        row = _get_row("dan_heng")  # datamine-only unit (no curated kit)
        stats = LV.character_stats_at_level(row["stats_json"], 80)
        # wiki Dan Heng Lv.80: HP 882, ATK 547, DEF 397, SPD 110
        assert stats["max_hp"] == pytest.approx(882, abs=1.0)
        assert stats["atk"] == pytest.approx(547, abs=1.0)
        assert stats["def"] == pytest.approx(397, abs=1.0)
        assert stats["spd"] == pytest.approx(110, abs=0.01)
        assert stats["energy_max"] == pytest.approx(100)

    def test_curated_seele_keeps_handcrafted_stats(self):
        """Curated kits take precedence: seele's stats come from the
        hand-crafted file, not the datamine curve."""
        row = _get_row("seele")
        assert not LV.has_promotion_curve(row["stats_json"])
        assert row["stats_json"]["spd"] == 115

    def test_level_monotonic_and_bounds(self):
        row = _get_row("dan_heng")
        curve = row["stats_json"]["promotion_curve"]["values"]
        hp1 = LV.promotion_stat(curve, "hp", 1)
        hp40 = LV.promotion_stat(curve, "hp", 40)
        hp80 = LV.promotion_stat(curve, "hp", 80)
        assert hp1 < hp40 < hp80
        # clamped outside 1..80
        assert LV.promotion_stat(curve, "hp", 0) == hp1
        assert LV.promotion_stat(curve, "hp", 99) == hp80

    def test_tier_jumps_at_ascension_levels(self):
        """Base stat jumps at each ascension tier boundary (20/30/.../70)."""
        row = _get_row("dan_heng")
        curve = row["stats_json"]["promotion_curve"]["values"]
        prev = LV.promotion_stat(curve, "atk", 19)
        for lvl in (20, 30, 40, 50, 60, 70):
            cur = LV.promotion_stat(curve, "atk", lvl)
            assert cur > prev, f"no ascension jump at {lvl}"
            prev = cur

    def test_unit_uses_level_for_imported_characters(self):
        from cli_hsr.units import build_unit
        row = _get_row("dan_heng")
        u40 = build_unit(row, "A", 0, level=40)
        u80 = build_unit(row, "A", 0, level=80)
        assert u80.base.atk > u40.base.atk
        assert u80.base.max_hp > u40.base.max_hp
        assert u40.base.spd == u80.base.spd  # SPD unscaled by level

    def test_curated_characters_keep_static_stats(self):
        from cli_hsr.units import build_unit
        row = _get_row("acheron")  # hand-crafted kit, no curve
        u = build_unit(row, "A", 0, level=40)
        u80 = build_unit(row, "A", 0, level=80)
        assert u.base.atk == u80.base.atk  # static stats ignore level


class TestEnemyScaling:
    def test_enemy_stats_scale_with_level(self):
        row = _get_row("blaze_out_of_space")
        s40 = LV.enemy_stats_at_level(row["stats_json"], 40)
        s80 = row["stats_json"]
        assert s40["max_hp"] < s80["max_hp"]
        assert s40["atk"] < s80["atk"]

    def test_enemy_level_multiplier_reference(self):
        from cli_hsr.constants import level_multiplier
        assert LV.enemy_scale(80) == pytest.approx(1.0)
        assert LV.enemy_scale(40) == pytest.approx(
            level_multiplier(40) / level_multiplier(80))

    def test_enemy_unit_at_low_level_weaker(self):
        from cli_hsr.units import build_unit
        row = _get_row("blaze_out_of_space")
        u40 = build_unit(row, "B", 0, level=40)
        u80 = build_unit(row, "B", 0, level=80)
        assert u40.base.max_hp < u80.base.max_hp
        assert u40.effective_atk() < u80.effective_atk()
