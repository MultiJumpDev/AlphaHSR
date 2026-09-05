"""SQLite game database with automatic seeding from JSON files."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "game.db"

_local = threading.local()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_conn(db_path: str | Path | None = None) -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None or db_path is not None:
        conn = _connect(db_path)
        _local.conn = conn
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS units (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    unit_type   TEXT NOT NULL CHECK (unit_type IN ('character', 'normal', 'elite', 'boss')),
    element     TEXT NOT NULL,
    path        TEXT,
    rarity      INTEGER,
    category    TEXT,
    stats_json  TEXT NOT NULL,
    kit_json    TEXT NOT NULL,
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_units_type ON units(unit_type);

CREATE TABLE IF NOT EXISTS match_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    played_at   TEXT NOT NULL DEFAULT (datetime('now')),
    team_a      TEXT NOT NULL,
    team_b      TEXT NOT NULL,
    agent_a     TEXT NOT NULL,
    agent_b     TEXT NOT NULL,
    winner      TEXT NOT NULL CHECK (winner IN ('A', 'B', 'draw')),
    rounds      INTEGER,
    seed        INTEGER,
    log_json    TEXT
);

CREATE TABLE IF NOT EXISTS tournament_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    played_at     TEXT NOT NULL DEFAULT (datetime('now')),
    tournament    TEXT NOT NULL,
    round_name    TEXT NOT NULL,
    participants  TEXT NOT NULL,
    winner        TEXT NOT NULL,
    scoreboard    TEXT
);

CREATE TABLE IF NOT EXISTS light_cones (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    rarity      INTEGER,
    path        TEXT,
    data_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relic_sets (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    data_json   TEXT NOT NULL
);
"""


def seed(db_path: str | Path | None = None) -> None:
    """Load characters.json and enemies.json into the SQLite DB (idempotent)."""
    conn = _get_conn(db_path)
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM units WHERE 1=1")

    rows: list[tuple] = []
    curated_ids: set[str] = set()
    for char_filename in ("characters.json", "characters_new.json"):
        char_file = DATA_DIR / char_filename
        if char_file.exists():
            data = json.loads(char_file.read_text(encoding="utf-8"))
            for c in data["characters"]:
                curated_ids.add(c["id"])
                rows.append((
                    c["id"], c["name"], "character", c["element"], c.get("path"),
                    c.get("rarity"), None, json.dumps(c["stats"]), json.dumps(c),
                    c.get("notes", ""),
                ))

    # datamine roster (Mar-7th/StarRailRes): fills every playable character not
    # covered by a hand-crafted kit above.
    dm_file = DATA_DIR / "datamine_characters.json"
    if dm_file.exists():
        data = json.loads(dm_file.read_text(encoding="utf-8"))
        for c in data["characters"]:
            if c["id"] in curated_ids:
                continue
            stats = dict(c["stats"])
            stats.setdefault("def", None)
            rows.append((
                c["id"], c["name"], "character", c["element"], c.get("path"),
                c.get("rarity"), None, json.dumps(stats), json.dumps(c),
                c.get("notes", ""),
            ))

    enemy_file = DATA_DIR / "enemies.json"
    if enemy_file.exists():
        data = json.loads(enemy_file.read_text(encoding="utf-8"))
        for e in data["enemies"]:
            category = e.get("category", "normal")
            rows.append((
                e["id"], e["name"], category, e["element"], None, None,
                category, json.dumps(e["stats"]), json.dumps(e), e.get("notes", ""),
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO units "
        "(id, name, unit_type, element, path, rarity, category, stats_json, kit_json, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    # gear tables
    conn.execute("DELETE FROM light_cones")
    conn.execute("DELETE FROM relic_sets")
    cone_file = DATA_DIR / "light_cones.json"
    if cone_file.exists():
        data = json.loads(cone_file.read_text(encoding="utf-8"))
        conn.executemany(
            "INSERT OR REPLACE INTO light_cones (id, name, rarity, path, data_json) "
            "VALUES (?, ?, ?, ?, ?)",
            [(c["id"], c["name"], c.get("rarity"), c.get("path"), json.dumps(c))
             for c in data["light_cones"]],
        )
    relic_file = DATA_DIR / "relics.json"
    if relic_file.exists():
        data = json.loads(relic_file.read_text(encoding="utf-8"))
        conn.executemany(
            "INSERT OR REPLACE INTO relic_sets (id, name, data_json) VALUES (?, ?, ?)",
            [(r["id"], r["name"], json.dumps(r)) for r in data["relic_sets"]],
        )
    conn.commit()


def ensure_seeded(db_path: str | Path | None = None) -> None:
    conn = _get_conn(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    except sqlite3.OperationalError:
        count = 0
    if count == 0:
        seed(db_path)


def _parse_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    d["stats_json"] = json.loads(d["stats_json"])
    d["kit_json"] = json.loads(d["kit_json"])
    return d


def list_units(unit_type: str | None = None, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    ensure_seeded(db_path)
    conn = _get_conn(db_path)
    if unit_type:
        rows = conn.execute(
            "SELECT * FROM units WHERE unit_type = ? ORDER BY name", (unit_type,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM units ORDER BY unit_type, name").fetchall()
    return [dict(_parse_row(r)) for r in rows]


def get_unit(unit_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    ensure_seeded(db_path)
    conn = _get_conn(db_path)
    row = conn.execute("SELECT * FROM units WHERE id = ?", (unit_id,)).fetchone()
    return _parse_row(row)


def get_light_cone(cone_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    ensure_seeded(db_path)
    conn = _get_conn(db_path)
    row = conn.execute("SELECT data_json FROM light_cones WHERE id = ?", (cone_id,)).fetchone()
    return json.loads(row[0]) if row else None


def get_relic_set(set_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    ensure_seeded(db_path)
    conn = _get_conn(db_path)
    row = conn.execute("SELECT data_json FROM relic_sets WHERE id = ?", (set_id,)).fetchone()
    return json.loads(row[0]) if row else None


def list_light_cones(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    ensure_seeded(db_path)
    conn = _get_conn(db_path)
    rows = conn.execute("SELECT id, name, rarity, path FROM light_cones ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def list_relic_sets(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    ensure_seeded(db_path)
    conn = _get_conn(db_path)
    rows = conn.execute("SELECT id, name FROM relic_sets ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def character_ids(db_path: str | Path | None = None) -> list[str]:
    return [r["id"] for r in list_units("character", db_path)]


def enemy_ids(db_path: str | Path | None = None) -> list[str]:
    conn = _get_conn(db_path)
    ensure_seeded(db_path)
    return [
        r["id"] for r in conn.execute(
            "SELECT id FROM units WHERE unit_type IN ('normal', 'elite', 'boss')"
        )
    ]


def save_match_result(
    team_a: str, team_b: str, agent_a: str, agent_b: str,
    winner: str, rounds: int, seed_val: int | None, log: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO match_results (team_a, team_b, agent_a, agent_b, winner, rounds, seed, log_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (team_a, team_b, agent_a, agent_b, winner, rounds, seed_val, log),
    )
    conn.commit()


def save_tournament_result(
    tournament: str, round_name: str, participants: list[str],
    winner: str, scoreboard: str | None = None, db_path: str | Path | None = None,
) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO tournament_results (tournament, round_name, participants, winner, scoreboard) "
        "VALUES (?, ?, ?, ?, ?)",
        (tournament, round_name, json.dumps(participants), winner, scoreboard),
    )
    conn.commit()
