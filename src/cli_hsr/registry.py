"""Checkpoint registry: persisted trained models + their metadata.

A *bundle* is a directory containing:
  - model.zip    (the SB3 checkpoint)
  - metadata.json (team, gear, algo, side, win-rate...)

`contestant_from_checkpoint` wraps a loaded model as a tournament
Contestant whose agent infers with the correct battle perspective.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINTS_DIR = Path("checkpoints")


@dataclass
class ModelMetadata:
    name: str
    algo: str
    team: list[str]
    gear: list[dict] = field(default_factory=list)
    level: int = 80
    max_av: float = 3000.0
    max_rounds: int = 80
    trained_timesteps: int = 0
    eval_stats: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    extra: dict = field(default_factory=dict)


def save_model_bundle(model: Any, metadata: ModelMetadata,
                      path: str | Path | None = None) -> Path:
    """Save model.zip + metadata.json under `path` (default checkpoints/<name>)."""
    p = Path(path) if path else CHECKPOINTS_DIR / metadata.name
    p.mkdir(parents=True, exist_ok=True)
    model.save(str(p / "model"))
    (p / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_model_bundle(path: str | Path, device: str = "auto") -> tuple[Any, ModelMetadata]:
    """Reload a bundle; returns (model, metadata)."""
    from sb3_contrib import MaskablePPO  # requires the rl extra

    p = Path(path)
    meta_raw = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
    meta = ModelMetadata(**{k: v for k, v in meta_raw.items()
                            if k in ModelMetadata.__dataclass_fields__})
    if meta.algo == "dqn":
        from .rl.maskable_dqn import MaskableDQN
        model = MaskableDQN.load(str(p / "model"), device=device,
                                 custom_objects={"_last_env_mask": None})
    else:
        model = MaskablePPO.load(str(p / "model"), device=device)
    return model, meta


def bundle_exists(path: str | Path) -> bool:
    p = Path(path)
    return (p / "model.zip").exists() and (p / "metadata.json").exists()


# ---------------------------------------------------------------------- #
# tournament integration                                                 #
# ---------------------------------------------------------------------- #
class CheckpointAgent:
    """Agent interface over a trained SB3 model; perspective-correct."""

    def __init__(self, model: Any, name: str = "TrainedModel") -> None:
        self.name = name
        self.model = model

    def choose(self, battle, actor, legal):
        if not legal:
            return {"kind": "basic", "target": None}
        from .env import battle_to_observation

        obs = battle_to_observation(battle, actor.side)
        mask = [False] * 32
        for i in range(min(len(legal), 32)):
            mask[i] = True
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        idx = int(action)
        return legal[idx] if idx < len(legal) else legal[0]

    def observe(self, *args: Any, **kwargs: Any) -> None:
        pass

    def end_battle(self, won: bool, draw: bool) -> None:
        pass


def contestant_from_checkpoint(path: str | Path, name: str | None = None,
                               device: str = "auto"):
    """Load a bundle and return a Contestant ready for `Tournament`."""
    from .tournament import Contestant

    model, meta = load_model_bundle(path, device=device)
    return Contestant(
        name=name or meta.name,
        team=meta.team,
        agent_factory=lambda: CheckpointAgent(model, name or meta.name),
        gear=meta.gear,
        seed=0,
    )
