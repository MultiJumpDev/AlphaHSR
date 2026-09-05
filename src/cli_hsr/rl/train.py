"""RL training pipelines on the HSR environment (Stable-Baselines3 + sb3-contrib).

Both algorithms support action masking (required by the discrete masked
action space): MaskablePPO (recommended default, `MaskableActorCriticPolicy`)
and MaskableDQN (discrete-only). Models are saved as SB3 checkpoints (zip)
and can be reloaded for tournaments via `cli_hsr.registry`.

Hardware: `TrainConfig(device="auto")` resolves to "cuda" when
`torch.cuda.is_available()` (e.g. on moLab GPU instances) and falls back to
"cpu" otherwise. Use `auto_device()` for explicit detection.

Checkpoints: `CheckpointCallback` writes intermediate checkpoints into
`TrainConfig.checkpoint_dir` (default `checkpoints/ppo_run/`, directories are
created automatically) every `TrainConfig.checkpoint_freq` steps. Final
models are packaged by `cli_hsr.registry.save_model_bundle` as
`model.zip` + `metadata.json`.

Hugging Face Hub export: `hf_token()` resolves auth non-interactively (the
``HF_TOKEN`` env var — injected automatically by moLab Remote Storage — or a
cached ``huggingface-cli login`` token); `hf_upload_bundle()` creates the
target repo (private by default, `exist_ok=True`) and pushes the bundle plus
optional intermediate checkpoints; `build_model_card()` renders the README.md
model card. All helpers degrade gracefully — a missing token, missing
``huggingface_hub``, or a network/auth failure is reported, never raised.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..agents import GreedyAgent, RandomAgent, random_action
from ..engine import Battle
from .rewards import RewardConfig, RewardShaper

TrainablePolicy = Any  # MaskablePPO | MaskableDQN when sb3 is installed

DEFAULT_CHECKPOINT_DIR = "checkpoints/ppo_run"
DEFAULT_CHECKPOINT_FREQ = 25_000


def _need_sb3() -> tuple:
    try:
        from sb3_contrib import MaskablePPO  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Training requires stable-baselines3 and sb3-contrib: "
            "pip install 'cli-hsr[rl]'"
        ) from e
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from .maskable_dqn import MaskableDQN
    return MaskablePPO, MaskableDQN, MaskableActorCriticPolicy


# ---------------------------------------------------------------------- #
# hardware / device detection                                            #
# ---------------------------------------------------------------------- #
def auto_device() -> str:
    """Return "cuda" when a CUDA GPU is available, else "cpu".

    Never raises: torch being absent (or broken) degrades to "cpu".
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch optional at import time
        return "cpu"


def resolve_device(device: str | None) -> str:
    """Map "auto"/None -> auto_device(); pass through "cuda"/"cpu"/... else."""
    if device is None or device == "auto":
        return auto_device()
    return device


def gpu_info() -> dict[str, str]:
    """Best-effort GPU description for logging/UI (empty dict on CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, "total_memory", 0) or 0
            return {
                "device": "cuda",
                "name": props.name,
                "vram_gb": f"{mem / 1024**3:.1f}",
                "torch": torch.__version__,
            }
        return {"device": "cpu", "torch": torch.__version__}
    except Exception:  # pragma: no cover
        return {"device": "cpu"}


# ---------------------------------------------------------------------- #
# opponent pool (self-play)                                              #
# ---------------------------------------------------------------------- #
def policy_fn_from(model: TrainablePolicy) -> Callable[[Battle, Any, list], dict]:
    """Wrap an SB3 model into the engine's chooser signature, with masking
    and the correct perspective at inference time."""

    def chooser(battle: Battle, actor, legal):
        if not legal:
            return {"kind": "basic", "target": None}
        from ..env import battle_to_observation
        side = actor.side
        obs = battle_to_observation(battle, side)
        mask = [False] * 32
        for i in range(min(len(legal), 32)):
            mask[i] = True
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        idx = int(action)
        return legal[idx] if idx < len(legal) else legal[0]

    return chooser


class SelfPlayOpponent:
    """Samples an opponent policy each episode from a difficulty pool.

    `probs` maps to (random, greedy, self-play) probabilities and is
    normalized; regions for unavailable pool members (self-play without a
    model) fall back to random.
    """

    def __init__(self, model: TrainablePolicy | None = None,
                 probs: Sequence[float] = (0.4, 0.4, 0.2), seed: int | None = None) -> None:
        self.model = model
        p = [float(x) for x in probs]
        total = sum(p)
        self.probs = [x / total for x in p] if total > 0 else [1 / 3, 1 / 3, 1 / 3]
        self.rng = random.Random(seed)
        self._current: Callable[[Battle, Any, list], dict] = random_action

    def sample(self) -> Callable[[Battle, Any, list], dict]:
        r = self.rng.random()
        p_random, p_greedy, p_self = self.probs
        if self.model is not None and r < p_self:
            self._current = policy_fn_from(self.model)
        elif r < p_self + p_greedy:
            agent = GreedyAgent()
            self._current = lambda b, a, legal: agent.choose(b, a, legal)
        else:
            self._current = random_action
        return self._current


# ---------------------------------------------------------------------- #
# training configs                                                       #
# ---------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    algo: str = "ppo"                 # "ppo" | "dqn"
    total_timesteps: int = 100_000
    n_envs: int = 1
    # ppo
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    # dqn
    buffer_size: int = 100_000
    learning_starts: int = 1000
    train_freq: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = 0.3
    # self-play
    self_play_prob: float = 0.2
    # io
    device: str = "auto"              # "auto" -> cuda when available, else cpu
    log_dir: str = "runs/"
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR   # e.g. checkpoints/ppo_molab
    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ # env steps between checkpoints (0 = off)
    verbose: int = 1
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def make_model(config: TrainConfig, env):
    """Build MaskablePPO (MaskableActorCriticPolicy) or MaskableDQN on `env`.

    The device in `config` is resolved ("auto" -> cuda/cpu) and written back.
    """
    MaskablePPO, MaskableDQN, MaskableActorCriticPolicy = _need_sb3()
    device = resolve_device(config.device)
    config.device = device
    # tensorboard is optional: only pass tensorboard_log when it's installed
    try:
        import tensorboard  # noqa: F401
        tb_log = config.log_dir or None
    except ImportError:
        tb_log = None
    if config.algo == "ppo":
        return MaskablePPO(
            MaskableActorCriticPolicy, env, verbose=config.verbose, seed=config.seed,
            n_steps=config.n_steps, batch_size=config.batch_size,
            n_epochs=config.n_epochs, learning_rate=config.learning_rate,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_range=config.clip_range, ent_coef=config.ent_coef,
            tensorboard_log=tb_log, device=device,
        )
    if config.algo == "dqn":
        return MaskableDQN(
            "MlpPolicy", env, verbose=config.verbose, seed=config.seed,
            buffer_size=config.buffer_size, batch_size=config.batch_size,
            learning_rate=config.learning_rate, gamma=config.gamma,
            learning_starts=config.learning_starts, train_freq=config.train_freq,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            tensorboard_log=tb_log, device=device,
        )
    raise ValueError(f"unknown algo: {config.algo}")


# ---------------------------------------------------------------------- #
# callbacks                                                              #
# ---------------------------------------------------------------------- #
def make_episode_stats_callback(log_every: int = 20, verbose: bool = True):
    """SB3 callback that reports win/draw/loss rates and mean episode reward.

    Reads SB3 rollout locals directly (`rewards`, `dones`, `infos`), so it
    works with raw Gym envs and VecEnvs, with or without Monitor wrappers.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeStatsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.episodes = 0
            self.wins = 0
            self.draws = 0
            self.losses = 0
            self.recent_rewards: list[float] = []
            self._ep_reward: list[float] = []  # per running env
            self.log_every = log_every
            self.verbose_log = verbose

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")
            infos = self.locals.get("infos", [])
            n_envs = len(infos) if infos else 1
            for i in range(n_envs):
                if rewards is not None:
                    r = float(rewards[i] if hasattr(rewards, "__len__") else rewards)
                    while len(self._ep_reward) <= i:
                        self._ep_reward.append(0.0)
                    self._ep_reward[i] += r
                done = bool(dones[i]) if dones is not None and hasattr(dones, "__len__") else bool(dones)
                if not done:
                    continue
                info = infos[i] if i < len(infos) else {}
                self.episodes += 1
                total = self._ep_reward[i] if i < len(self._ep_reward) else 0.0
                self._ep_reward[i] = 0.0
                self.recent_rewards.append(total)
                if len(self.recent_rewards) > 100:
                    self.recent_rewards.pop(0)
                winner = info.get("winner")
                if winner == "draw":
                    self.draws += 1
                elif winner in ("A", "B"):
                    self.wins += 1
                else:
                    self.losses += 1
                if self.verbose_log and self.episodes % self.log_every == 0:
                    mean_r = sum(self.recent_rewards) / max(1, len(self.recent_rewards))
                    n = max(1, self.episodes)
                    print(f"[cli-hsr] episodes={self.episodes} "
                          f"win={self.wins / n:.1%} draw={self.draws / n:.1%} "
                          f"loss={self.losses / n:.1%} mean_ep_reward={mean_r:.2f}")
            return True

    return EpisodeStatsCallback()


def build_callbacks(config: TrainConfig) -> list:
    """Callbacks wired from TrainConfig: intermediate model checkpoints."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    except ImportError:  # pragma: no cover - sb3 guaranteed by _need_sb3 upstream
        return []
    callbacks: list[BaseCallback] = []
    freq = int(getattr(config, "checkpoint_freq", 0) or 0)
    if freq > 0:
        n_envs = max(1, int(getattr(config, "n_envs", 1) or 1))
        callbacks.append(CheckpointCallback(
            save_freq=max(freq // n_envs, 1),  # callback counts env.step() calls
            save_path=str(Path(config.checkpoint_dir)),
            name_prefix=str(config.algo),
            verbose=config.verbose,
        ))
    return callbacks


# ---------------------------------------------------------------------- #
# Hugging Face Hub export                                                #
# ---------------------------------------------------------------------- #
def hf_token() -> str | None:
    """Resolve a Hugging Face token, or None when unavailable.

    Sources (first hit wins, via huggingface_hub):
      1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env var (molab Remote
         Storage injects ``HF_TOKEN`` automatically)
      2. cached ``huggingface-cli login`` token

    Never raises: a missing or broken huggingface_hub install degrades to
    None so callers can fall back to local-only export.
    """
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:  # ImportError or token-read failure
        return None


def build_model_card(meta, repo_id: str, device: str = "", hardware: str = "",
                     n_envs: int = 1, elapsed_s: float | None = None) -> str:
    """Model-card markdown for a registry bundle (uploaded as README.md)."""
    import time

    eval_rows = "".join(f"\n| {k} | {v} |" for k, v in meta.eval_stats.items()) \
        if meta.eval_stats else "\n| win_rate | not evaluated |"
    elapsed = f"{elapsed_s / 60:.1f} min" if elapsed_s else "n/a"
    extra_rows = "".join(f"\n| {k} | {v} |" for k, v in meta.extra.items()) \
        if meta.extra else ""
    return f"""---
library_name: stable-baselines3
tags:
- reinforcement-learning
- stable-baselines3
- sb3-contrib
- maskable-ppo
- honkai-star-rail
license: other
---

# {meta.name} — cli-hsr {meta.algo.upper()} agent

Self-play trained agent for the [cli-hsr](https://github.com/) Honkai: Star Rail
combat simulator, packaged as a registry bundle (`model.zip` + `metadata.json`).

## Training

| | |
|---|---|
| **Algorithm** | {meta.algo.upper()} (masked action space) |
| **Team** | {", ".join(meta.team)} |
| **Timesteps** | {meta.trained_timesteps:,} |
| **Device** | {device or "auto"}{(" (" + hardware + ")") if hardware else ""} |
| **Parallel envs** | {n_envs} |
| **Wall time** | {elapsed} |
| **Trained at** | {meta.created_at} |
| **Level / AV / Rounds** | {meta.level} / {meta.max_av:.0f} / {meta.max_rounds} |{extra_rows}

## Evaluation

| metric | value |{eval_rows}

## Usage

```python
from cli_hsr.registry import load_model_bundle, CheckpointAgent
from cli_hsr.agents import run_battle_between, GreedyAgent

model, meta = load_model_bundle("checkpoints/{meta.name}")
agent = CheckpointAgent(model, meta.name)
battle = run_battle_between(agent, GreedyAgent(0), meta.team, ["kafka", "himeko"])
print(battle.winner)
```

Repo: `{repo_id}` · exported {time.strftime("%Y-%m-%d %H:%M:%S")}
"""


def hf_upload_bundle(bundle_path, repo_id: str | None, *, token: str | None = None,
                     include_checkpoints: bool = False, checkpoint_dir=None,
                     private: bool = True, commit_message: str | None = None,
                     card=None, verbose: bool = False) -> dict:
    """Upload a registry bundle to the Hugging Face Hub (best-effort).

    Creates `repo_id` as a **private** model repo when missing
    (`create_repo(..., private=True, exist_ok=True)`) and uploads
    `model.zip`, `metadata.json` (+ optional model card and intermediate
    checkpoints). Never raises on auth/network problems: callers get a
    status dict and can fall back to local export.

    Returns ``{"status": "ok"|"skipped"|"error", "reason": str|None,
    "url": str|None, "files": int}``.
    """
    result: dict = {"status": "skipped", "reason": None, "url": None, "files": 0}
    repo_id = (repo_id or "").strip()
    if not repo_id:
        result["reason"] = "no-repo-id"
        return result

    bundle_path = Path(bundle_path)
    if not (bundle_path / "model.zip").exists():
        result["reason"] = f"bundle not found: {bundle_path}"
        return result

    tok = token if token is not None else hf_token()
    if not tok:
        result["reason"] = ("no-token (set HF_TOKEN or run `huggingface-cli login`)"
                            )
        return result

    try:
        from huggingface_hub import HfApi
    except ImportError:
        result["reason"] = "huggingface_hub not installed (pip install huggingface_hub)"
        return result

    try:
        # optional model card next to the bundle
        card_path = bundle_path / "README.md"
        if card:
            card_path.write_text(card, encoding="utf-8")

        api = HfApi(token=tok)
        api.create_repo(repo_id, private=private, exist_ok=True)
        upload_kwargs: dict[str, Any] = {}
        if not include_checkpoints:
            # intermediate SB3 checkpoints are named "<prefix>_<N>_steps.zip"
            upload_kwargs["ignore_patterns"] = ["*_steps.zip"]
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(bundle_path),
            path_in_repo="",
            commit_message=commit_message or f"Upload {bundle_path.name} bundle (cli-hsr)",
            **upload_kwargs,
        )
        files = sum(1 for p in bundle_path.iterdir()
                    if p.is_file() and (include_checkpoints
                                        or not p.name.endswith("_steps.zip")))

        ckpts = Path(checkpoint_dir) if checkpoint_dir else bundle_path.parent / "checkpoints"
        if include_checkpoints and ckpts.is_dir():
            same_dir = ckpts.resolve() == bundle_path.resolve()
            if same_dir:  # checkpoints live inside the bundle dir (moLab layout)
                ckpt_names = [p for p in ckpts.iterdir()
                              if p.is_file() and p.name.endswith("_steps.zip")]
                ckpt_kwargs: dict[str, Any] = {"allow_patterns": ["*_steps.zip"]}
            else:
                ckpt_names = [p for p in ckpts.iterdir() if p.is_file()]
                ckpt_kwargs = {}
            if ckpt_names:
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(ckpts),
                    path_in_repo="checkpoints",
                    commit_message="Upload intermediate checkpoints",
                    **ckpt_kwargs,
                )
                files += len(ckpt_names)

        result.update(status="ok", url=f"https://huggingface.co/{repo_id}", files=files)
        if verbose:
            print(f"[hf] uploaded {files} file(s) -> {result['url']} (private={private})")
        return result
    except Exception as e:  # network, auth, or API failure -> graceful fallback
        result["status"] = "error"
        result["reason"] = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"[hf] upload failed: {result['reason']}")
        return result


# ---------------------------------------------------------------------- #
# public training entry points                                           #
# ---------------------------------------------------------------------- #
def train(team_a: list[str], team_b: list[str], config: TrainConfig | None = None,
          opponent_model: TrainablePolicy | None = None, env_kwargs: dict | None = None,
          callback=None):
    """Train MaskablePPO or MaskableDQN on one matchup.

    Returns (model, env). Save with `model.save(path)` or package with
    `cli_hsr.registry.save_model_bundle`.
    """
    cfg = config or TrainConfig()
    MaskablePPO, _, _ = _need_sb3()
    from ..env import HSREnv

    resolved_device = resolve_device(cfg.device)
    cfg.device = resolved_device
    if cfg.verbose:
        info = gpu_info()
        where = f"{info.get('name', 'GPU')} ({info.get('vram_gb', '?')} GB)" \
            if info.get("device") == "cuda" else "CPU"
        print(f"[cli-hsr] device={resolved_device} ({where}) | "
              f"algo={cfg.algo} | steps={cfg.total_timesteps:,} | "
              f"checkpoints -> {cfg.checkpoint_dir} every {cfg.checkpoint_freq:,}")

    opp = SelfPlayOpponent(model=opponent_model,
                           probs=(0.4, 0.6 - cfg.self_play_prob, cfg.self_play_prob),
                           seed=cfg.seed)

    def opponent_policy(battle, actor, legal):
        return opp.sample()(battle, actor, legal)

    kwargs = dict(env_kwargs or {})

    def env_factory(env_index: int = 0):
        return HSREnv(
            team_a=team_a, team_b=team_b,
            opponent_policy=opponent_policy,
            level=kwargs.get("level", 80),
            max_av=kwargs.get("max_av", 3000.0),
            max_rounds=kwargs.get("max_rounds", 80),
            reward_shaper=RewardShaper(RewardConfig()),
            seed=cfg.seed + env_index,
        )

    if cfg.n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda i=i: env_factory(i) for i in range(cfg.n_envs)])
    else:
        env = env_factory()

    model = make_model(cfg, env)
    callbacks = build_callbacks(cfg)
    if cfg.verbose:
        callbacks.append(make_episode_stats_callback())
    if callback is not None:
        callbacks.append(callback)
    learn_kwargs: dict[str, Any] = {"total_timesteps": cfg.total_timesteps}
    if callbacks:
        learn_kwargs["callback"] = callbacks if len(callbacks) > 1 else callbacks[0]
    model.learn(**learn_kwargs)
    return model, env


def evaluate_policy(model: TrainablePolicy, team_a: list[str], team_b: list[str],
                    n_games: int = 50, agent_side: str = "A",
                    opponent: str = "greedy", seed: int = 1234,
                    env_kwargs: dict | None = None) -> dict[str, float]:
    """Play n_games with the trained model (deterministic) and report stats."""
    from ..env import HSREnv

    opponent_policy: Callable
    if opponent == "greedy":
        agent = GreedyAgent()
        opponent_policy = lambda b, a, legal: agent.choose(b, a, legal)  # noqa: E731
    else:
        opponent_policy = random_action

    kwargs = dict(env_kwargs or {})
    env = HSREnv(
        team_a=team_a, team_b=team_b, agent_side=agent_side,
        opponent_policy=opponent_policy,
        level=kwargs.get("level", 80),
        max_av=kwargs.get("max_av", 3000.0),
        max_rounds=kwargs.get("max_rounds", 80),
        seed=seed,
    )
    wins = draws = losses = 0
    for g in range(n_games):
        obs, _ = env.reset(seed=seed + g)
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            done = term or trunc
        if info["winner"] == agent_side:
            wins += 1
        elif info["winner"] == "draw":
            draws += 1
        else:
            losses += 1
    n = max(1, wins + draws + losses)
    return {"win_rate": wins / n, "draw_rate": draws / n, "loss_rate": losses / n,
            "wins": wins, "draws": draws, "losses": losses}
